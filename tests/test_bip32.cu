/*
 * test_bip32.cu - BIP39 seeds and BIP32 private derivation against OpenSSL.
 *
 * OpenSSL has no BIP32, but every step of it is an OpenSSL primitive, so the
 * reference implementation below is written with HMAC(), BN_mod_add(),
 * EC_POINT_mul() and EC_POINT_point2oct(). Random seeds and random paths
 * (hardened and normal indexes, depth 0 to 6) are derived on the GPU and on
 * the host and compared with that reference. Half of the seeds come from
 * random mnemonics through bip39_seed(), checked against PKCS5_PBKDF2_HMAC().
 * The published BIP32 and BIP39 test vectors are checked first.
 */
#include <openssl/evp.h>
#include <openssl/hmac.h>

#include "test_util.h"

#define MAX_DEPTH 6
#define MAX_MNEMONIC 256
#define MAX_PASSPHRASE 64

struct HdIn {
    char mnemonic[MAX_MNEMONIC];
    char passphrase[MAX_PASSPHRASE];
    uint8_t seed[64];
    int mnemonic_len, passphrase_len, seed_len;
    int seed_from_mnemonic; /* 1: the BIP32 seed is bip39_seed(mnemonic, passphrase) */
    uint32_t path[MAX_DEPTH];
    int depth;
};

struct HdOut {
    uint8_t seed[64];
    bip32_key_t master, node;
    uint8_t pub[33];
    int seed_ok, master_ok, node_ok, pub_ok;
};

__constant__ ec_group_t c_group;

__host__ __device__ static void compute(const ec_group_t *g, const HdIn *in, HdOut *out)
{
    out->seed_ok = bip39_seed(in->mnemonic, in->mnemonic_len, in->passphrase, in->passphrase_len, out->seed);
    const uint8_t *seed = in->seed_from_mnemonic ? out->seed : in->seed;
    int seed_len = in->seed_from_mnemonic ? 64 : in->seed_len;
    out->master_ok = bip32_master(g, &out->master, seed, seed_len);
    out->node = out->master;
    out->node_ok = bip32_derive_path(g, &out->node, in->path, in->depth);
    out->pub_ok = bip32_pubkey(g, &out->node, out->pub);
}

__global__ void hd_kernel(const HdIn *in, HdOut *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        compute(&c_group, &in[i], &out[i]);
}

/* ---- reference implementation on OpenSSL primitives --------------------- */

struct RefKey {
    uint8_t key[32], cc[32];
};

static bool ref_master(Ossl &o, const uint8_t *seed, int len, RefKey &k)
{
    uint8_t I[64];
    unsigned int l;
    HMAC(EVP_sha512(), "Bitcoin seed", 12, seed, len, I, &l);
    memcpy(k.key, I, 32);
    memcpy(k.cc, I + 32, 32);
    BIGNUM *b = BN_bin2bn(I, 32, NULL);
    bool ok = !BN_is_zero(b) && BN_cmp(b, o.order) < 0;
    BN_free(b);
    return ok;
}

static bool ref_pub(Ossl &o, const RefKey &k, uint8_t out[33])
{
    BIGNUM *d = BN_bin2bn(k.key, 32, NULL);
    EC_POINT *P = EC_POINT_new(o.group);
    bool ok = EC_POINT_mul(o.group, P, d, NULL, NULL, o.ctx)
              && EC_POINT_point2oct(o.group, P, POINT_CONVERSION_COMPRESSED, out, 33, o.ctx) == 33;
    EC_POINT_free(P);
    BN_free(d);
    return ok;
}

static bool ref_ckd(Ossl &o, const RefKey &p, uint32_t index, RefKey &c)
{
    uint8_t data[37], I[64];
    unsigned int l;
    if (index & BIP32_HARDENED) {
        data[0] = 0;
        memcpy(data + 1, p.key, 32);
    } else if (!ref_pub(o, p, data)) {
        return false;
    }
    data[33] = (uint8_t)(index >> 24);
    data[34] = (uint8_t)(index >> 16);
    data[35] = (uint8_t)(index >> 8);
    data[36] = (uint8_t)index;
    HMAC(EVP_sha512(), p.cc, 32, data, 37, I, &l);
    BIGNUM *il = BN_bin2bn(I, 32, NULL), *k = BN_bin2bn(p.key, 32, NULL);
    bool ok = BN_cmp(il, o.order) < 0 && BN_mod_add(k, il, k, o.order, o.ctx) && !BN_is_zero(k);
    if (ok) {
        BN_bn2binpad(k, c.key, 32);
        memcpy(c.cc, I + 32, 32);
    }
    BN_free(il);
    BN_free(k);
    return ok;
}

/* ---- helpers ------------------------------------------------------------ */

static std::string hex(const uint8_t *p, int n)
{
    static const char *digits = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < n; i++) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 15];
    }
    return s;
}

static std::string key_hex(const bip32_key_t &k)
{
    uint8_t b[32];
    bn_to_bytes_be(&k.key, b, 32);
    return hex(b, 32);
}

static std::string path_str(const uint32_t *path, int depth)
{
    std::string s = "m";
    for (int i = 0; i < depth; i++) {
        s += "/" + std::to_string(path[i] & ~BIP32_HARDENED);
        if (path[i] & BIP32_HARDENED)
            s += "'";
    }
    return s;
}

static bool same_key(const bip32_key_t &k, const RefKey &r)
{
    return key_hex(k) == hex(r.key, 32) && memcmp(k.chain_code, r.cc, 32) == 0;
}

static int from_hex_bytes(const char *h, uint8_t *out)
{
    int n = (int)strlen(h) / 2;
    for (int i = 0; i < n; i++)
        sscanf(h + 2 * i, "%2hhx", &out[i]);
    return n;
}

/* ---- published vectors -------------------------------------------------- */

static void check_vector(Stats &st, const ec_group_t *g, const char *seed_hex, const char *path, const char *priv,
                         const char *cc, const char *pub)
{
    uint8_t seed[64], pubkey[33];
    uint32_t idx[MAX_DEPTH];
    int len = from_hex_bytes(seed_hex, seed);
    int depth = bip32_parse_path(path, idx, MAX_DEPTH);
    bip32_key_t k;
    int ok = depth >= 0 && bip32_master(g, &k, seed, len) && bip32_derive_path(g, &k, idx, depth)
             && bip32_pubkey(g, &k, pubkey);
    st.check(ok && key_hex(k) == priv && hex(k.chain_code, 32) == cc && hex(pubkey, 33) == pub,
             "bip32 vector %s: key %s cc %s pub %s", path, key_hex(k).c_str(), hex(k.chain_code, 32).c_str(),
             hex(pubkey, 33).c_str());
}

static void known_answers(Stats &st, const ec_group_t *g)
{
    /* BIP32 test vector 1 */
    const char *s1 = "000102030405060708090a0b0c0d0e0f";
    check_vector(st, g, s1, "m", "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35",
                 "873dff81c02f525623fd1fe5167eac3a55a049de3d314bb42ee227ffed37d508",
                 "0339a36013301597daef41fbe593a02cc513d0b55527ec2df1050e2e8ff49c85c2");
    check_vector(st, g, s1, "m/0'", "edb2e14f9ee77d26dd93b4ecede8d16ed408ce149b6cd80b0715a2d911a0afea",
                 "47fdacbd0f1097043b78c63c20c34ef4ed9a111d980047ad16282c7ae6236141",
                 "035a784662a4a20a65bf6aab9ae98a6c068a81c52e4b032c0fb5400c706cfccc56");
    check_vector(st, g, s1, "m/0'/1", "3c6cb8d0f6a264c91ea8b5030fadaa8e538b020f0a387421a12de9319dc93368",
                 "2a7857631386ba23dacac34180dd1983734e444fdbf774041578e9b6adb37c19",
                 "03501e454bf00751f24b1b489aa925215d66af2234e3891c3b21a52bedb3cd711c");
    check_vector(st, g, s1, "m/0'/1/2'", "cbce0d719ecf7431d88e6a89fa1483e02e35092af60c042b1df2ff59fa424dca",
                 "04466b9cc8e161e966409ca52986c584f07e9dc81f735db683c3ff6ec7b1503f",
                 "0357bfe1e341d01c69fe5654309956cbea516822fba8a601743a012a7896ee8dc2");
    check_vector(st, g, s1, "m/0'/1/2'/2", "0f479245fb19a38a1954c5c7c0ebab2f9bdfd96a17563ef28a6a4b1a2a764ef4",
                 "cfb71883f01676f587d023cc53a35bc7f88f724b1f8c2892ac1275ac822a3edd",
                 "02e8445082a72f29b75ca48748a914df60622a609cacfce8ed0e35804560741d29");
    check_vector(st, g, s1, "m/0'/1/2'/2/1000000000",
                 "471b76e389e528d6de6d816857e012c5455051cad6660850e58372a6c3e6e7c8",
                 "c783e67b921d2beb8f6b389cc646d7263b4145701dadd2161548a8b078e65e9e",
                 "022a471424da5e657499d1ff51cb43c47481a03b1e77f951fe64cec9f5a48f7011");

    /* BIP32 test vector 2 */
    const char *s2 = "fffcf9f6f3f0edeae7e4e1dedbd8d5d2cfccc9c6c3c0bdbab7b4b1aeaba8a5a2"
                     "9f9c999693908d8a8784817e7b7875726f6c696663605d5a5754514e4b484542";
    check_vector(st, g, s2, "m", "4b03d6fc340455b363f51020ad3ecca4f0850280cf436c70c727923f6db46c3e",
                 "60499f801b896d83179a4374aeb7822aaeaceaa0db1f85ee3e904c4defbd9689",
                 "03cbcaa9c98c877a26977d00825c956a238e8dddfbd322cce4f74b0b5bd6ace4a7");
    check_vector(st, g, s2, "m/0", "abe74a98f6c7eabee0428f53798f0ab8aa1bd37873999041703c742f15ac7e1e",
                 "f0909affaa7ee7abe5dd4e100598d4dc53cd709d5a5c2cac40e7412f232f7c9c",
                 "02fc9e5af0ac8d9b3cecfe2a888e2117ba3d089d8585886c9c826b6b22a98d12ea");

    /* BIP39 test vector 1 (passphrase "TREZOR") */
    const char *m = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    uint8_t seed[64];
    bip39_seed(m, strlen(m), "TREZOR", 6, seed);
    st.check(hex(seed, 64) == "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e5349553"
                              "1f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04",
             "bip39 seed: %s", hex(seed, 64).c_str());

    /* path parser */
    uint32_t p[MAX_DEPTH];
    st.check(bip32_parse_path("m/44'/0h/0'/0/2147483647", p, MAX_DEPTH) == 5 && p[0] == (44 | BIP32_HARDENED)
                 && p[1] == BIP32_HARDENED && p[3] == 0 && p[4] == 2147483647u,
             "parse m/44'/0h/0'/0/2147483647");
    st.check(bip32_parse_path("m", p, MAX_DEPTH) == 0 && bip32_parse_path("0/1", p, MAX_DEPTH) == 2,
             "parse m and 0/1");
    st.check(bip32_parse_path("m/2147483648", p, MAX_DEPTH) < 0 && bip32_parse_path("m/0/", p, MAX_DEPTH) < 0
                 && bip32_parse_path("m//0", p, MAX_DEPTH) < 0 && bip32_parse_path("x/0", p, MAX_DEPTH) < 0
                 && bip32_parse_path("m/0/0/0/0/0/0/0", p, MAX_DEPTH) < 0,
             "parser rejects malformed paths");
}

/* ---- random inputs ------------------------------------------------------ */

static std::vector<HdIn> make_inputs()
{
    std::vector<HdIn> v;
    const int seed_lens[] = { 16, 20, 32, 48, 64 };
    const uint32_t special[] = { 0, 1, 0x7fffffffu, BIP32_HARDENED, 0xffffffffu, 1000000000u,
                                 44 | BIP32_HARDENED, 60 | BIP32_HARDENED, 9000 | BIP32_HARDENED };
    int n = batch_size();

    for (int i = 0; i < n; i++) {
        HdIn in;
        memset(&in, 0, sizeof(in));

        in.mnemonic_len = 12 + (int)(rng()() % (MAX_MNEMONIC - 12));
        for (int j = 0; j < in.mnemonic_len; j++)
            in.mnemonic[j] = (rng()() % 7 == 0) ? ' ' : (char)('a' + rng()() % 26);
        in.passphrase_len = (i % 3 == 0) ? 0 : (int)(rng()() % (MAX_PASSPHRASE + 1));
        for (int j = 0; j < in.passphrase_len; j++)
            in.passphrase[j] = (char)(0x21 + rng()() % 94);

        in.seed_from_mnemonic = i % 2;
        in.seed_len = seed_lens[rng()() % 5];
        for (int j = 0; j < in.seed_len; j++)
            in.seed[j] = (uint8_t)rng()();

        in.depth = (int)(rng()() % (MAX_DEPTH + 1));
        for (int j = 0; j < in.depth; j++) {
            uint64_t r = rng()() % 10;
            if (r < 4)
                in.path[j] = (uint32_t)(rng()() % BIP32_HARDENED);
            else if (r < 7)
                in.path[j] = (uint32_t)(rng()() % BIP32_HARDENED) | BIP32_HARDENED;
            else
                in.path[j] = special[rng()() % 9];
        }
        if (i == 1)
            in.depth = 0;
        if (i == 2) { /* the usual wallet path */
            in.depth = bip32_parse_path("m/44'/0'/0'/0/0", in.path, MAX_DEPTH);
        }
        v.push_back(in);
    }
    return v;
}

int main()
{
    Stats st;
    Ossl o;
    ec_group_t group;
    ec_group_init_secp256k1(&group);
    CUDA_CHECK(cudaMemcpyToSymbol(c_group, &group, sizeof(group)));

    known_answers(st, &group);

    std::vector<HdIn> in = make_inputs();
    std::vector<HdOut> out = run_on_device<HdIn, HdOut>(in, [](const HdIn *d_in, HdOut *d_out, int n) {
        hd_kernel<<<(n + 127) / 128, 128>>>(d_in, d_out, n);
    });

    for (size_t i = 0; i < in.size(); i++) {
        const HdIn &x = in[i];
        std::string path = path_str(x.path, x.depth);

        HdOut host;
        compute(&group, &x, &host);

        /* BIP39 seed against OpenSSL's PBKDF2 */
        uint8_t ref_seed[64];
        std::string salt = "mnemonic" + std::string(x.passphrase, x.passphrase_len);
        PKCS5_PBKDF2_HMAC(x.mnemonic, x.mnemonic_len, (const uint8_t *)salt.data(), (int)salt.size(), 2048,
                          EVP_sha512(), 64, ref_seed);
        st.check(out[i].seed_ok && memcmp(out[i].seed, ref_seed, 64) == 0, "[%zu] gpu bip39 seed: %s != %s", i,
                 hex(out[i].seed, 64).c_str(), hex(ref_seed, 64).c_str());
        st.check(host.seed_ok && memcmp(host.seed, ref_seed, 64) == 0, "[%zu] host bip39 seed: %s != %s", i,
                 hex(host.seed, 64).c_str(), hex(ref_seed, 64).c_str());

        /* BIP32 chain against the OpenSSL-based reference */
        RefKey rm, rn;
        const uint8_t *seed = x.seed_from_mnemonic ? ref_seed : x.seed;
        int seed_len = x.seed_from_mnemonic ? 64 : x.seed_len;
        if (!ref_master(o, seed, seed_len, rm)) {
            printf("[%zu] note: seed gives an invalid master key, skipped\n", i);
            continue;
        }
        rn = rm;
        bool chain_ok = true;
        for (int d = 0; d < x.depth && chain_ok; d++)
            chain_ok = ref_ckd(o, rn, x.path[d], rn);
        if (!chain_ok) {
            printf("[%zu] note: path %s hits an invalid child, skipped\n", i, path.c_str());
            continue;
        }
        uint8_t ref_pubkey[33];
        ref_pub(o, rn, ref_pubkey);

        const HdOut *res[] = { &out[i], &host };
        const char *who[] = { "gpu", "host" };
        for (int w = 0; w < 2; w++) {
            const HdOut &r = *res[w];
            st.check(r.master_ok && same_key(r.master, rm), "[%zu] %s master: key %s cc %s != %s %s", i, who[w],
                     key_hex(r.master).c_str(), hex(r.master.chain_code, 32).c_str(), hex(rm.key, 32).c_str(),
                     hex(rm.cc, 32).c_str());
            st.check(r.node_ok && same_key(r.node, rn), "[%zu] %s %s: key %s cc %s != %s %s", i, who[w],
                     path.c_str(), key_hex(r.node).c_str(), hex(r.node.chain_code, 32).c_str(),
                     hex(rn.key, 32).c_str(), hex(rn.cc, 32).c_str());
            st.check(r.pub_ok && memcmp(r.pub, ref_pubkey, 33) == 0, "[%zu] %s %s pub: %s != %s", i, who[w],
                     path.c_str(), hex(r.pub, 33).c_str(), hex(ref_pubkey, 33).c_str());
        }
    }
    return st.finish("test_bip32");
}
