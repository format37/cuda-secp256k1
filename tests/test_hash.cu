/*
 * test_hash.cu - SHA-256, SHA-512 and their HMACs against OpenSSL.
 *
 * Every message is hashed on the GPU three ways (one shot, split into two
 * updates at a chosen point, and one byte per update), the HMAC two ways
 * (one shot and split), and all of it again on the host. Each digest is
 * compared with OpenSSL's SHA256() / SHA512() / HMAC(). Message lengths
 * cover every padding case of both block sizes (0..400 bytes exhaustively)
 * and key lengths cover short, block-sized and longer-than-block keys.
 */
#include <openssl/hmac.h>
#include <openssl/sha.h>

#include "test_util.h"

#define MAX_MSG 1024
#define MAX_KEY 256

struct HashIn {
    uint8_t msg[MAX_MSG];
    uint8_t key[MAX_KEY];
    int msg_len;
    int key_len;
    int split; /* where the two-update variants cut the message */
};

struct HashOut {
    uint8_t sha256_digest[3][32]; /* one shot, split, bytewise */
    uint8_t hmac256_digest[2][32]; /* one shot, split */
    uint8_t sha512_digest[3][64];
    uint8_t hmac512_digest[2][64];
};

__host__ __device__ static void compute(const HashIn *in, HashOut *out)
{
    const uint8_t *m = in->msg;
    int n = in->msg_len, s = in->split;

    sha256(m, n, out->sha256_digest[0]);
    sha256_ctx_t c256;
    sha256_init(&c256);
    sha256_update(&c256, m, s);
    sha256_update(&c256, m + s, n - s);
    sha256_final(&c256, out->sha256_digest[1]);
    sha256_init(&c256);
    for (int i = 0; i < n; i++)
        sha256_update(&c256, m + i, 1);
    sha256_final(&c256, out->sha256_digest[2]);

    hmac_sha256(in->key, in->key_len, m, n, out->hmac256_digest[0]);
    hmac_sha256_ctx_t h256;
    hmac_sha256_init(&h256, in->key, in->key_len);
    hmac_sha256_update(&h256, m, s);
    hmac_sha256_update(&h256, m + s, n - s);
    hmac_sha256_final(&h256, out->hmac256_digest[1]);

    sha512(m, n, out->sha512_digest[0]);
    sha512_ctx_t c512;
    sha512_init(&c512);
    sha512_update(&c512, m, s);
    sha512_update(&c512, m + s, n - s);
    sha512_final(&c512, out->sha512_digest[1]);
    sha512_init(&c512);
    for (int i = 0; i < n; i++)
        sha512_update(&c512, m + i, 1);
    sha512_final(&c512, out->sha512_digest[2]);

    hmac_sha512(in->key, in->key_len, m, n, out->hmac512_digest[0]);
    hmac_sha512_ctx_t h512;
    hmac_sha512_init(&h512, in->key, in->key_len);
    hmac_sha512_update(&h512, m, s);
    hmac_sha512_update(&h512, m + s, n - s);
    hmac_sha512_final(&h512, out->hmac512_digest[1]);
}

__global__ void hash_kernel(const HashIn *in, HashOut *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        compute(&in[i], &out[i]);
}

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

static void fill_random(uint8_t *p, int n)
{
    for (int i = 0; i < n; i++)
        p[i] = (uint8_t)rng()();
}

static std::vector<HashIn> make_inputs()
{
    std::vector<HashIn> v;
    const int key_lens[] = { 0, 1, 20, 32, 55, 56, 63, 64, 65, 100, 111, 112, 127, 128, 129, 200, MAX_KEY };
    const int n_key_lens = sizeof(key_lens) / sizeof(key_lens[0]);
    int n = batch_size();

    for (int i = 0; i < n; i++) {
        HashIn in;
        memset(&in, 0, sizeof(in));
        if (i <= 400) {
            /* Every length through three SHA-512 blocks: hits all padding branches. */
            in.msg_len = i;
            in.key_len = key_lens[i % n_key_lens];
        } else {
            in.msg_len = (i % 8 == 0) ? MAX_MSG : (int)(rng()() % (MAX_MSG + 1));
            in.key_len = (int)(rng()() % (MAX_KEY + 1));
        }
        fill_random(in.msg, in.msg_len);
        fill_random(in.key, in.key_len);
        in.split = in.msg_len ? (int)(rng()() % (in.msg_len + 1)) : 0;
        v.push_back(in);
    }
    return v;
}

/* A few published vectors, so a failure here is obvious without OpenSSL. */
static void known_answers(Stats &st)
{
    uint8_t out[64];
    const uint8_t abc[] = { 'a', 'b', 'c' };
    const uint8_t jefe[] = { 'J', 'e', 'f', 'e' };
    const char *q = "what do ya want for nothing?"; /* RFC 4231 test case 2 */
    uint8_t long_key[131];                           /* RFC 4231 test case 6 */
    const char *lk = "Test Using Larger Than Block-Size Key - Hash Key First";
    memset(long_key, 0xaa, sizeof(long_key));

    sha256(abc, 0, out);
    st.check(hex(out, 32) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
             "sha256(\"\") = %s", hex(out, 32).c_str());
    sha256(abc, 3, out);
    st.check(hex(out, 32) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
             "sha256(\"abc\") = %s", hex(out, 32).c_str());
    hmac_sha256(jefe, 4, (const uint8_t *)q, strlen(q), out);
    st.check(hex(out, 32) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
             "hmac-sha256 rfc4231 #2 = %s", hex(out, 32).c_str());
    hmac_sha256(long_key, sizeof(long_key), (const uint8_t *)lk, strlen(lk), out);
    st.check(hex(out, 32) == "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
             "hmac-sha256 rfc4231 #6 = %s", hex(out, 32).c_str());

    sha512(abc, 0, out);
    st.check(hex(out, 64) == "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
                             "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
             "sha512(\"\") = %s", hex(out, 64).c_str());
    sha512(abc, 3, out);
    st.check(hex(out, 64) == "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
                             "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
             "sha512(\"abc\") = %s", hex(out, 64).c_str());
    hmac_sha512(jefe, 4, (const uint8_t *)q, strlen(q), out);
    st.check(hex(out, 64) == "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea250554"
                             "9758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737",
             "hmac-sha512 rfc4231 #2 = %s", hex(out, 64).c_str());
    hmac_sha512(long_key, sizeof(long_key), (const uint8_t *)lk, strlen(lk), out);
    st.check(hex(out, 64) == "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f352"
                             "6b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598",
             "hmac-sha512 rfc4231 #6 = %s", hex(out, 64).c_str());
}

int main()
{
    Stats st;
    known_answers(st);

    std::vector<HashIn> in = make_inputs();
    std::vector<HashOut> out = run_on_device<HashIn, HashOut>(in, [](const HashIn *d_in, HashOut *d_out, int n) {
        hash_kernel<<<(n + 127) / 128, 128>>>(d_in, d_out, n);
    });

    for (size_t i = 0; i < in.size(); i++) {
        const HashIn &x = in[i];
        uint8_t ref256[32], ref_h256[32], ref512[64], ref_h512[64];
        unsigned int len = 0;
        SHA256(x.msg, x.msg_len, ref256);
        HMAC(EVP_sha256(), x.key, x.key_len, x.msg, x.msg_len, ref_h256, &len);
        SHA512(x.msg, x.msg_len, ref512);
        HMAC(EVP_sha512(), x.key, x.key_len, x.msg, x.msg_len, ref_h512, &len);

        HashOut host;
        compute(&x, &host);

        struct Check {
            const char *what;
            const uint8_t *dev, *cpu, *ref;
            int size;
        } checks[] = {
            { "sha256", out[i].sha256_digest[0], host.sha256_digest[0], ref256, 32 },
            { "sha256 split", out[i].sha256_digest[1], host.sha256_digest[1], ref256, 32 },
            { "sha256 bytewise", out[i].sha256_digest[2], host.sha256_digest[2], ref256, 32 },
            { "hmac-sha256", out[i].hmac256_digest[0], host.hmac256_digest[0], ref_h256, 32 },
            { "hmac-sha256 split", out[i].hmac256_digest[1], host.hmac256_digest[1], ref_h256, 32 },
            { "sha512", out[i].sha512_digest[0], host.sha512_digest[0], ref512, 64 },
            { "sha512 split", out[i].sha512_digest[1], host.sha512_digest[1], ref512, 64 },
            { "sha512 bytewise", out[i].sha512_digest[2], host.sha512_digest[2], ref512, 64 },
            { "hmac-sha512", out[i].hmac512_digest[0], host.hmac512_digest[0], ref_h512, 64 },
            { "hmac-sha512 split", out[i].hmac512_digest[1], host.hmac512_digest[1], ref_h512, 64 },
        };
        for (const Check &c : checks) {
            st.check(memcmp(c.dev, c.ref, c.size) == 0, "[%zu] gpu %s msg_len=%d key_len=%d split=%d: %s != %s", i,
                     c.what, x.msg_len, x.key_len, x.split, hex(c.dev, c.size).c_str(), hex(c.ref, c.size).c_str());
            st.check(memcmp(c.cpu, c.ref, c.size) == 0, "[%zu] host %s msg_len=%d key_len=%d split=%d: %s != %s", i,
                     c.what, x.msg_len, x.key_len, x.split, hex(c.cpu, c.size).c_str(), hex(c.ref, c.size).c_str());
        }
    }
    return st.finish("test_hash");
}
