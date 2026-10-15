/*
 * test_address.cu - Base58Check, Bech32 and the address formats.
 *
 * Base58Check is compared with a reference built on OpenSSL: sha256d via
 * SHA256() and the base conversion via BN_div_word(). Bech32 is checked
 * against the BIP173 test vectors and, for random inputs, by an independent
 * decoder in this file that verifies the checksum and recovers the data.
 * P2PKH, P2WPKH and Avalanche addresses of random public keys are checked
 * the same way with the hash160 taken from OpenSSL (SHA256() and
 * RIPEMD160()). Everything runs on the GPU and on the host.
 */
/*
 * RIPEMD160() is a deprecated low-level call in OpenSSL 3, but unlike
 * EVP_ripemd160() it works without loading the legacy provider.
 */
#define OPENSSL_SUPPRESS_DEPRECATED
#include <openssl/ripemd.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

#include "test_util.h"

#define MAX_PAYLOAD BASE58CHECK_MAX_PAYLOAD
#define MAX_HRP 10
#define MAX_DATA 60

struct AddrIn {
    uint8_t payload[MAX_PAYLOAD];
    int payload_len;
    uint8_t pub[33];
    char hrp[MAX_HRP + 1];
    uint8_t data[MAX_DATA];
    int data_len;
};

struct AddrOut {
    char b58[BASE58CHECK_MAX_LEN];
    char p2pkh[64];
    char p2wpkh[BECH32_MAX_LEN + 1];
    char avax[BECH32_MAX_LEN + 1];
    char bech[BECH32_MAX_LEN + 1];
    int b58_len, p2pkh_len, p2wpkh_len, avax_len, bech_len;
};

__host__ __device__ static void compute(const AddrIn *in, AddrOut *out)
{
    out->b58_len = base58check_encode(in->payload, in->payload_len, out->b58, sizeof(out->b58));
    out->p2pkh_len = address_p2pkh(0x00, in->pub, out->p2pkh, sizeof(out->p2pkh));
    out->p2wpkh_len = address_p2wpkh("bc", in->pub, out->p2wpkh, sizeof(out->p2wpkh));
    out->avax_len = address_avalanche("avax", in->pub, out->avax, sizeof(out->avax));
    out->bech_len = bech32_encode(in->hrp, in->data, in->data_len, out->bech, sizeof(out->bech));
}

__global__ void addr_kernel(const AddrIn *in, AddrOut *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        compute(&in[i], &out[i]);
}

/* ---- references --------------------------------------------------------- */

static const char *B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const char *B32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

static std::string ref_base58check(const uint8_t *payload, int len)
{
    uint8_t buf[MAX_PAYLOAD + 4], h[32];
    memcpy(buf, payload, len);
    SHA256(payload, len, h);
    SHA256(h, 32, h);
    memcpy(buf + len, h, 4);
    len += 4;

    std::string s;
    BIGNUM *x = BN_bin2bn(buf, len, NULL);
    while (!BN_is_zero(x))
        s.insert(s.begin(), B58[BN_div_word(x, 58)]);
    BN_free(x);
    for (int i = 0; i < len && buf[i] == 0; i++)
        s.insert(s.begin(), '1');
    return s;
}

/* BIP173 checksum verification written the straightforward way (generator table and loop). */
static uint32_t ref_polymod(const std::vector<uint8_t> &values)
{
    static const uint32_t gen[5] = { 0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3 };
    uint32_t chk = 1;
    for (uint8_t v : values) {
        uint32_t b = chk >> 25;
        chk = ((chk & 0x1ffffff) << 5) ^ v;
        for (int i = 0; i < 5; i++)
            if ((b >> i) & 1)
                chk ^= gen[i];
    }
    return chk;
}

/* Split a bech32 string, verify its checksum, return hrp and data (without checksum). */
static bool ref_bech32_decode(const std::string &s, std::string &hrp, std::vector<uint8_t> &data)
{
    size_t pos = s.rfind('1');
    if (pos == std::string::npos || pos == 0 || pos + 7 > s.size() || s.size() > 90)
        return false;
    hrp = s.substr(0, pos);
    std::vector<uint8_t> values;
    for (char c : hrp)
        values.push_back((uint8_t)(c >> 5));
    values.push_back(0);
    for (char c : hrp)
        values.push_back((uint8_t)(c & 31));
    data.clear();
    for (size_t i = pos + 1; i < s.size(); i++) {
        const char *p = strchr(B32, s[i]);
        if (!p || s[i] == '\0')
            return false;
        values.push_back((uint8_t)(p - B32));
        data.push_back((uint8_t)(p - B32));
    }
    if (ref_polymod(values) != 1)
        return false;
    data.resize(data.size() - 6);
    return true;
}

static std::vector<uint8_t> ref_5to8(const std::vector<uint8_t> &in)
{
    std::vector<uint8_t> out;
    uint32_t acc = 0;
    int bits = 0;
    for (uint8_t v : in) {
        acc = (acc << 5) | v;
        bits += 5;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)((acc >> bits) & 0xff));
        }
    }
    return out;
}

static std::vector<uint8_t> ref_hash160(const uint8_t *p, int n)
{
    std::vector<uint8_t> h(20);
    uint8_t s[32];
    SHA256(p, n, s);
    RIPEMD160(s, 32, h.data());
    return h;
}

/* Decode a bech32 address and check hrp, optional witness version and the hash160 payload. */
static bool ref_check_bech32_address(const std::string &addr, const char *hrp, int witness_version,
                                     const std::vector<uint8_t> &h160)
{
    std::string got_hrp;
    std::vector<uint8_t> data;
    if (!ref_bech32_decode(addr, got_hrp, data) || got_hrp != hrp)
        return false;
    if (witness_version >= 0) {
        if (data.empty() || data[0] != witness_version)
            return false;
        data.erase(data.begin());
    }
    return ref_5to8(data) == h160;
}

/* ---- published vectors -------------------------------------------------- */

static void known_answers(Stats &st)
{
    char out[BECH32_MAX_LEN + 1];

    /* BIP173 valid bech32 strings: decode the data part, re-encode, compare. */
    const char *vectors[] = {
        "a12uel5l",
        "an83characterlonghumanreadablepartthatcontainsthenumber1andtheexcludedcharactersbio1tt5tgs",
        "abcdef1qpzry9x8gf2tvdw0s3jn54khce6mua7lmqqqxw",
        "11qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqc8247j",
        "split1checkupstagehandshakeupstreamerranterredcaperred2y9e3w",
        "?1ezyfcl",
    };
    for (const char *v : vectors) {
        std::string hrp;
        std::vector<uint8_t> data;
        bool ok = ref_bech32_decode(v, hrp, data);
        int n = ok ? bech32_encode(hrp.c_str(), data.data(), data.size(), out, sizeof(out)) : 0;
        st.check(n > 0 && std::string(out) == v, "bech32 vector %s: got %s", v, n ? out : "(error)");
    }

    /* Private key 1: the generator point, in every format. */
    const uint8_t g[33] = { 0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0,
                            0x62, 0x95, 0xce, 0x87, 0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d,
                            0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98 };
    address_p2pkh(0x00, g, out, sizeof(out));
    st.check(!strcmp(out, "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"), "p2pkh(G) = %s", out);
    address_p2pkh(0x6f, g, out, sizeof(out));
    st.check(!strcmp(out, "mrCDrCybB6J1vRfbwM5hemdJz73FwDBC8r"), "p2pkh testnet(G) = %s", out);
    address_p2wpkh("bc", g, out, sizeof(out));
    st.check(!strcmp(out, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"), "p2wpkh(G) = %s", out);
    address_p2wpkh("tb", g, out, sizeof(out));
    st.check(!strcmp(out, "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx"), "p2wpkh testnet(G) = %s", out);

    /* Wallet import format of private key 1: Base58Check with longer payloads. */
    uint8_t wif[34] = { 0x80 };
    wif[32] = 0x01;
    wif[33] = 0x01;
    base58check_encode(wif, 33, out, sizeof(out));
    st.check(!strcmp(out, "5HpHagT65TZzG1PH3CSu63k8DbpvD8s5ip4nEB3kEsreAnchuDf"), "wif uncompressed = %s", out);
    base58check_encode(wif, 34, out, sizeof(out));
    st.check(!strcmp(out, "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn"), "wif compressed = %s", out);

    /* Rejections */
    uint8_t bad[1] = { 32 };
    st.check(bech32_encode("bc", bad, 1, out, sizeof(out)) == 0, "bech32 rejects data >= 32");
    st.check(bech32_encode("BC", bad, 0, out, sizeof(out)) == 0, "bech32 rejects uppercase hrp");
    st.check(bech32_encode("", bad, 0, out, sizeof(out)) == 0, "bech32 rejects empty hrp");
    st.check(address_p2pkh(0x00, g, out, 34) == 0 && address_p2pkh(0x00, g, out, 35) == 34,
             "p2pkh reports a too-small buffer");
    st.check(base58check_encode(wif, MAX_PAYLOAD + 1, out, sizeof(out)) == 0, "base58check rejects long payload");
}

/* ---- random inputs ------------------------------------------------------ */

static std::vector<AddrIn> make_inputs(Ossl &o)
{
    std::vector<AddrIn> v;
    int n = batch_size();
    EC_POINT *P = EC_POINT_new(o.group);
    BIGNUM *k = BN_new();

    for (int i = 0; i < n; i++) {
        AddrIn in;
        memset(&in, 0, sizeof(in));

        in.payload_len = 1 + (int)(rng()() % MAX_PAYLOAD);
        int zeros = (int)(rng()() % 4);
        for (int j = 0; j < in.payload_len; j++)
            in.payload[j] = j < zeros ? 0 : (uint8_t)rng()();
        if (i % 16 == 0)
            in.payload[0] = 0; /* version 0x00 like mainnet P2PKH, or a leading zero in the middle of a run */

        BN_rand_range(k, o.order);
        if (BN_is_zero(k))
            BN_one(k);
        EC_POINT_mul(o.group, P, k, NULL, NULL, o.ctx);
        EC_POINT_point2oct(o.group, P, POINT_CONVERSION_COMPRESSED, in.pub, 33, o.ctx);

        int hrp_len = 1 + (int)(rng()() % MAX_HRP);
        for (int j = 0; j < hrp_len; j++) {
            int c = 0x21 + (int)(rng()() % 94);
            if (c >= 'A' && c <= 'Z')
                c += 'a' - 'A';
            in.hrp[j] = (char)c;
        }
        in.data_len = (int)(rng()() % (MAX_DATA + 1));
        for (int j = 0; j < in.data_len; j++)
            in.data[j] = (uint8_t)(rng()() % 32);
        v.push_back(in);
    }
    BN_free(k);
    EC_POINT_free(P);
    return v;
}

int main()
{
    Stats st;
    Ossl o;
    known_answers(st);

    std::vector<AddrIn> in = make_inputs(o);
    std::vector<AddrOut> out = run_on_device<AddrIn, AddrOut>(in, [](const AddrIn *d_in, AddrOut *d_out, int n) {
        addr_kernel<<<(n + 127) / 128, 128>>>(d_in, d_out, n);
    });

    for (size_t i = 0; i < in.size(); i++) {
        const AddrIn &x = in[i];
        AddrOut host;
        compute(&x, &host);

        std::string ref_b58 = ref_base58check(x.payload, x.payload_len);
        std::vector<uint8_t> h160 = ref_hash160(x.pub, 33);
        uint8_t p2pkh_payload[21] = { 0x00 };
        memcpy(p2pkh_payload + 1, h160.data(), 20);
        std::string ref_p2pkh = ref_base58check(p2pkh_payload, 21);

        const AddrOut *res[] = { &out[i], &host };
        const char *who[] = { "gpu", "host" };
        for (int w = 0; w < 2; w++) {
            const AddrOut &r = *res[w];
            st.check(r.b58_len == (int)ref_b58.size() && ref_b58 == r.b58, "[%zu] %s base58check(%d bytes): %s != %s",
                     i, who[w], x.payload_len, r.b58, ref_b58.c_str());
            st.check(r.p2pkh_len == (int)ref_p2pkh.size() && ref_p2pkh == r.p2pkh, "[%zu] %s p2pkh: %s != %s", i,
                     who[w], r.p2pkh, ref_p2pkh.c_str());
            st.check(r.p2wpkh_len == (int)strlen(r.p2wpkh) && ref_check_bech32_address(r.p2wpkh, "bc", 0, h160),
                     "[%zu] %s p2wpkh: %s", i, who[w], r.p2wpkh);
            st.check(r.avax_len == (int)strlen(r.avax) && ref_check_bech32_address(r.avax, "avax", -1, h160),
                     "[%zu] %s avalanche: %s", i, who[w], r.avax);

            std::string hrp;
            std::vector<uint8_t> data;
            bool ok = r.bech_len == (int)strlen(r.bech) && ref_bech32_decode(r.bech, hrp, data) && hrp == x.hrp
                      && data == std::vector<uint8_t>(x.data, x.data + x.data_len);
            st.check(ok, "[%zu] %s bech32 hrp=%s len=%d: %s", i, who[w], x.hrp, x.data_len, r.bech);
        }
        st.check(!strcmp(out[i].b58, host.b58) && !strcmp(out[i].p2wpkh, host.p2wpkh) && !strcmp(out[i].avax, host.avax)
                     && !strcmp(out[i].bech, host.bech),
                 "[%zu] gpu and host strings differ", i);
    }
    return st.finish("test_address");
}
