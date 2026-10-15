/*
 * test_hash.cu - SHA-256 and HMAC-SHA256 against OpenSSL.
 *
 * Every message is hashed on the GPU three ways (one shot, split into two
 * updates at a chosen point, and one byte per update) and on the host, and
 * all four digests are compared with OpenSSL's SHA256() and HMAC().
 * Message lengths cover every padding case (0..321 bytes exhaustively) and
 * key lengths cover short, block-sized and longer-than-block keys.
 */
#include <openssl/hmac.h>
#include <openssl/sha.h>

#include "test_util.h"

#define MAX_MSG 1024
#define MAX_KEY 200

struct HashIn {
    uint8_t msg[MAX_MSG];
    uint8_t key[MAX_KEY];
    int msg_len;
    int key_len;
    int split; /* where the two-update variant cuts the message */
};

struct HashOut {
    uint8_t sha[32];
    uint8_t sha_split[32];
    uint8_t sha_bytewise[32];
    uint8_t hmac[32];
    uint8_t hmac_split[32];
};

__host__ __device__ static void compute(const HashIn *in, HashOut *out)
{
    sha256(in->msg, in->msg_len, out->sha);

    sha256_ctx_t c;
    sha256_init(&c);
    sha256_update(&c, in->msg, in->split);
    sha256_update(&c, in->msg + in->split, in->msg_len - in->split);
    sha256_final(&c, out->sha_split);

    sha256_init(&c);
    for (int i = 0; i < in->msg_len; i++)
        sha256_update(&c, in->msg + i, 1);
    sha256_final(&c, out->sha_bytewise);

    hmac_sha256(in->key, in->key_len, in->msg, in->msg_len, out->hmac);

    hmac_sha256_ctx_t h;
    hmac_sha256_init(&h, in->key, in->key_len);
    hmac_sha256_update(&h, in->msg, in->split);
    hmac_sha256_update(&h, in->msg + in->split, in->msg_len - in->split);
    hmac_sha256_final(&h, out->hmac_split);
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
    const int key_lens[] = { 0, 1, 20, 32, 55, 56, 63, 64, 65, 100, 128, MAX_KEY };
    const int n_key_lens = sizeof(key_lens) / sizeof(key_lens[0]);
    int n = batch_size();

    for (int i = 0; i < n; i++) {
        HashIn in;
        memset(&in, 0, sizeof(in));
        if (i <= 321) {
            /* Every length through five blocks: hits all padding branches. */
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
static int known_answers(Stats &st)
{
    uint8_t out[32];
    const uint8_t abc[] = { 'a', 'b', 'c' };

    sha256(abc, 0, out);
    st.check(hex(out, 32) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
             "sha256(\"\") = %s", hex(out, 32).c_str());
    sha256(abc, 3, out);
    st.check(hex(out, 32) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
             "sha256(\"abc\") = %s", hex(out, 32).c_str());

    /* RFC 4231 test case 2 */
    const uint8_t key[] = { 'J', 'e', 'f', 'e' };
    const char *msg = "what do ya want for nothing?";
    hmac_sha256(key, 4, (const uint8_t *)msg, strlen(msg), out);
    st.check(hex(out, 32) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
             "hmac rfc4231 #2 = %s", hex(out, 32).c_str());

    /* RFC 4231 test case 6: key longer than a block */
    uint8_t long_key[131];
    memset(long_key, 0xaa, sizeof(long_key));
    msg = "Test Using Larger Than Block-Size Key - Hash Key First";
    hmac_sha256(long_key, sizeof(long_key), (const uint8_t *)msg, strlen(msg), out);
    st.check(hex(out, 32) == "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
             "hmac rfc4231 #6 = %s", hex(out, 32).c_str());
    return 0;
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
        uint8_t ref_sha[32], ref_hmac[32];
        unsigned int hmac_len = 0;
        SHA256(x.msg, x.msg_len, ref_sha);
        HMAC(EVP_sha256(), x.key, x.key_len, x.msg, x.msg_len, ref_hmac, &hmac_len);

        HashOut host;
        compute(&x, &host);

        const char *what[] = { "sha256", "sha256 split", "sha256 bytewise", "hmac", "hmac split" };
        const uint8_t *dev[] = { out[i].sha, out[i].sha_split, out[i].sha_bytewise, out[i].hmac, out[i].hmac_split };
        const uint8_t *cpu[] = { host.sha, host.sha_split, host.sha_bytewise, host.hmac, host.hmac_split };
        for (int k = 0; k < 5; k++) {
            const uint8_t *ref = k < 3 ? ref_sha : ref_hmac;
            st.check(memcmp(dev[k], ref, 32) == 0, "[%zu] gpu %s msg_len=%d key_len=%d split=%d: %s != %s", i,
                     what[k], x.msg_len, x.key_len, x.split, hex(dev[k], 32).c_str(), hex(ref, 32).c_str());
            st.check(memcmp(cpu[k], ref, 32) == 0, "[%zu] host %s msg_len=%d key_len=%d split=%d: %s != %s", i,
                     what[k], x.msg_len, x.key_len, x.split, hex(cpu[k], 32).c_str(), hex(ref, 32).c_str());
        }
    }
    return st.finish("test_hash");
}
