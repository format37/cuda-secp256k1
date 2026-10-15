/*
 * test_pbkdf2.cu - PBKDF2-HMAC-SHA256 and PBKDF2-HMAC-SHA512 against OpenSSL.
 *
 * Password and salt lengths cover empty, block-sized and longer-than-block
 * values; output lengths cover partial and whole digest blocks; iteration
 * counts run from 1 to 4096. Every derived key is computed on the GPU and
 * on the host and compared with PKCS5_PBKDF2_HMAC().
 */
#include <openssl/evp.h>

#include "test_util.h"

#define MAX_PW 128
#define MAX_SALT 128
#define MAX_OUT 160

struct KdfIn {
    uint8_t pw[MAX_PW];
    uint8_t salt[MAX_SALT];
    int pw_len;
    int salt_len;
    uint32_t iterations;
    int out_len;
};

struct KdfOut {
    uint8_t dk256[MAX_OUT];
    uint8_t dk512[MAX_OUT];
    int ok256;
    int ok512;
};

__host__ __device__ static void compute(const KdfIn *in, KdfOut *out)
{
    out->ok256 = pbkdf2_hmac_sha256(in->pw, in->pw_len, in->salt, in->salt_len, in->iterations, out->dk256, in->out_len);
    out->ok512 = pbkdf2_hmac_sha512(in->pw, in->pw_len, in->salt, in->salt_len, in->iterations, out->dk512, in->out_len);
}

__global__ void kdf_kernel(const KdfIn *in, KdfOut *out, int n)
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

static std::vector<KdfIn> make_inputs()
{
    std::vector<KdfIn> v;
    const int pw_lens[] = { 0, 1, 8, 20, 63, 64, 65, 127, 128 };
    const int salt_lens[] = { 0, 1, 4, 8, 16, 32, 64, 100, 128 };
    const uint32_t iters[] = { 1, 2, 3, 4, 7, 16, 64, 100 };
    const int out_lens[] = { 1, 16, 31, 32, 33, 63, 64, 65, 96, 100, 127, 128, 129, 160 };
    int n = batch_size();

    for (int i = 0; i < n; i++) {
        KdfIn in;
        memset(&in, 0, sizeof(in));
        in.pw_len = pw_lens[rng()() % 9];
        in.salt_len = salt_lens[rng()() % 9];
        in.iterations = iters[rng()() % 8];
        in.out_len = out_lens[rng()() % 14];
        if (i == 0) { /* everything at its maximum */
            in.pw_len = MAX_PW; in.salt_len = MAX_SALT; in.out_len = MAX_OUT;
        } else if (i == 1) { /* rejected: zero iterations */
            in.iterations = 0;
        } else if (i == 2) { /* everything empty */
            in.pw_len = 0; in.salt_len = 0; in.out_len = 1;
        } else if (i % 128 == 64) {
            in.iterations = 4096;
        } else if (i % 32 == 16) {
            in.iterations = 2048;
        }
        fill_random(in.pw, in.pw_len);
        fill_random(in.salt, in.salt_len);
        v.push_back(in);
    }
    return v;
}

/* Published vectors, so a failure here is obvious without OpenSSL. */
static void known_answers(Stats &st)
{
    uint8_t out[64];
    const uint8_t *pw = (const uint8_t *)"password";
    const uint8_t *salt = (const uint8_t *)"salt";

    pbkdf2_hmac_sha256(pw, 8, salt, 4, 1, out, 32);
    st.check(hex(out, 32) == "120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b",
             "pbkdf2-sha256 c=1: %s", hex(out, 32).c_str());
    pbkdf2_hmac_sha256(pw, 8, salt, 4, 4096, out, 32);
    st.check(hex(out, 32) == "c5e478d59288c841aa530db6845c4c8d962893a001ce4e11a4963873aa98134a",
             "pbkdf2-sha256 c=4096: %s", hex(out, 32).c_str());

    pbkdf2_hmac_sha512(pw, 8, salt, 4, 1, out, 64);
    st.check(hex(out, 64) == "867f70cf1ade02cff3752599a3a53dc4af34c7a669815ae5d513554e1c8cf252"
                             "c02d470a285a0501bad999bfe943c08f050235d7d68b1da55e63f73b60a57fce",
             "pbkdf2-sha512 c=1: %s", hex(out, 64).c_str());
    pbkdf2_hmac_sha512(pw, 8, salt, 4, 4096, out, 64);
    st.check(hex(out, 64) == "d197b1b33db0143e018b12f3d1d1479e6cdebdcc97c5c0f87f6902e072f457b5"
                             "143f30602641b3d55cd335988cb36b84376060ecd532e039b742a239434af2d5",
             "pbkdf2-sha512 c=4096: %s", hex(out, 64).c_str());

    /* BIP39 seed: first vector of the reference test set, passphrase "TREZOR". */
    const char *mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    const char *bip39_salt = "mnemonicTREZOR";
    pbkdf2_hmac_sha512((const uint8_t *)mnemonic, strlen(mnemonic), (const uint8_t *)bip39_salt, strlen(bip39_salt),
                       2048, out, 64);
    st.check(hex(out, 64) == "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e5349553"
                             "1f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04",
             "bip39 seed: %s", hex(out, 64).c_str());
}

int main()
{
    Stats st;
    known_answers(st);

    std::vector<KdfIn> in = make_inputs();
    std::vector<KdfOut> out = run_on_device<KdfIn, KdfOut>(in, [](const KdfIn *d_in, KdfOut *d_out, int n) {
        kdf_kernel<<<(n + 127) / 128, 128>>>(d_in, d_out, n);
    });

    for (size_t i = 0; i < in.size(); i++) {
        const KdfIn &x = in[i];
        KdfOut host;
        compute(&x, &host);

        if (x.iterations == 0) {
            st.check(!out[i].ok256 && !out[i].ok512 && !host.ok256 && !host.ok512,
                     "[%zu] zero iterations must be rejected", i);
            continue;
        }

        uint8_t ref256[MAX_OUT], ref512[MAX_OUT];
        if (!PKCS5_PBKDF2_HMAC((const char *)x.pw, x.pw_len, x.salt, x.salt_len, (int)x.iterations, EVP_sha256(),
                               x.out_len, ref256)
            || !PKCS5_PBKDF2_HMAC((const char *)x.pw, x.pw_len, x.salt, x.salt_len, (int)x.iterations, EVP_sha512(),
                                  x.out_len, ref512)) {
            fprintf(stderr, "OpenSSL PBKDF2 failed\n");
            return 2;
        }

        struct Check {
            const char *what;
            int ok_dev, ok_cpu;
            const uint8_t *dev, *cpu, *ref;
        } checks[] = {
            { "sha256", out[i].ok256, host.ok256, out[i].dk256, host.dk256, ref256 },
            { "sha512", out[i].ok512, host.ok512, out[i].dk512, host.dk512, ref512 },
        };
        for (const Check &c : checks) {
            st.check(c.ok_dev && memcmp(c.dev, c.ref, x.out_len) == 0,
                     "[%zu] gpu pbkdf2-%s pw=%d salt=%d c=%u dklen=%d: %s != %s", i, c.what, x.pw_len, x.salt_len,
                     x.iterations, x.out_len, hex(c.dev, x.out_len).c_str(), hex(c.ref, x.out_len).c_str());
            st.check(c.ok_cpu && memcmp(c.cpu, c.ref, x.out_len) == 0,
                     "[%zu] host pbkdf2-%s pw=%d salt=%d c=%u dklen=%d: %s != %s", i, c.what, x.pw_len, x.salt_len,
                     x.iterations, x.out_len, hex(c.cpu, x.out_len).c_str(), hex(c.ref, x.out_len).c_str());
        }
    }
    return st.finish("test_pbkdf2");
}
