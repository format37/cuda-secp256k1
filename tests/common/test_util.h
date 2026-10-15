/*
 * test_util.h - host-side helpers shared by the tests.
 *
 * Tests run the library on the GPU over a batch of inputs, then check every
 * result against OpenSSL on the host. The input batch mixes hand-picked edge
 * cases with values from a seeded generator, so a failure is reproducible:
 * set CUDA_SECP256K1_SEED to change the seed and CUDA_SECP256K1_N to change
 * the batch size. Without a GPU (or with CUDA_SECP256K1_HOST_ONLY=1) the
 * kernel's work is done on the host by the same __host__ __device__ code,
 * so the suite also runs on a machine without CUDA hardware, such as CI.
 */
#ifndef CUDA_SECP256K1_TEST_UTIL_H
#define CUDA_SECP256K1_TEST_UTIL_H

#include <cuda_runtime.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <string>
#include <vector>

#include <cuda_secp256k1/secp256k1.cuh>

#define CUDA_CHECK(call)                                                                \
    do {                                                                                \
        cudaError_t err__ = (call);                                                     \
        if (err__ != cudaSuccess) {                                                     \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                         \
            exit(2);                                                                    \
        }                                                                               \
    } while (0)

static long env_long(const char *name, long def)
{
    const char *v = getenv(name);
    return v ? atol(v) : def;
}

static std::mt19937_64 &rng()
{
    static std::mt19937_64 r((uint64_t)env_long("CUDA_SECP256K1_SEED", 1));
    return r;
}

static int batch_size()
{
    return (int)env_long("CUDA_SECP256K1_N", 4096);
}

/* ---- pass/fail accounting ---------------------------------------------- */

struct Stats {
    int passed = 0;
    int failed = 0;
    int shown = 0;

    void check(bool ok, const char *fmt, ...)
    {
        if (ok) {
            passed++;
            return;
        }
        failed++;
        if (shown < 10) {
            shown++;
            va_list ap;
            va_start(ap, fmt);
            vfprintf(stderr, fmt, ap);
            va_end(ap);
            fputc('\n', stderr);
        }
    }

    int finish(const char *name)
    {
        printf("%s: %d passed, %d failed\n", name, passed, failed);
        return failed ? 1 : 0;
    }
};

/* ---- conversions between bn_t and OpenSSL BIGNUM ------------------------ */

static void to_ossl(BIGNUM *out, const bn_t *a)
{
    unsigned char buf[BN_WORDS * 8];
    for (int i = 0; i < BN_WORDS; i++)
        for (int j = 0; j < 8; j++)
            buf[i * 8 + j] = (unsigned char)(a->d[i] >> (8 * j));
    BN_lebin2bn(buf, sizeof(buf), out);
    BN_set_negative(out, a->neg);
}

static void from_ossl(bn_t *out, const BIGNUM *a)
{
    unsigned char buf[BN_WORDS * 8];
    if (BN_bn2lebinpad(a, buf, sizeof(buf)) < 0) {
        fprintf(stderr, "from_ossl: value does not fit in %d bits\n", BN_MAX_BITS);
        exit(2);
    }
    bn_zero(out);
    for (int i = 0; i < BN_WORDS; i++)
        for (int j = 0; j < 8; j++)
            out->d[i] |= (bn_word)buf[i * 8 + j] << (8 * j);
    out->top = BN_WORDS;
    bn_normalize(out);
    out->neg = BN_is_negative(a) && out->top;
}

static bool equals(const bn_t *a, const BIGNUM *b)
{
    BIGNUM *t = BN_new();
    to_ossl(t, a);
    bool eq = BN_cmp(t, b) == 0;
    BN_free(t);
    return eq;
}

static std::string hex(const bn_t *a)
{
    char buf[BN_WORDS * 16 + 2];
    bn_to_hex(a, buf);
    return buf;
}

static std::string hex(const BIGNUM *a)
{
    char *s = BN_bn2hex(a);
    std::string r(s);
    OPENSSL_free(s);
    for (auto &c : r)
        c = (char)tolower(c);
    return r;
}

/* ---- input generation -------------------------------------------------- */

/* Uniform random value with exactly `bits` bits (top bit set), or zero. */
static bn_t rand_bits(int bits)
{
    bn_t r;
    bn_zero(&r);
    if (bits <= 0)
        return r;
    for (int i = 0; i < (bits + 63) / 64; i++)
        r.d[i] = rng()();
    int extra = (bits + 63) / 64 * 64 - bits;
    if (extra)
        r.d[(bits - 1) / 64] &= ~(bn_word)0 >> extra;
    r.d[(bits - 1) / 64] |= (bn_word)1 << ((bits - 1) % 64);
    r.top = BN_WORDS;
    bn_normalize(&r);
    return r;
}

/* Random value with a random bit length in [0, max_bits]. */
static bn_t rand_upto(int max_bits)
{
    return rand_bits((int)(rng()() % (uint64_t)(max_bits + 1)));
}

/* 2^k, for 0 <= k < BN_MAX_BITS. */
static bn_t power_of_two(int k)
{
    bn_t r;
    bn_zero(&r);
    r.d[k / 64] = (bn_word)1 << (k % 64);
    r.top = k / 64 + 1;
    return r;
}

/* 2^k - 1, all of the low k bits set, for 0 <= k <= BN_MAX_BITS. */
static bn_t all_ones(int k)
{
    bn_t r;
    bn_zero(&r);
    for (int i = 0; i < k / 64; i++)
        r.d[i] = ~(bn_word)0;
    if (k % 64)
        r.d[k / 64] = ~(bn_word)0 >> (64 - k % 64);
    r.top = BN_WORDS;
    bn_normalize(&r);
    return r;
}

/* Uniform random value in [0, n). */
static bn_t rand_below(const bn_t *n)
{
    int bits = bn_num_bits(n);
    for (;;) {
        bn_t r = rand_bits(bits);
        for (int i = 0; i < (bits + 63) / 64; i++)
            r.d[i] = rng()();
        int extra = (bits + 63) / 64 * 64 - bits;
        if (extra)
            r.d[(bits - 1) / 64] &= ~(bn_word)0 >> extra;
        r.top = BN_WORDS;
        bn_normalize(&r);
        if (bn_ucmp(&r, n) < 0)
            return r;
    }
}

static bn_t from_hex(const char *h)
{
    bn_t r;
    if (!bn_from_hex(&r, h)) {
        fprintf(stderr, "bad hex constant: %s\n", h);
        exit(2);
    }
    return r;
}

/* Values that tend to expose carry, borrow and normalization mistakes. */
static std::vector<bn_t> edge_values()
{
    std::vector<bn_t> v;
    const char *hexes[] = {
        "0", "1", "2", "3", "7", "ff", "100",
        "ffffffffffffffff", "10000000000000000", "ffffffffffffffffffffffffffffffff",
        "8000000000000000", "80000000000000000000000000000000",
        "8000000000000000000000000000000000000000000000000000000000000000",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f", /* p */
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2e", /* p - 1 */
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc30", /* p + 1 */
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141", /* n */
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140", /* n - 1 */
        "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0", /* (n-1)/2 */
        "1000003d1",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", /* Gx */
    };
    for (const char *h : hexes)
        v.push_back(from_hex(h));
    /* Word boundaries above 256 bits, up to the full width of a bn_t. */
    for (int k : {256, 320, 448, BN_MAX_BITS - 1})
        v.push_back(power_of_two(k));
    for (int k : {320, 384, 448, BN_MAX_BITS})
        v.push_back(all_ones(k));
    return v;
}

/* ---- OpenSSL objects --------------------------------------------------- */

struct Ossl {
    BN_CTX *ctx;
    EC_GROUP *group;
    BIGNUM *p, *order;

    Ossl()
    {
        ctx = BN_CTX_new();
        group = EC_GROUP_new_by_curve_name(NID_secp256k1);
        p = BN_new();
        order = BN_new();
        if (!ctx || !group || !EC_GROUP_get_curve(group, p, NULL, NULL, ctx)
            || !EC_GROUP_get_order(group, order, ctx)) {
            fprintf(stderr, "OpenSSL setup failed\n");
            exit(2);
        }
    }
    ~Ossl()
    {
        BN_free(order);
        BN_free(p);
        EC_GROUP_free(group);
        BN_CTX_free(ctx);
    }
};

/* True when a CUDA device is usable; false on a machine without one or with CUDA_SECP256K1_HOST_ONLY=1. */
static bool have_cuda_device()
{
    static int cached = -1;
    if (cached < 0) {
        int count = 0;
        cached = !env_long("CUDA_SECP256K1_HOST_ONLY", 0) && cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
        if (!cached)
            printf("no CUDA device in use: running the kernel's work on the host\n");
    }
    return cached != 0;
}

/*
 * Upload a host vector, run a kernel through `launch`, download the result.
 * Without a device, `host_fn(&in[i], &out[i])` does the same work on the host.
 */
template <typename In, typename Out, typename Launch, typename HostFn>
static std::vector<Out> run_on_device(const std::vector<In> &in, Launch launch, HostFn host_fn)
{
    size_t n = in.size();
    if (!have_cuda_device()) {
        std::vector<Out> out(n);
        for (size_t i = 0; i < n; i++)
            host_fn(&in[i], &out[i]);
        return out;
    }
    In *d_in = NULL;
    Out *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(In)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Out)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(In), cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    launch(d_in, d_out, (int)n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("kernel: %zu inputs in %.1f ms\n", n, ms);
    std::vector<Out> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(Out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return out;
}

#endif
