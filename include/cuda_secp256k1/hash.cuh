/*
 * hash.cuh - SHA-256 and SHA-512 (FIPS 180-4) with HMAC (RFC 2104).
 *
 * Plain __host__ __device__ code with no dynamic memory, so the same
 * functions run in a GPU thread and on the CPU. The streaming interface
 * (init / update / final) lets a caller absorb a key or prefix once and
 * then copy the context for every message that shares it, which is what
 * HMAC-based key derivation (PBKDF2, BIP32) needs.
 */
#ifndef CUDA_SECP256K1_HASH_CUH
#define CUDA_SECP256K1_HASH_CUH

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef HASH_HD
#define HASH_HD static __host__ __device__ inline
#endif

#define SHA256_BLOCK_SIZE 64
#define SHA256_DIGEST_SIZE 32

typedef struct {
    uint32_t h[8];                 /* chaining value */
    uint8_t buf[SHA256_BLOCK_SIZE]; /* unprocessed tail, len % 64 bytes used */
    uint64_t len;                  /* total bytes absorbed so far */
} sha256_ctx_t;

/* ---- round constants --------------------------------------------------- */

/*
 * The same table is needed by host and device code. A __constant__ array is
 * not readable from the host and a host array is not visible to the device,
 * so the table is defined once as a macro and instantiated for each side.
 */
#define SHA256_K_VALUES                                                                 \
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, \
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, \
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, \
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, \
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, \
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, \
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, \
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, \
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, \
    0xc67178f2

static const uint32_t sha256_k_host[64] = { SHA256_K_VALUES };
static __constant__ uint32_t sha256_k_device[64] = { SHA256_K_VALUES };

#if defined(__CUDA_ARCH__)
#define SHA256_K sha256_k_device
#else
#define SHA256_K sha256_k_host
#endif

/* ---- compression function ---------------------------------------------- */

HASH_HD uint32_t sha256_rotr(uint32_t x, int n)
{
    return (x >> n) | (x << (32 - n));
}

HASH_HD uint32_t sha256_load_be32(const uint8_t *p)
{
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | (uint32_t)p[3];
}

HASH_HD void sha256_store_be32(uint8_t *p, uint32_t x)
{
    p[0] = (uint8_t)(x >> 24);
    p[1] = (uint8_t)(x >> 16);
    p[2] = (uint8_t)(x >> 8);
    p[3] = (uint8_t)x;
}

/* Absorb one 64-byte block into h. */
HASH_HD void sha256_block(uint32_t h[8], const uint8_t block[SHA256_BLOCK_SIZE])
{
    uint32_t w[16];
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

    /*
     * The message schedule only ever looks back 16 words, so a 16-word ring
     * is enough. With the loop fully unrolled every index is a constant and
     * the ring stays in registers.
     */
#pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t wi;
        if (i < 16) {
            wi = sha256_load_be32(block + 4 * i);
        } else {
            uint32_t w2 = w[(i - 2) & 15], w15 = w[(i - 15) & 15];
            uint32_t s0 = sha256_rotr(w15, 7) ^ sha256_rotr(w15, 18) ^ (w15 >> 3);
            uint32_t s1 = sha256_rotr(w2, 17) ^ sha256_rotr(w2, 19) ^ (w2 >> 10);
            wi = s1 + w[(i - 7) & 15] + s0 + w[(i - 16) & 15];
        }
        w[i & 15] = wi;

        uint32_t S1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + SHA256_K[i] + wi;
        uint32_t S0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;

        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

/* ---- streaming interface ----------------------------------------------- */

HASH_HD void sha256_init(sha256_ctx_t *c)
{
    c->h[0] = 0x6a09e667;
    c->h[1] = 0xbb67ae85;
    c->h[2] = 0x3c6ef372;
    c->h[3] = 0xa54ff53a;
    c->h[4] = 0x510e527f;
    c->h[5] = 0x9b05688c;
    c->h[6] = 0x1f83d9ab;
    c->h[7] = 0x5be0cd19;
    c->len = 0;
}

HASH_HD void sha256_update(sha256_ctx_t *c, const uint8_t *data, size_t len)
{
    size_t have = (size_t)(c->len % SHA256_BLOCK_SIZE);
    c->len += len;

    /* Top up a partially filled buffer first. */
    if (have) {
        size_t need = SHA256_BLOCK_SIZE - have;
        if (len < need) {
            memcpy(c->buf + have, data, len);
            return;
        }
        memcpy(c->buf + have, data, need);
        sha256_block(c->h, c->buf);
        data += need;
        len -= need;
    }

    /* Whole blocks straight from the input, without copying. */
    while (len >= SHA256_BLOCK_SIZE) {
        sha256_block(c->h, data);
        data += SHA256_BLOCK_SIZE;
        len -= SHA256_BLOCK_SIZE;
    }

    if (len)
        memcpy(c->buf, data, len);
}

HASH_HD void sha256_final(sha256_ctx_t *c, uint8_t out[SHA256_DIGEST_SIZE])
{
    size_t have = (size_t)(c->len % SHA256_BLOCK_SIZE);
    uint64_t bits = c->len * 8;

    /* Padding: 0x80, zeros, then the message length in bits as 8 big-endian bytes. */
    c->buf[have++] = 0x80;
    if (have > SHA256_BLOCK_SIZE - 8) {
        memset(c->buf + have, 0, SHA256_BLOCK_SIZE - have);
        sha256_block(c->h, c->buf);
        have = 0;
    }
    memset(c->buf + have, 0, SHA256_BLOCK_SIZE - 8 - have);
    sha256_store_be32(c->buf + SHA256_BLOCK_SIZE - 8, (uint32_t)(bits >> 32));
    sha256_store_be32(c->buf + SHA256_BLOCK_SIZE - 4, (uint32_t)bits);
    sha256_block(c->h, c->buf);

    for (int i = 0; i < 8; i++)
        sha256_store_be32(out + 4 * i, c->h[i]);
}

HASH_HD void sha256(const uint8_t *data, size_t len, uint8_t out[SHA256_DIGEST_SIZE])
{
    sha256_ctx_t c;
    sha256_init(&c);
    sha256_update(&c, data, len);
    sha256_final(&c, out);
}

/* ---- HMAC-SHA256 ------------------------------------------------------- */

/*
 * After hmac_sha256_init the two contexts hold the states after absorbing
 * (key ^ ipad) and (key ^ opad). A caller that reuses a key can keep this
 * struct and copy it per message instead of calling init again.
 */
typedef struct {
    sha256_ctx_t inner;
    sha256_ctx_t outer;
} hmac_sha256_ctx_t;

HASH_HD void hmac_sha256_init(hmac_sha256_ctx_t *c, const uint8_t *key, size_t key_len)
{
    uint8_t pad[SHA256_BLOCK_SIZE];

    /* Keys longer than a block are replaced by their hash; shorter ones are zero-padded. */
    memset(pad, 0, sizeof(pad));
    if (key_len > SHA256_BLOCK_SIZE)
        sha256(key, key_len, pad);
    else
        memcpy(pad, key, key_len);

    for (int i = 0; i < SHA256_BLOCK_SIZE; i++)
        pad[i] ^= 0x36;
    sha256_init(&c->inner);
    sha256_update(&c->inner, pad, sizeof(pad));

    for (int i = 0; i < SHA256_BLOCK_SIZE; i++)
        pad[i] ^= 0x36 ^ 0x5c;
    sha256_init(&c->outer);
    sha256_update(&c->outer, pad, sizeof(pad));
}

HASH_HD void hmac_sha256_update(hmac_sha256_ctx_t *c, const uint8_t *data, size_t len)
{
    sha256_update(&c->inner, data, len);
}

HASH_HD void hmac_sha256_final(hmac_sha256_ctx_t *c, uint8_t out[SHA256_DIGEST_SIZE])
{
    uint8_t inner_hash[SHA256_DIGEST_SIZE];
    sha256_final(&c->inner, inner_hash);
    sha256_update(&c->outer, inner_hash, sizeof(inner_hash));
    sha256_final(&c->outer, out);
}

HASH_HD void hmac_sha256(const uint8_t *key, size_t key_len, const uint8_t *data, size_t len,
                         uint8_t out[SHA256_DIGEST_SIZE])
{
    hmac_sha256_ctx_t c;
    hmac_sha256_init(&c, key, key_len);
    hmac_sha256_update(&c, data, len);
    hmac_sha256_final(&c, out);
}

/* ====================================================================== */
/* SHA-512 and HMAC-SHA512: the same construction with 64-bit words,      */
/* 128-byte blocks, 80 rounds and a 128-bit length field.                 */
/* ====================================================================== */

#define SHA512_BLOCK_SIZE 128
#define SHA512_DIGEST_SIZE 64

typedef struct {
    uint64_t h[8];
    uint8_t buf[SHA512_BLOCK_SIZE];
    uint64_t len;
} sha512_ctx_t;

#define SHA512_K_VALUES                                                                   \
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,       \
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,       \
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,       \
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,       \
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,       \
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,       \
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,       \
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,       \
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,       \
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,       \
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,       \
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,       \
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,       \
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,       \
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,       \
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,       \
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,       \
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,       \
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,       \
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817

static const uint64_t sha512_k_host[80] = { SHA512_K_VALUES };
static __constant__ uint64_t sha512_k_device[80] = { SHA512_K_VALUES };

#if defined(__CUDA_ARCH__)
#define SHA512_K sha512_k_device
#else
#define SHA512_K sha512_k_host
#endif

HASH_HD uint64_t sha512_rotr(uint64_t x, int n)
{
    return (x >> n) | (x << (64 - n));
}

HASH_HD uint64_t sha512_load_be64(const uint8_t *p)
{
    return (uint64_t)sha256_load_be32(p) << 32 | sha256_load_be32(p + 4);
}

HASH_HD void sha512_store_be64(uint8_t *p, uint64_t x)
{
    sha256_store_be32(p, (uint32_t)(x >> 32));
    sha256_store_be32(p + 4, (uint32_t)x);
}

/* Absorb one 128-byte block into h. */
HASH_HD void sha512_block(uint64_t h[8], const uint8_t block[SHA512_BLOCK_SIZE])
{
    uint64_t w[16];
    uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];

#pragma unroll
    for (int i = 0; i < 80; i++) {
        uint64_t wi;
        if (i < 16) {
            wi = sha512_load_be64(block + 8 * i);
        } else {
            uint64_t w2 = w[(i - 2) & 15], w15 = w[(i - 15) & 15];
            uint64_t s0 = sha512_rotr(w15, 1) ^ sha512_rotr(w15, 8) ^ (w15 >> 7);
            uint64_t s1 = sha512_rotr(w2, 19) ^ sha512_rotr(w2, 61) ^ (w2 >> 6);
            wi = s1 + w[(i - 7) & 15] + s0 + w[(i - 16) & 15];
        }
        w[i & 15] = wi;

        uint64_t S1 = sha512_rotr(e, 14) ^ sha512_rotr(e, 18) ^ sha512_rotr(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        uint64_t t1 = hh + S1 + ch + SHA512_K[i] + wi;
        uint64_t S0 = sha512_rotr(a, 28) ^ sha512_rotr(a, 34) ^ sha512_rotr(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + maj;

        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

HASH_HD void sha512_init(sha512_ctx_t *c)
{
    c->h[0] = 0x6a09e667f3bcc908;
    c->h[1] = 0xbb67ae8584caa73b;
    c->h[2] = 0x3c6ef372fe94f82b;
    c->h[3] = 0xa54ff53a5f1d36f1;
    c->h[4] = 0x510e527fade682d1;
    c->h[5] = 0x9b05688c2b3e6c1f;
    c->h[6] = 0x1f83d9abfb41bd6b;
    c->h[7] = 0x5be0cd19137e2179;
    c->len = 0;
}

HASH_HD void sha512_update(sha512_ctx_t *c, const uint8_t *data, size_t len)
{
    size_t have = (size_t)(c->len % SHA512_BLOCK_SIZE);
    c->len += len;

    if (have) {
        size_t need = SHA512_BLOCK_SIZE - have;
        if (len < need) {
            memcpy(c->buf + have, data, len);
            return;
        }
        memcpy(c->buf + have, data, need);
        sha512_block(c->h, c->buf);
        data += need;
        len -= need;
    }

    while (len >= SHA512_BLOCK_SIZE) {
        sha512_block(c->h, data);
        data += SHA512_BLOCK_SIZE;
        len -= SHA512_BLOCK_SIZE;
    }

    if (len)
        memcpy(c->buf, data, len);
}

HASH_HD void sha512_final(sha512_ctx_t *c, uint8_t out[SHA512_DIGEST_SIZE])
{
    size_t have = (size_t)(c->len % SHA512_BLOCK_SIZE);

    /* Padding: 0x80, zeros, then the length in bits as 16 big-endian bytes. */
    c->buf[have++] = 0x80;
    if (have > SHA512_BLOCK_SIZE - 16) {
        memset(c->buf + have, 0, SHA512_BLOCK_SIZE - have);
        sha512_block(c->h, c->buf);
        have = 0;
    }
    memset(c->buf + have, 0, SHA512_BLOCK_SIZE - 16 - have);
    sha512_store_be64(c->buf + SHA512_BLOCK_SIZE - 16, c->len >> 61);
    sha512_store_be64(c->buf + SHA512_BLOCK_SIZE - 8, c->len << 3);
    sha512_block(c->h, c->buf);

    for (int i = 0; i < 8; i++)
        sha512_store_be64(out + 8 * i, c->h[i]);
}

HASH_HD void sha512(const uint8_t *data, size_t len, uint8_t out[SHA512_DIGEST_SIZE])
{
    sha512_ctx_t c;
    sha512_init(&c);
    sha512_update(&c, data, len);
    sha512_final(&c, out);
}

typedef struct {
    sha512_ctx_t inner;
    sha512_ctx_t outer;
} hmac_sha512_ctx_t;

HASH_HD void hmac_sha512_init(hmac_sha512_ctx_t *c, const uint8_t *key, size_t key_len)
{
    uint8_t pad[SHA512_BLOCK_SIZE];

    memset(pad, 0, sizeof(pad));
    if (key_len > SHA512_BLOCK_SIZE)
        sha512(key, key_len, pad);
    else
        memcpy(pad, key, key_len);

    for (int i = 0; i < SHA512_BLOCK_SIZE; i++)
        pad[i] ^= 0x36;
    sha512_init(&c->inner);
    sha512_update(&c->inner, pad, sizeof(pad));

    for (int i = 0; i < SHA512_BLOCK_SIZE; i++)
        pad[i] ^= 0x36 ^ 0x5c;
    sha512_init(&c->outer);
    sha512_update(&c->outer, pad, sizeof(pad));
}

HASH_HD void hmac_sha512_update(hmac_sha512_ctx_t *c, const uint8_t *data, size_t len)
{
    sha512_update(&c->inner, data, len);
}

HASH_HD void hmac_sha512_final(hmac_sha512_ctx_t *c, uint8_t out[SHA512_DIGEST_SIZE])
{
    uint8_t inner_hash[SHA512_DIGEST_SIZE];
    sha512_final(&c->inner, inner_hash);
    sha512_update(&c->outer, inner_hash, sizeof(inner_hash));
    sha512_final(&c->outer, out);
}

HASH_HD void hmac_sha512(const uint8_t *key, size_t key_len, const uint8_t *data, size_t len,
                         uint8_t out[SHA512_DIGEST_SIZE])
{
    hmac_sha512_ctx_t c;
    hmac_sha512_init(&c, key, key_len);
    hmac_sha512_update(&c, data, len);
    hmac_sha512_final(&c, out);
}

#endif
