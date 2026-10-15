/*
 * hash.cuh - SHA-256 and SHA-512 (FIPS 180-4) with HMAC (RFC 2104), RIPEMD-160,
 * and the hash160 / sha256d compositions used by address formats.
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

/* ====================================================================== */
/* RIPEMD-160 (ISO/IEC 10118-3): little-endian, two parallel lines of 80  */
/* steps, 160-bit output. Used for hash160 = RIPEMD-160(SHA-256(x)).       */
/* ====================================================================== */

#define RIPEMD160_BLOCK_SIZE 64
#define RIPEMD160_DIGEST_SIZE 20

typedef struct {
    uint32_t h[5];
    uint8_t buf[RIPEMD160_BLOCK_SIZE];
    uint64_t len;
} ripemd160_ctx_t;

HASH_HD uint32_t ripemd160_rotl(uint32_t x, int n)
{
    return (x << n) | (x >> (32 - n));
}

HASH_HD uint32_t ripemd160_load_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

HASH_HD void ripemd160_store_le32(uint8_t *p, uint32_t x)
{
    p[0] = (uint8_t)x;
    p[1] = (uint8_t)(x >> 8);
    p[2] = (uint8_t)(x >> 16);
    p[3] = (uint8_t)(x >> 24);
}

/* Boolean function of round 0..4; the right line uses them in reverse order. */
HASH_HD uint32_t ripemd160_f(int round, uint32_t x, uint32_t y, uint32_t z)
{
    switch (round) {
    case 0:
        return x ^ y ^ z;
    case 1:
        return (x & y) | (~x & z);
    case 2:
        return (x | ~y) ^ z;
    case 3:
        return (x & z) | (y & ~z);
    default:
        return x ^ (y | ~z);
    }
}

HASH_HD uint32_t ripemd160_k_left(int round)
{
    switch (round) {
    case 0:
        return 0x00000000;
    case 1:
        return 0x5a827999;
    case 2:
        return 0x6ed9eba1;
    case 3:
        return 0x8f1bbcdc;
    default:
        return 0xa953fd4e;
    }
}

HASH_HD uint32_t ripemd160_k_right(int round)
{
    switch (round) {
    case 0:
        return 0x50a28be6;
    case 1:
        return 0x5c4dd124;
    case 2:
        return 0x6d703ef3;
    case 3:
        return 0x7a6d76e9;
    default:
        return 0x00000000;
    }
}

/* Absorb one 64-byte block into h. */
HASH_HD void ripemd160_block(uint32_t h[5], const uint8_t block[RIPEMD160_BLOCK_SIZE])
{
    /*
     * Message word index (r, r') and rotation amount (s, s') of each step,
     * for the left and the right line. The step loop is fully unrolled, so
     * these lookups and the x[] indexes become constants.
     */
    const uint8_t r[80] = { 0,  1, 2,  3,  4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 7,  4,  13, 1,
                            10, 6, 15, 3,  12, 0, 9,  5,  2,  14, 11, 8,  3,  10, 14, 4,  9,  15, 8,  1,
                            2,  7, 0,  6,  13, 11, 5, 12, 1,  9,  11, 10, 0,  8,  12, 4,  13, 3,  7,  15,
                            14, 5, 6,  2,  4,  0, 5,  9,  7,  12, 2,  10, 14, 1,  3,  8,  11, 6,  15, 13 };
    const uint8_t rr[80] = { 5,  14, 7, 0,  9,  2,  11, 4,  13, 6,  15, 8,  1,  10, 3,  12, 6,  11, 3,  7,
                             0,  13, 5, 10, 14, 15, 8,  12, 4,  9,  1,  2,  15, 5,  1,  3,  7,  14, 6,  9,
                             11, 8,  12, 2, 10, 0,  4,  13, 8,  6,  4,  1,  3,  11, 15, 0,  5,  12, 2,  13,
                             9,  7,  10, 14, 12, 15, 10, 4,  1,  5,  8,  7,  6,  2,  13, 14, 0,  3,  9,  11 };
    const uint8_t s[80] = { 11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,  7,  9,  8,  7,  6,  8,  13,
                            11, 9,  7,  15, 7,  12, 15, 9,  11, 7,  13, 12, 11, 13, 6,  7,  14, 9,  13, 15,
                            14, 8,  13, 6,  5,  12, 7,  5,  11, 12, 14, 15, 14, 15, 9,  8,  9,  14, 5,  6,
                            8,  6,  5,  12, 9,  15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,  5,  6 };
    const uint8_t ss[80] = { 8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14, 14, 12, 6,  9,  13, 15, 7,
                             12, 8,  9,  11, 7,  7,  12, 7,  6,  15, 13, 11, 9,  7,  15, 11, 8,  6,  6,  14,
                             12, 13, 5,  14, 13, 13, 7,  5,  15, 5,  8,  11, 14, 14, 6,  14, 6,  9,  12, 9,
                             12, 5,  15, 8,  8,  5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13, 11, 11 };
    uint32_t x[16];
    uint32_t al = h[0], bl = h[1], cl = h[2], dl = h[3], el = h[4];
    uint32_t ar = al, br = bl, cr = cl, dr = dl, er = el;

#pragma unroll
    for (int i = 0; i < 16; i++)
        x[i] = ripemd160_load_le32(block + 4 * i);

#pragma unroll
    for (int j = 0; j < 80; j++) {
        int round = j / 16;
        uint32_t t;

        t = ripemd160_rotl(al + ripemd160_f(round, bl, cl, dl) + x[r[j]] + ripemd160_k_left(round), s[j]) + el;
        al = el;
        el = dl;
        dl = ripemd160_rotl(cl, 10);
        cl = bl;
        bl = t;

        t = ripemd160_rotl(ar + ripemd160_f(4 - round, br, cr, dr) + x[rr[j]] + ripemd160_k_right(round), ss[j]) + er;
        ar = er;
        er = dr;
        dr = ripemd160_rotl(cr, 10);
        cr = br;
        br = t;
    }

    uint32_t t = h[1] + cl + dr;
    h[1] = h[2] + dl + er;
    h[2] = h[3] + el + ar;
    h[3] = h[4] + al + br;
    h[4] = h[0] + bl + cr;
    h[0] = t;
}

HASH_HD void ripemd160_init(ripemd160_ctx_t *c)
{
    c->h[0] = 0x67452301;
    c->h[1] = 0xefcdab89;
    c->h[2] = 0x98badcfe;
    c->h[3] = 0x10325476;
    c->h[4] = 0xc3d2e1f0;
    c->len = 0;
}

HASH_HD void ripemd160_update(ripemd160_ctx_t *c, const uint8_t *data, size_t len)
{
    size_t have = (size_t)(c->len % RIPEMD160_BLOCK_SIZE);
    c->len += len;

    if (have) {
        size_t need = RIPEMD160_BLOCK_SIZE - have;
        if (len < need) {
            memcpy(c->buf + have, data, len);
            return;
        }
        memcpy(c->buf + have, data, need);
        ripemd160_block(c->h, c->buf);
        data += need;
        len -= need;
    }

    while (len >= RIPEMD160_BLOCK_SIZE) {
        ripemd160_block(c->h, data);
        data += RIPEMD160_BLOCK_SIZE;
        len -= RIPEMD160_BLOCK_SIZE;
    }

    if (len)
        memcpy(c->buf, data, len);
}

HASH_HD void ripemd160_final(ripemd160_ctx_t *c, uint8_t out[RIPEMD160_DIGEST_SIZE])
{
    size_t have = (size_t)(c->len % RIPEMD160_BLOCK_SIZE);
    uint64_t bits = c->len * 8;

    /* Padding: 0x80, zeros, then the length in bits as 8 little-endian bytes. */
    c->buf[have++] = 0x80;
    if (have > RIPEMD160_BLOCK_SIZE - 8) {
        memset(c->buf + have, 0, RIPEMD160_BLOCK_SIZE - have);
        ripemd160_block(c->h, c->buf);
        have = 0;
    }
    memset(c->buf + have, 0, RIPEMD160_BLOCK_SIZE - 8 - have);
    ripemd160_store_le32(c->buf + RIPEMD160_BLOCK_SIZE - 8, (uint32_t)bits);
    ripemd160_store_le32(c->buf + RIPEMD160_BLOCK_SIZE - 4, (uint32_t)(bits >> 32));
    ripemd160_block(c->h, c->buf);

    for (int i = 0; i < 5; i++)
        ripemd160_store_le32(out + 4 * i, c->h[i]);
}

HASH_HD void ripemd160(const uint8_t *data, size_t len, uint8_t out[RIPEMD160_DIGEST_SIZE])
{
    ripemd160_ctx_t c;
    ripemd160_init(&c);
    ripemd160_update(&c, data, len);
    ripemd160_final(&c, out);
}

/* hash160(x) = RIPEMD-160(SHA-256(x)), the hash behind Bitcoin and Avalanche addresses. */
HASH_HD void hash160(const uint8_t *data, size_t len, uint8_t out[RIPEMD160_DIGEST_SIZE])
{
    uint8_t h[SHA256_DIGEST_SIZE];
    sha256(data, len, h);
    ripemd160(h, sizeof(h), out);
}

/* sha256d(x) = SHA-256(SHA-256(x)), used for Base58Check checksums. */
HASH_HD void sha256d(const uint8_t *data, size_t len, uint8_t out[SHA256_DIGEST_SIZE])
{
    sha256(data, len, out);
    sha256(out, SHA256_DIGEST_SIZE, out);
}

#endif
