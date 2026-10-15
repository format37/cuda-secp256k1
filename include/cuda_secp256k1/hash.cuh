/*
 * hash.cuh - SHA-256 (FIPS 180-4) and HMAC-SHA256 (RFC 2104).
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
    for (int i = 0; i < 8; i++)
        c->buf[SHA256_BLOCK_SIZE - 1 - i] = (uint8_t)(bits >> (8 * i));
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

#endif
