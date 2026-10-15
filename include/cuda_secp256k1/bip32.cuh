/*
 * bip32.cuh - BIP32 hierarchical deterministic derivation of private keys.
 *
 * Covers what a derivation pipeline needs: master key from a seed, private
 * child derivation (hardened and normal), walking a path, and the compressed
 * public key. Public-only derivation (CKDpub) and the xprv/xpub string
 * format are not included.
 */
#ifndef CUDA_SECP256K1_BIP32_CUH
#define CUDA_SECP256K1_BIP32_CUH

#include "ec.cuh"
#include "hash.cuh"

#define BIP32_HARDENED 0x80000000u

typedef struct {
    bn_t key; /* private key, in [1, n-1] */
    uint8_t chain_code[32];
} bip32_key_t;

/*
 * Master key: I = HMAC-SHA512("Bitcoin seed", seed); key = I_L, chain code = I_R.
 * Returns 0 for the (practically impossible) seeds where I_L is 0 or >= n.
 */
HASH_HD int bip32_master(const ec_group_t *g, bip32_key_t *k, const uint8_t *seed, size_t seed_len)
{
    uint8_t I[64];
    hmac_sha512((const uint8_t *)"Bitcoin seed", 12, seed, seed_len, I);
    bn_from_bytes_be(&k->key, I, 32);
    memcpy(k->chain_code, I + 32, 32);
    return !bn_is_zero(&k->key) && bn_ucmp(&k->key, &g->order) < 0;
}

/*
 * Private child derivation, CKDpriv((k_par, c_par), i):
 *   hardened: I = HMAC-SHA512(c_par, 0x00 || ser256(k_par) || ser32(i))
 *   normal:   I = HMAC-SHA512(c_par, serP(k_par * G) || ser32(i))
 *   k_i = (I_L + k_par) mod n,  c_i = I_R
 * child may be the same object as parent. Returns 0 when I_L >= n or
 * k_i == 0 (probability about 2^-127); BIP32 says to skip such an index.
 */
HASH_HD int bip32_ckd_priv(const ec_group_t *g, bip32_key_t *child, const bip32_key_t *parent, uint32_t index)
{
    uint8_t data[37];
    uint8_t I[64];
    bn_t il;

    if (index & BIP32_HARDENED) {
        data[0] = 0x00;
        bn_to_bytes_be(&parent->key, data + 1, 32);
    } else {
        ec_affine_t pub;
        if (!ec_pubkey_create(g, &pub, &parent->key))
            return 0;
        ec_affine_to_bytes(&pub, data, 1);
    }
    data[33] = (uint8_t)(index >> 24);
    data[34] = (uint8_t)(index >> 16);
    data[35] = (uint8_t)(index >> 8);
    data[36] = (uint8_t)index;

    hmac_sha512(parent->chain_code, 32, data, sizeof(data), I);

    bn_from_bytes_be(&il, I, 32);
    if (bn_ucmp(&il, &g->order) >= 0)
        return 0;
    bn_mod_add(&child->key, &il, &parent->key, &g->order);
    memcpy(child->chain_code, I + 32, 32);
    return !bn_is_zero(&child->key);
}

/* Walk a list of child indexes starting from k (normally the master key); k is updated in place. */
HASH_HD int bip32_derive_path(const ec_group_t *g, bip32_key_t *k, const uint32_t *path, int depth)
{
    for (int i = 0; i < depth; i++)
        if (!bip32_ckd_priv(g, k, k, path[i]))
            return 0;
    return 1;
}

/* Compressed public key (33 bytes) of an extended private key. */
HASH_HD int bip32_pubkey(const ec_group_t *g, const bip32_key_t *k, uint8_t out[33])
{
    ec_affine_t pub;
    if (!ec_pubkey_create(g, &pub, &k->key))
        return 0;
    ec_affine_to_bytes(&pub, out, 1);
    return 1;
}

/*
 * Parse a path such as "m/44'/0'/0'/0/0" (hardened marked by ' or h) into
 * child indexes. Returns the depth, or -1 if the string is malformed or
 * longer than max_depth.
 */
HASH_HD int bip32_parse_path(const char *s, uint32_t *path, int max_depth)
{
    int depth = 0;

    if (*s == 'm' || *s == 'M') {
        s++;
        if (*s == '\0')
            return 0;
        if (*s != '/')
            return -1;
        s++;
    }
    while (*s) {
        if (depth == max_depth || *s < '0' || *s > '9')
            return -1;
        uint32_t v = 0;
        while (*s >= '0' && *s <= '9') {
            if (v > (BIP32_HARDENED - 1) / 10)
                return -1;
            v = v * 10 + (uint32_t)(*s++ - '0');
            if (v >= BIP32_HARDENED)
                return -1;
        }
        if (*s == '\'' || *s == 'h' || *s == 'H') {
            v |= BIP32_HARDENED;
            s++;
        }
        path[depth++] = v;
        if (*s == '/') {
            s++;
            if (*s == '\0')
                return -1;
        } else if (*s != '\0') {
            return -1;
        }
    }
    return depth;
}

#endif
