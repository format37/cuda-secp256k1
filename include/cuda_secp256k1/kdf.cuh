/*
 * kdf.cuh - PBKDF2 (RFC 8018) over HMAC-SHA256 and HMAC-SHA512.
 *
 * The password is absorbed into the HMAC state once; every iteration then
 * starts from a copy of that state, so an iteration costs two compression
 * calls of the underlying hash, the same as OpenSSL's implementation.
 */
#ifndef CUDA_SECP256K1_KDF_CUH
#define CUDA_SECP256K1_KDF_CUH

#include "hash.cuh"

/*
 * Derive out_len bytes from password and salt. Returns 0 if iterations is
 * zero: RFC 8018 requires at least one, and OpenSSL rejects zero as well.
 *
 * Block i of the output is T_i = U_1 ^ U_2 ^ ... ^ U_c, where
 * U_1 = HMAC(password, salt || i) and U_j = HMAC(password, U_{j-1}).
 */
HASH_HD int pbkdf2_hmac_sha256(const uint8_t *password, size_t password_len, const uint8_t *salt,
                               size_t salt_len, uint32_t iterations, uint8_t *out, size_t out_len)
{
    if (iterations == 0)
        return 0;

    hmac_sha256_ctx_t keyed;
    hmac_sha256_init(&keyed, password, password_len);

    for (uint32_t i = 1; out_len > 0; i++) {
        uint8_t u[SHA256_DIGEST_SIZE], t[SHA256_DIGEST_SIZE];
        uint8_t index[4] = { (uint8_t)(i >> 24), (uint8_t)(i >> 16), (uint8_t)(i >> 8), (uint8_t)i };

        hmac_sha256_ctx_t c = keyed;
        hmac_sha256_update(&c, salt, salt_len);
        hmac_sha256_update(&c, index, sizeof(index));
        hmac_sha256_final(&c, u);
        memcpy(t, u, sizeof(t));

        for (uint32_t j = 1; j < iterations; j++) {
            c = keyed;
            hmac_sha256_update(&c, u, sizeof(u));
            hmac_sha256_final(&c, u);
            for (int k = 0; k < SHA256_DIGEST_SIZE; k++)
                t[k] ^= u[k];
        }

        size_t n = out_len < SHA256_DIGEST_SIZE ? out_len : SHA256_DIGEST_SIZE;
        memcpy(out, t, n);
        out += n;
        out_len -= n;
    }
    return 1;
}

HASH_HD int pbkdf2_hmac_sha512(const uint8_t *password, size_t password_len, const uint8_t *salt,
                               size_t salt_len, uint32_t iterations, uint8_t *out, size_t out_len)
{
    if (iterations == 0)
        return 0;

    hmac_sha512_ctx_t keyed;
    hmac_sha512_init(&keyed, password, password_len);

    for (uint32_t i = 1; out_len > 0; i++) {
        uint8_t u[SHA512_DIGEST_SIZE], t[SHA512_DIGEST_SIZE];
        uint8_t index[4] = { (uint8_t)(i >> 24), (uint8_t)(i >> 16), (uint8_t)(i >> 8), (uint8_t)i };

        hmac_sha512_ctx_t c = keyed;
        hmac_sha512_update(&c, salt, salt_len);
        hmac_sha512_update(&c, index, sizeof(index));
        hmac_sha512_final(&c, u);
        memcpy(t, u, sizeof(t));

        for (uint32_t j = 1; j < iterations; j++) {
            c = keyed;
            hmac_sha512_update(&c, u, sizeof(u));
            hmac_sha512_final(&c, u);
            for (int k = 0; k < SHA512_DIGEST_SIZE; k++)
                t[k] ^= u[k];
        }

        size_t n = out_len < SHA512_DIGEST_SIZE ? out_len : SHA512_DIGEST_SIZE;
        memcpy(out, t, n);
        out += n;
        out_len -= n;
    }
    return 1;
}

#endif
