/*
 * bip39.cuh - BIP39 seed from a mnemonic sentence.
 *
 *   seed = PBKDF2-HMAC-SHA512(mnemonic, "mnemonic" || passphrase, 2048, 64 bytes)
 *
 * The mnemonic and passphrase are used as given. BIP39 requires both to be
 * NFKD-normalized UTF-8, which is the identity for the English word list
 * and ASCII passphrases. Word-list and checksum validation are not part of
 * this function.
 */
#ifndef CUDA_SECP256K1_BIP39_CUH
#define CUDA_SECP256K1_BIP39_CUH

#include "kdf.cuh"

#define BIP39_SEED_SIZE 64
#define BIP39_ITERATIONS 2048
#define BIP39_MAX_PASSPHRASE 248

/* Returns 0 if the passphrase is longer than BIP39_MAX_PASSPHRASE bytes. */
HASH_HD int bip39_seed(const char *mnemonic, size_t mnemonic_len, const char *passphrase,
                       size_t passphrase_len, uint8_t seed[BIP39_SEED_SIZE])
{
    uint8_t salt[8 + BIP39_MAX_PASSPHRASE];
    if (passphrase_len > BIP39_MAX_PASSPHRASE)
        return 0;
    memcpy(salt, "mnemonic", 8);
    memcpy(salt + 8, passphrase, passphrase_len);
    return pbkdf2_hmac_sha512((const uint8_t *)mnemonic, mnemonic_len, salt, 8 + passphrase_len,
                              BIP39_ITERATIONS, seed, BIP39_SEED_SIZE);
}

#endif
