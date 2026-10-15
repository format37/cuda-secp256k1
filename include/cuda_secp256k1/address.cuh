/*
 * address.cuh - address encodings on top of hash160.
 *
 *   Base58Check (legacy Bitcoin P2PKH, "1...")
 *   Bech32 (BIP173): segwit v0 P2WPKH ("bc1q...") and Avalanche X/P-chain
 *   ("avax1...", written with a "P-" or "X-" chain prefix by the wallet).
 *
 * All functions write a NUL-terminated string and return its length, or 0
 * if the output buffer is too small or an input is out of range.
 */
#ifndef CUDA_SECP256K1_ADDRESS_CUH
#define CUDA_SECP256K1_ADDRESS_CUH

#include "hash.cuh"

/* ---- Base58Check -------------------------------------------------------- */

#define BASE58CHECK_MAX_PAYLOAD 80
#define BASE58CHECK_MAX_LEN 120 /* enough for the longest payload above plus NUL */

/* payload || first 4 bytes of sha256d(payload), in base 58; leading zero bytes become '1'. */
HASH_HD int base58check_encode(const uint8_t *payload, size_t len, char *out, size_t out_size)
{
    const char *alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    uint8_t buf[BASE58CHECK_MAX_PAYLOAD + 4];
    uint8_t checksum[SHA256_DIGEST_SIZE];
    uint8_t digits[BASE58CHECK_MAX_LEN]; /* base-58 digits, least significant first */
    int ndigits = 0, zeros = 0;

    if (len > BASE58CHECK_MAX_PAYLOAD)
        return 0;
    memcpy(buf, payload, len);
    sha256d(payload, len, checksum);
    memcpy(buf + len, checksum, 4);
    len += 4;

    while ((size_t)zeros < len && buf[zeros] == 0)
        zeros++;

    /* Schoolbook base conversion: fold each byte into the digit array. */
    for (size_t i = (size_t)zeros; i < len; i++) {
        uint32_t carry = buf[i];
        for (int k = 0; k < ndigits; k++) {
            carry += (uint32_t)digits[k] << 8;
            digits[k] = (uint8_t)(carry % 58);
            carry /= 58;
        }
        while (carry) {
            if (ndigits == BASE58CHECK_MAX_LEN)
                return 0;
            digits[ndigits++] = (uint8_t)(carry % 58);
            carry /= 58;
        }
    }

    if ((size_t)(zeros + ndigits + 1) > out_size)
        return 0;
    int n = 0;
    for (int i = 0; i < zeros; i++)
        out[n++] = '1';
    for (int i = ndigits - 1; i >= 0; i--)
        out[n++] = alphabet[digits[i]];
    out[n] = '\0';
    return n;
}

/* ---- Bech32 (BIP173) ---------------------------------------------------- */

#define BECH32_MAX_LEN 90
#define BECH32_MAX_DATA 65 /* 5-bit groups: 40 bytes of payload after conversion */

HASH_HD uint32_t bech32_polymod_step(uint32_t chk, uint8_t v)
{
    uint8_t top = (uint8_t)(chk >> 25);
    chk = ((chk & 0x1ffffff) << 5) ^ v;
    if (top & 1)
        chk ^= 0x3b6a57b2;
    if (top & 2)
        chk ^= 0x26508e6d;
    if (top & 4)
        chk ^= 0x1ea119fa;
    if (top & 8)
        chk ^= 0x3d4233dd;
    if (top & 16)
        chk ^= 0x2a1462b3;
    return chk;
}

/* Regroup bytes into 5-bit values, zero-padding the last group. Returns the number of groups. */
HASH_HD int bech32_convert_8to5(const uint8_t *in, size_t len, uint8_t *out)
{
    uint32_t acc = 0;
    int bits = 0, n = 0;
    for (size_t i = 0; i < len; i++) {
        acc = (acc << 8) | in[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            out[n++] = (uint8_t)((acc >> bits) & 31);
        }
    }
    if (bits)
        out[n++] = (uint8_t)((acc << (5 - bits)) & 31);
    return n;
}

/*
 * hrp || '1' || data || 6-character checksum. hrp must be lowercase ASCII in
 * 0x21..0x7e, data values must be < 32.
 */
HASH_HD int bech32_encode(const char *hrp, const uint8_t *data, size_t len, char *out, size_t out_size)
{
    const char *charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    size_t hrp_len = 0;
    uint32_t chk = 1;

    while (hrp[hrp_len])
        hrp_len++;
    if (hrp_len == 0 || hrp_len + 1 + len + 6 > BECH32_MAX_LEN || hrp_len + 1 + len + 6 + 1 > out_size)
        return 0;
    for (size_t i = 0; i < hrp_len; i++)
        if (hrp[i] < 0x21 || hrp[i] > 0x7e || (hrp[i] >= 'A' && hrp[i] <= 'Z'))
            return 0;
    for (size_t i = 0; i < len; i++)
        if (data[i] >= 32)
            return 0;

    for (size_t i = 0; i < hrp_len; i++)
        chk = bech32_polymod_step(chk, (uint8_t)(hrp[i] >> 5));
    chk = bech32_polymod_step(chk, 0);
    for (size_t i = 0; i < hrp_len; i++)
        chk = bech32_polymod_step(chk, (uint8_t)(hrp[i] & 31));
    for (size_t i = 0; i < len; i++)
        chk = bech32_polymod_step(chk, data[i]);
    for (int i = 0; i < 6; i++)
        chk = bech32_polymod_step(chk, 0);
    chk ^= 1;

    int n = 0;
    for (size_t i = 0; i < hrp_len; i++)
        out[n++] = hrp[i];
    out[n++] = '1';
    for (size_t i = 0; i < len; i++)
        out[n++] = charset[data[i]];
    for (int i = 0; i < 6; i++)
        out[n++] = charset[(chk >> (5 * (5 - i))) & 31];
    out[n] = '\0';
    return n;
}

/* ---- addresses from a compressed public key ----------------------------- */

/* Base58Check(version || hash160(pub)); version 0x00 gives a mainnet "1..." address. */
HASH_HD int address_p2pkh(uint8_t version, const uint8_t pub[33], char *out, size_t out_size)
{
    uint8_t payload[21];
    payload[0] = version;
    hash160(pub, 33, payload + 1);
    return base58check_encode(payload, sizeof(payload), out, out_size);
}

/* Segwit v0 P2WPKH: bech32(hrp, 0 || hash160(pub) in 5-bit groups); hrp "bc" gives "bc1q...". */
HASH_HD int address_p2wpkh(const char *hrp, const uint8_t pub[33], char *out, size_t out_size)
{
    uint8_t h[RIPEMD160_DIGEST_SIZE];
    uint8_t data[1 + 32];
    hash160(pub, 33, h);
    data[0] = 0; /* witness version */
    int n = 1 + bech32_convert_8to5(h, sizeof(h), data + 1);
    return bech32_encode(hrp, data, n, out, out_size);
}

/* Avalanche X/P-chain: bech32(hrp, hash160(pub) in 5-bit groups); hrp "avax" for mainnet. */
HASH_HD int address_avalanche(const char *hrp, const uint8_t pub[33], char *out, size_t out_size)
{
    uint8_t h[RIPEMD160_DIGEST_SIZE];
    uint8_t data[32];
    hash160(pub, 33, h);
    int n = bech32_convert_8to5(h, sizeof(h), data);
    return bech32_encode(hrp, data, n, out, out_size);
}

#endif
