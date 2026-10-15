# cuda-secp256k1

[![CI](https://github.com/format37/cuda-secp256k1/actions/workflows/ci.yml/badge.svg)](https://github.com/format37/cuda-secp256k1/actions/workflows/ci.yml)

Big-number and secp256k1 elliptic-curve arithmetic for CUDA, header-only,
with every operation checked against OpenSSL.

It gives a GPU thread what it needs to do public-key math on its own:
256-bit integers in registers, Montgomery multiplication, Jacobian point
addition and doubling, a Montgomery-ladder scalar multiplication, and the
hashing and key-derivation steps around them: SHA-256, SHA-512, RIPEMD-160,
HMAC, PBKDF2, BIP39 seeds, BIP32 child keys and Base58Check / Bech32
addresses. All of it is plain `__host__ __device__` code, so the same
functions run and can be debugged on the CPU.

```cuda
#include <cuda_secp256k1/secp256k1.cuh>

__constant__ ec_group_t c_group;   /* filled once from the host */

__global__ void pubkeys(const bn_t *priv, uint8_t (*out)[33], int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ec_affine_t pub;
    if (ec_pubkey_create(&c_group, &pub, &priv[i]))   /* pub = priv * G */
        ec_affine_to_bytes(&pub, out[i], 1);          /* 02/03 || x */
}

/* host side */
ec_group_t g;
ec_group_init_secp256k1(&g);
cudaMemcpyToSymbol(c_group, &g, sizeof(g));
```

## What it is, and is not

- **Fixed width, no allocation.** A `bn_t` is 8 x 64-bit words (512 bits,
  enough for the product of two 256-bit values) plus a word count and sign.
  Values that do not fit are reported, not truncated.
- **Same semantics as OpenSSL's BN and EC code**, so results can be compared
  one to one: truncated division, `BN_nnmod`, Montgomery reduction word by
  word, the `ossl_ec_GFp_simple_*` point formulas and the
  `ossl_ec_scalar_mul_ladder` scalar multiplication.
- **Not a port of the OpenSSL API.** There is no `BN_CTX`, no error queue, no
  variable-size numbers. The algorithms were reimplemented for fixed-size
  values that fit a GPU thread.
- **Not side-channel hardened.** The ladder performs the same field operations
  for every scalar, but there is no blinding and no attempt at constant-time
  memory access. This is a library for bulk computation (key derivation,
  batch verification, research), not for protecting a secret on shared
  hardware.
- **Moduli up to 256 bits.** Montgomery contexts take moduli of at most 4
  words; the generic `bn_*` functions work on anything up to 512 bits.

## API overview

| Header | Contents |
|---|---|
| `bn.cuh` | `bn_t`; `bn_add`, `bn_sub`, `bn_mul`, `bn_div` (Knuth D), `bn_mod`, `bn_nnmod`, shifts, `bn_cmp`, `bn_mod_inverse` (extended Euclid), the `*_quick` modular helpers, `bn_consttime_swap`, hex and byte conversion |
| `mont.cuh` | `bn_mont_t` context (`bn_mont_init`), `bn_mont_mul`, `bn_to_mont`, `bn_from_mont`, `bn_mont_exp`, `bn_mod_exp_mont` |
| `ec.cuh` | `ec_group_t` (`ec_group_init_secp256k1`, or `ec_group_init` for another short-Weierstrass curve with cofactor 1), field helpers `fe_*`, `ec_point_t` (Jacobian, Montgomery form) and `ec_affine_t`, `ec_point_add`, `ec_point_dbl`, `ec_point_mul` (ladder), `ec_point_mul_vartime` (double-and-add), `ec_point_to_affine`, `ec_pubkey_create`, `ec_affine_to_bytes`, `ec_affine_is_on_curve` |
| `hash.cuh` | SHA-256, SHA-512 and RIPEMD-160, streaming (`sha256_init` / `sha256_update` / `sha256_final`, likewise `sha512_*`) and one-shot (`sha256`, `sha512`); HMAC for both (`hmac_sha256_init` / `_update` / `_final`, `hmac_sha256`, likewise `hmac_sha512_*`); `hash160`, `sha256d` |
| `kdf.cuh` | `pbkdf2_hmac_sha256`, `pbkdf2_hmac_sha512` (RFC 8018, any output length) |
| `bip39.cuh` | `bip39_seed` (mnemonic and passphrase to the 64-byte seed) |
| `bip32.cuh` | `bip32_key_t`, `bip32_master`, `bip32_ckd_priv` (hardened and normal), `bip32_derive_path`, `bip32_pubkey`, `bip32_parse_path`; private derivation only, no xprv/xpub strings |
| `address.cuh` | `base58check_encode`, `bech32_encode`, `address_p2pkh` ("1..."), `address_p2wpkh` ("bc1q..."), `address_avalanche` ("avax1...") |

Every function is safe to call with the result aliasing an input.

## Build and test

Requirements: CUDA 12 or newer (tested with 12.5), CMake 3.18+, and OpenSSL
development headers for the tests (they link `libcrypto`). CMake refuses
older toolkits: CUDA 11.5 cannot target recent GPUs, and in our tests its
optimizer produced wrong HMAC-SHA512 digests for some inputs.

```sh
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89   # 89 = RTX 40xx, 86 = RTX 30xx, 80 = A100
cmake --build build -j
cd build && ctest --output-on-failure
```

If a distribution package installed an older `nvcc` in `/usr/bin`, CMake may
pick it over the one on your `PATH`. Add
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` to the first command; the
configure output names the compiler it found.

The tests run each operation on the GPU over a batch of inputs and compare
every result with OpenSSL on the host:

- `test_bn`: add, sub, mul, div/mod, nnmod, shifts, modular inverse, the
  `*_quick` helpers and conversions, on all pairs of edge values (0, 1, word
  boundaries up to the full 512 bits, `p`, `n`, 2^256 - 1, ...) with every
  sign combination, divisions built to reach the rare correction steps of
  long division, plus random operands of up to 512 bits, divisors included.
- `test_mont`: context setup (`n0`, `R mod n`, `R^2 mod n`), conversions,
  Montgomery products and exponentiation against `BN_MONT_CTX` /
  `BN_mod_mul_montgomery` / `BN_mod_exp`, for `p`, `n`, moduli of 1 to 4 words
  and some degenerate ones.
- `test_ec`: scalar multiplication (ladder and double-and-add) for scalars
  including 0, 1, n-1, n, n+1, 2^256-1, negative and 512-bit values; point
  addition and doubling in every Z-coordinate branch (both inputs affine,
  one projective, both projective, equal points, inverse points, infinity);
  public-key derivation and compressed encoding, all against
  `EC_POINT_mul` / `EC_POINT_add` / `EC_POINT_point2oct`.
- `test_hash`: SHA-256, SHA-512, RIPEMD-160, both HMACs and hash160 for every message length from
  0 to 400 bytes and random lengths up to 1 KiB, each computed one-shot, split
  across two updates and one byte at a time, with keys from empty to longer
  than a block, on both GPU and host, against `SHA256()` / `SHA512()` /
  `RIPEMD160()` / `HMAC()`.
- `test_pbkdf2`: PBKDF2 over both HMACs for password and salt lengths from
  empty to longer than a block, output lengths from 1 to 160 bytes and 1 to
  4096 iterations, on GPU and host, against `PKCS5_PBKDF2_HMAC()`.
- `test_bip32`: BIP39 seeds from random mnemonics and passphrases against
  `PKCS5_PBKDF2_HMAC()`; BIP32 master keys and random paths of depth 0 to 6
  with hardened and normal indexes against a reference written on OpenSSL
  primitives (`HMAC()`, `BN_mod_add()`, `EC_POINT_mul()`,
  `EC_POINT_point2oct()`); the published BIP32 and BIP39 test vectors.
- `test_address`: Base58Check against a reference on `SHA256()` and
  `BN_div_word()`; Bech32 against the BIP173 vectors and an independent
  decoder; P2PKH, P2WPKH and Avalanche addresses of random keys with the
  hash160 from OpenSSL; the addresses and WIF strings of private key 1.

`CUDA_SECP256K1_N` sets the batch size (default 4096) and
`CUDA_SECP256K1_SEED` the random seed, so a failing input can be reproduced.
`CUDA_SECP256K1_HOST_ONLY=1` runs the kernels' work on the CPU with the same
code, which is how CI runs the suite on machines without a GPU.

See `CONTRIBUTING.md` for how to add a function and its test.

## Examples

- `derive_pubkey`: hex private keys in, compressed public keys out.
- `mnemonic_to_key`: BIP39 mnemonics in (one per line on stdin, or as
  arguments), private key and compressed public key at a BIP32 path out.
  `-p` sets the passphrase and `-d` the path, default `m/44'/0'/0'/0/0`.
  Mnemonics are used as given; there is no word-list or checksum validation.
- `mnemonic_to_address`: the same pipeline ending in an address: `-f p2pkh`
  (`1...`, path m/44'/0'/0'/0/0), `-f p2wpkh` (`bc1q...`, m/84'/0'/0'/0/0) or
  `-f avax` (`P-avax1...`, m/44'/9000'/0'/0/0).

```sh
echo "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  | ./build/mnemonic_to_key -p TREZOR
./build/mnemonic_to_address -f p2wpkh "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
# bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu abandon abandon ...
```

## Performance

`bench_scalar_mul` derives one public key per thread and `mnemonic_to_key`
runs the whole BIP39 and BIP32 pipeline per thread. On an RTX 4090 (CUDA
12.5, 262144 inputs):

| Workload | per second |
|---|---|
| `ec_point_mul` (Montgomery ladder) | ~480 000 keys |
| `ec_point_mul_vartime` (double-and-add) | ~375 000 keys |
| `mnemonic_to_key` or `mnemonic_to_address`: BIP39 seed (2048 PBKDF2 rounds), m/44'/0'/0'/0/0 (three scalar multiplications) and the address | ~140 000 mnemonics |

The arithmetic is generic multi-precision code (64-bit limbs, `__int128`
products), not a hand-tuned secp256k1 field implementation, so there is room
for a faster special-purpose field layer; the point of this library is that
each piece is simple to read and is verified.

## Design notes

- `bn_t` keeps OpenSSL's invariants: `top` counts significant words, words
  above `top` are zero, a zero has `top == 0` and is never negative.
- Division is schoolbook long division with 64-bit quotient digits (Knuth,
  TAOCP vol. 2, algorithm D). It is only used by the generic modular code;
  the field arithmetic uses Montgomery multiplication and never divides.
- A Montgomery context stores `n`, `n0 = -n^-1 mod 2^64`, `R mod n` and
  `R^2 mod n` with `R = 2^(64 * words)`. `n0` comes from Newton's iteration on
  one word; the other two from repeated doubling, so setup needs no division.
- Field inversion is Fermat's little theorem (`a^(p-2)`) in Montgomery form.
- Points are Jacobian with coordinates in Montgomery form, like OpenSSL's
  `EC_GFp_mont_method`. The ladder works in homogeneous (X : Z) coordinates
  and recovers `y` at the end exactly as OpenSSL does.

## Origins

The code grew out of [avax-pw-crack](https://github.com/format37/avax-pw-crack),
a CUDA passphrase-recovery tool that needed secp256k1 on the GPU. This
repository is that arithmetic extracted, rewritten for clarity, and put under
differential tests. The point and ladder formulas follow OpenSSL's
`crypto/ec/ecp_smpl.c` and `crypto/ec/ec_mult.c`; the Montgomery reduction
follows `crypto/bn/bn_mont.c`.

## License

Apache License 2.0, see `LICENSE`.
