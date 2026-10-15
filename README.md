# cuda-secp256k1

Big-number and secp256k1 elliptic-curve arithmetic for CUDA, header-only,
with every operation checked against OpenSSL.

It gives a GPU thread what it needs to do public-key math on its own:
256-bit integers in registers, Montgomery multiplication, Jacobian point
addition and doubling, a Montgomery-ladder scalar multiplication, and SHA-256,
SHA-512, HMAC and PBKDF2 for the key-derivation steps around them. All of
it is plain `__host__ __device__` code, so the same functions run and can be
debugged on the CPU.

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
| `hash.cuh` | SHA-256 and SHA-512, streaming (`sha256_init` / `sha256_update` / `sha256_final`, likewise `sha512_*`) and one-shot (`sha256`, `sha512`); HMAC for both (`hmac_sha256_init` / `_update` / `_final`, `hmac_sha256`, likewise `hmac_sha512_*`) |
| `kdf.cuh` | `pbkdf2_hmac_sha256`, `pbkdf2_hmac_sha512` (RFC 8018, any output length) |

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
  boundaries, `p`, `n`, 2^256 - 1, ...) with every sign combination, plus
  random operands of up to 512 bits.
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
- `test_hash`: SHA-256, SHA-512 and both HMACs for every message length from
  0 to 400 bytes and random lengths up to 1 KiB, each computed one-shot, split
  across two updates and one byte at a time, with keys from empty to longer
  than a block, on both GPU and host, against `SHA256()` / `SHA512()` /
  `HMAC()`.
- `test_pbkdf2`: PBKDF2 over both HMACs for password and salt lengths from
  empty to longer than a block, output lengths from 1 to 160 bytes and 1 to
  4096 iterations, on GPU and host, against `PKCS5_PBKDF2_HMAC()`.

`CUDA_SECP256K1_N` sets the batch size (default 4096) and
`CUDA_SECP256K1_SEED` the random seed, so a failing input can be reproduced.

## Performance

`bench_scalar_mul` derives one public key per thread. On an RTX 4090
(CUDA 12.5, 262144 keys):

| Method | keys / s |
|---|---|
| `ec_point_mul` (Montgomery ladder) | ~480 000 |
| `ec_point_mul_vartime` (double-and-add) | ~375 000 |

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
