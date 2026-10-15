# Contributing

Thanks for taking a look. The library is small on purpose. The bar for a
change is that a reviewer can read it in one sitting and that a test proves
it against OpenSSL.

## Ground rules

- Every function is `__host__ __device__`, allocates nothing, is safe to call
  with the result aliasing an input, and reports failure by returning 0. No
  `printf`, no traps, no global state.
- Where OpenSSL has the operation, the semantics match OpenSSL exactly:
  truncated division, `BN_nnmod`, word-by-word Montgomery reduction, the
  point formulas of `ecp_smpl.c` and the ladder of `ec_mult.c`. Any
  deviation is stated in the function's comment.
- Fixed width. A `bn_t` is 8 words. A value that does not fit is an error,
  not a truncation.
- No side-channel claims. A routine whose running time depends on secret
  data carries `_vartime` in its name or says so in its comment.
- Plain C-style C++: no templates or classes in the public headers, no new
  macros beyond the handful that exist, no dependencies beyond the CUDA
  runtime. The tests may use C++ and link `libcrypto`.

## Adding or changing a function

1. Put it in the header it belongs to (`bn`, `mont`, `ec`, `hash`, `kdf`,
   `bip39`, `bip32`, `address`) with a short comment: what it computes, what
   it returns on error, and which OpenSSL function or specification section
   it mirrors.
2. Cover it in the matching `tests/test_*.cu`. Every test follows the same
   shape: a batch of inputs is built on the host, a `__host__ __device__`
   `compute()` function runs the library over one input, a kernel runs it on
   the GPU for the whole batch, the host runs it again, and both results are
   compared with OpenSSL. Inputs mix hand-picked edge values (0, 1, word
   boundaries, `p`, `n`, 2^256 - 1, negatives where signs exist) with
   seeded random values. Check the aliased form (`r == a`) too.
3. When OpenSSL has no reference (BIP32, Bech32), write the reference in the
   test from OpenSSL primitives and add the published test vectors.
4. Update the API table and the test description in `README.md`.

## Build and test

```sh
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89   # your GPU's architecture
cmake --build build -j
cd build && ctest --output-on-failure
```

Before opening a pull request, run the suite at the default size and once
with a larger batch and another seed:

```sh
CUDA_SECP256K1_N=65536 CUDA_SECP256K1_SEED=2 ctest --output-on-failure
```

`CUDA_SECP256K1_HOST_ONLY=1` runs the kernels' work on the CPU instead,
which is what CI does because its runners have no GPU. It checks the
arithmetic, not the device compilation; a real GPU run is still expected
for anything that touches the headers.

## Style

- Four-space indentation, braces as in the existing code, lines up to about
  110 characters.
- Comments explain why, or cite the specification or OpenSSL function being
  followed. No commented-out code, no debugging output.
- Commit messages: an imperative subject line, then a body saying what
  changed, why, and which test covers it.

## Scope

In scope: anything a GPU thread needs to derive and encode secp256k1 keys,
as long as it stays readable and verified. Performance work is welcome when
it keeps the API, the differential tests and the readability, for example a
dedicated secp256k1 field layer behind the existing `fe_*` functions.

Out of scope: a port of the OpenSSL API, constant-time hardening, and
clever batching that makes the code hard to follow.

## Reporting a problem

Open an issue with the failing test's output, the `CUDA_SECP256K1_N` and
`CUDA_SECP256K1_SEED` values, the index of the failing input, and your GPU,
CUDA and OpenSSL versions. The seeded inputs make the failure reproducible.
