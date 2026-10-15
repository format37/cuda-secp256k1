/*
 * bn.cuh - fixed-size multi-precision integers for CUDA.
 *
 * A bn_t holds a signed integer of up to BN_WORDS 64-bit words in a plain
 * struct, so it lives in registers/local memory and needs no allocation.
 * Every function is __host__ __device__, which lets the same code run in a
 * kernel or in ordinary host code (the tests use that for debugging).
 *
 * Representation invariants, kept by every function here:
 *   - d[0] is the least significant word.
 *   - top is the number of significant words; words at index >= top are zero.
 *   - top == 0 means the value is zero, and a zero is never negative.
 *
 * The semantics follow OpenSSL's BN_* functions (truncated division, sign of
 * the remainder equal to the sign of the dividend, BN_nnmod, the *_quick
 * modular helpers), so results can be checked against OpenSSL directly.
 *
 * All functions are safe to call with the result aliasing an input.
 */
#ifndef CUDA_SECP256K1_BN_CUH
#define CUDA_SECP256K1_BN_CUH

#include <stdint.h>
#include <stdio.h>

#ifndef BN_WORDS
#define BN_WORDS 8 /* 512 bits: enough for the product of two 256-bit values */
#endif
#define BN_WORD_BITS 64
#define BN_MAX_BITS (BN_WORDS * BN_WORD_BITS)

typedef uint64_t bn_word;
typedef unsigned __int128 bn_dword;

#define BN_HD static __host__ __device__ inline

typedef struct {
    bn_word d[BN_WORDS];
    int top;
    int neg;
} bn_t;

/* ---- basic queries and assignment ------------------------------------- */

BN_HD void bn_zero(bn_t *a)
{
    for (int i = 0; i < BN_WORDS; i++)
        a->d[i] = 0;
    a->top = 0;
    a->neg = 0;
}

BN_HD void bn_set_word(bn_t *a, bn_word w)
{
    bn_zero(a);
    a->d[0] = w;
    a->top = (w != 0);
}

BN_HD void bn_copy(bn_t *r, const bn_t *a)
{
    if (r != a)
        *r = *a;
}

/* Recompute top from the stored words and clear the sign of a zero. */
BN_HD void bn_normalize(bn_t *a)
{
    while (a->top > 0 && a->d[a->top - 1] == 0)
        a->top--;
    if (a->top == 0)
        a->neg = 0;
}

/* Zero the words from index `from` upwards. The clamp tells gcc the range is bounded, which avoids a bogus overflow warning. */
BN_HD void bn_clear_from(bn_t *r, int from)
{
    for (int i = from < 0 ? 0 : from; i < BN_WORDS; i++)
        r->d[i] = 0;
}

BN_HD int bn_is_zero(const bn_t *a) { return a->top == 0; }
BN_HD int bn_is_one(const bn_t *a) { return a->top == 1 && a->d[0] == 1 && !a->neg; }
BN_HD int bn_is_odd(const bn_t *a) { return a->top > 0 && (a->d[0] & 1); }
BN_HD int bn_is_negative(const bn_t *a) { return a->neg; }

BN_HD int bn_num_bits_word(bn_word w)
{
    if (w == 0)
        return 0;
#ifdef __CUDA_ARCH__
    return BN_WORD_BITS - __clzll((long long)w);
#else
    return BN_WORD_BITS - __builtin_clzll(w);
#endif
}

BN_HD int bn_num_bits(const bn_t *a)
{
    if (a->top == 0)
        return 0;
    return (a->top - 1) * BN_WORD_BITS + bn_num_bits_word(a->d[a->top - 1]);
}

BN_HD int bn_is_bit_set(const bn_t *a, int n)
{
    if (n < 0)
        return 0;
    int w = n / BN_WORD_BITS;
    if (w >= a->top)
        return 0;
    return (int)((a->d[w] >> (n % BN_WORD_BITS)) & 1);
}

/* Compare magnitudes: -1 if |a| < |b|, 0 if equal, 1 if |a| > |b|. */
BN_HD int bn_ucmp(const bn_t *a, const bn_t *b)
{
    if (a->top != b->top)
        return a->top > b->top ? 1 : -1;
    for (int i = a->top - 1; i >= 0; i--) {
        if (a->d[i] != b->d[i])
            return a->d[i] > b->d[i] ? 1 : -1;
    }
    return 0;
}

/* Signed compare: -1 if a < b, 0 if equal, 1 if a > b. */
BN_HD int bn_cmp(const bn_t *a, const bn_t *b)
{
    if (a->neg != b->neg)
        return a->neg ? -1 : 1;
    int c = bn_ucmp(a, b);
    return a->neg ? -c : c;
}

/* ---- word-array primitives --------------------------------------------- */

/* r[0..n) = a[0..n) - b[0..n); returns the borrow out (0 or 1). */
BN_HD bn_word bn_sub_words(bn_word *r, const bn_word *a, const bn_word *b, int n)
{
    bn_word borrow = 0;
    for (int i = 0; i < n; i++) {
        bn_dword s = (bn_dword)a[i] - b[i] - borrow;
        r[i] = (bn_word)s;
        borrow = (bn_word)(s >> BN_WORD_BITS) & 1;
    }
    return borrow;
}

/* r[0..n) += a[0..n) * w; returns the carry out (a full word). */
BN_HD bn_word bn_mul_add_words(bn_word *r, const bn_word *a, int n, bn_word w)
{
    bn_word carry = 0;
    for (int i = 0; i < n; i++) {
        bn_dword t = (bn_dword)a[i] * w + r[i] + carry;
        r[i] = (bn_word)t;
        carry = (bn_word)(t >> BN_WORD_BITS);
    }
    return carry;
}

/* ---- unsigned add / sub ------------------------------------------------ */

/* r = |a| + |b|. Returns 0 if the result does not fit in BN_WORDS words. */
BN_HD int bn_uadd(bn_t *r, const bn_t *a, const bn_t *b)
{
    const bn_t *x = a, *y = b;
    if (x->top < y->top) {
        x = b;
        y = a;
    }
    int top = x->top;
    bn_word carry = 0;
    for (int i = 0; i < top; i++) {
        bn_word yi = (i < y->top) ? y->d[i] : 0;
        bn_dword s = (bn_dword)x->d[i] + yi + carry;
        r->d[i] = (bn_word)s;
        carry = (bn_word)(s >> BN_WORD_BITS);
    }
    if (carry) {
        if (top == BN_WORDS)
            return 0;
        r->d[top++] = carry;
    }
    bn_clear_from(r, top);
    r->top = top;
    r->neg = 0;
    return 1;
}

/* r = |a| - |b|; requires |a| >= |b|. */
BN_HD void bn_usub(bn_t *r, const bn_t *a, const bn_t *b)
{
    int top = a->top;
    bn_word borrow = 0;
    for (int i = 0; i < top; i++) {
        bn_word bi = (i < b->top) ? b->d[i] : 0;
        bn_dword s = (bn_dword)a->d[i] - bi - borrow;
        r->d[i] = (bn_word)s;
        borrow = (bn_word)(s >> BN_WORD_BITS) & 1;
    }
    bn_clear_from(r, top);
    r->top = top;
    r->neg = 0;
    bn_normalize(r);
}

/* ---- signed add / sub / mul -------------------------------------------- */

/* r = a + b. Returns 0 on overflow. */
BN_HD int bn_add(bn_t *r, const bn_t *a, const bn_t *b)
{
    if (a->neg == b->neg) {
        int neg = a->neg;
        if (!bn_uadd(r, a, b))
            return 0;
        r->neg = neg && r->top;
        return 1;
    }
    int c = bn_ucmp(a, b);
    if (c == 0) {
        bn_zero(r);
    } else if (c > 0) {
        int neg = a->neg;
        bn_usub(r, a, b);
        r->neg = neg && r->top;
    } else {
        int neg = b->neg;
        bn_usub(r, b, a);
        r->neg = neg && r->top;
    }
    return 1;
}

/* r = a - b. Returns 0 on overflow. */
BN_HD int bn_sub(bn_t *r, const bn_t *a, const bn_t *b)
{
    bn_t nb = *b;
    nb.neg = !b->neg && nb.top;
    return bn_add(r, a, &nb);
}

/* r = a * b. Returns 0 if a->top + b->top > BN_WORDS (r is left unchanged). */
BN_HD int bn_mul(bn_t *r, const bn_t *a, const bn_t *b)
{
    if (a->top == 0 || b->top == 0) {
        bn_zero(r);
        return 1;
    }
    if (a->top + b->top > BN_WORDS)
        return 0;
    bn_word t[BN_WORDS];
    for (int i = 0; i < BN_WORDS; i++)
        t[i] = 0;
    for (int i = 0; i < a->top; i++)
        t[i + b->top] = bn_mul_add_words(&t[i], b->d, b->top, a->d[i]);
    int neg = a->neg ^ b->neg;
    int top = a->top + b->top;
    for (int i = 0; i < BN_WORDS; i++)
        r->d[i] = t[i];
    r->top = top;
    r->neg = 0;
    bn_normalize(r);
    r->neg = neg && r->top;
    return 1;
}

BN_HD int bn_sqr(bn_t *r, const bn_t *a) { return bn_mul(r, a, a); }

/* ---- shifts ------------------------------------------------------------ */

/* r = a << 1. Returns 0 on overflow. */
BN_HD int bn_lshift1(bn_t *r, const bn_t *a)
{
    bn_word carry = 0;
    int top = a->top;
    for (int i = 0; i < top; i++) {
        bn_word w = a->d[i];
        r->d[i] = (w << 1) | carry;
        carry = w >> (BN_WORD_BITS - 1);
    }
    if (carry) {
        if (top == BN_WORDS)
            return 0;
        r->d[top++] = carry;
    }
    bn_clear_from(r, top);
    r->top = top;
    r->neg = a->neg;
    return 1;
}

/* r = |a| >> 1 with the sign of a kept (same as BN_rshift1). */
BN_HD void bn_rshift1(bn_t *r, const bn_t *a)
{
    int top = a->top;
    int neg = a->neg;
    bn_word carry = 0;
    for (int i = top - 1; i >= 0; i--) {
        bn_word w = a->d[i];
        r->d[i] = (w >> 1) | (carry << (BN_WORD_BITS - 1));
        carry = w & 1;
    }
    bn_clear_from(r, top);
    r->top = top;
    r->neg = neg;
    bn_normalize(r);
}

/* r = a << n. Returns 0 on overflow or n < 0. */
BN_HD int bn_lshift(bn_t *r, const bn_t *a, int n)
{
    if (n < 0)
        return 0;
    if (a->top == 0) {
        bn_zero(r);
        return 1;
    }
    if (bn_num_bits(a) + n > BN_MAX_BITS)
        return 0;
    int nw = n / BN_WORD_BITS, nb = n % BN_WORD_BITS;
    bn_t t;
    bn_zero(&t);
    for (int i = a->top - 1; i >= 0; i--) {
        bn_word w = a->d[i];
        if (nb) {
            if (i + nw + 1 < BN_WORDS)
                t.d[i + nw + 1] |= w >> (BN_WORD_BITS - nb);
            t.d[i + nw] |= w << nb;
        } else {
            t.d[i + nw] = w;
        }
    }
    t.top = a->top + nw + 1;
    if (t.top > BN_WORDS)
        t.top = BN_WORDS;
    t.neg = a->neg;
    bn_normalize(&t);
    *r = t;
    return 1;
}

/* r = |a| >> n with the sign of a kept (same as BN_rshift). Returns 0 if n < 0. */
BN_HD int bn_rshift(bn_t *r, const bn_t *a, int n)
{
    if (n < 0)
        return 0;
    int nw = n / BN_WORD_BITS, nb = n % BN_WORD_BITS;
    if (nw >= a->top) {
        bn_zero(r);
        return 1;
    }
    bn_t t;
    bn_zero(&t);
    int top = a->top - nw;
    for (int i = 0; i < top; i++) {
        bn_word lo = a->d[i + nw];
        bn_word hi = (i + nw + 1 < a->top) ? a->d[i + nw + 1] : 0;
        t.d[i] = nb ? ((lo >> nb) | (hi << (BN_WORD_BITS - nb))) : lo;
    }
    t.top = top;
    t.neg = a->neg;
    bn_normalize(&t);
    *r = t;
    return 1;
}

/* ---- division ---------------------------------------------------------- */

/*
 * q = a / b truncated toward zero, r = a - q * b (so r has the sign of a).
 * Either q or r may be NULL. Returns 0 if b is zero.
 * Schoolbook long division (Knuth, TAOCP vol. 2, algorithm D) on 64-bit words.
 */
BN_HD int bn_div(bn_t *q, bn_t *r, const bn_t *a, const bn_t *b)
{
    if (b->top == 0)
        return 0;

    int qneg = a->neg ^ b->neg;
    int rneg = a->neg;

    if (bn_ucmp(a, b) < 0) {
        bn_t rem = *a;
        if (q != NULL)
            bn_zero(q);
        if (r != NULL)
            *r = rem;
        return 1;
    }

    int n = b->top, m = a->top; /* m >= n >= 1 */
    bn_word qw[BN_WORDS], rw[BN_WORDS];
    for (int i = 0; i < BN_WORDS; i++)
        qw[i] = rw[i] = 0;

    if (n == 1) {
        bn_word dv = b->d[0];
        bn_dword rem = 0;
        for (int j = m - 1; j >= 0; j--) {
            bn_dword cur = (rem << BN_WORD_BITS) | a->d[j];
            qw[j] = (bn_word)(cur / dv);
            rem = cur % dv;
        }
        rw[0] = (bn_word)rem;
    } else {
        /* Normalize so that the top word of the divisor has its high bit set. */
        int s = BN_WORD_BITS - bn_num_bits_word(b->d[n - 1]);
        bn_word un[BN_WORDS + 1], vn[BN_WORDS];
        for (int i = n - 1; i > 0; i--)
            vn[i] = (b->d[i] << s) | (s ? b->d[i - 1] >> (BN_WORD_BITS - s) : 0);
        vn[0] = b->d[0] << s;
        un[m] = s ? a->d[m - 1] >> (BN_WORD_BITS - s) : 0;
        for (int i = m - 1; i > 0; i--)
            un[i] = (a->d[i] << s) | (s ? a->d[i - 1] >> (BN_WORD_BITS - s) : 0);
        un[0] = a->d[0] << s;

        for (int j = m - n; j >= 0; j--) {
            /* Estimate the quotient word from the top two words. */
            bn_dword num = ((bn_dword)un[j + n] << BN_WORD_BITS) | un[j + n - 1];
            bn_dword qhat = num / vn[n - 1];
            bn_dword rhat = num % vn[n - 1];
            while ((qhat >> BN_WORD_BITS) != 0
                   || qhat * vn[n - 2] > ((rhat << BN_WORD_BITS) | un[j + n - 2])) {
                qhat--;
                rhat += vn[n - 1];
                if ((rhat >> BN_WORD_BITS) != 0)
                    break;
            }

            /* un[j..j+n] -= qhat * vn */
            bn_word carry = 0, borrow = 0;
            for (int i = 0; i < n; i++) {
                bn_dword p = (bn_dword)(bn_word)qhat * vn[i] + carry;
                carry = (bn_word)(p >> BN_WORD_BITS);
                bn_dword t = (bn_dword)un[i + j] - (bn_word)p - borrow;
                un[i + j] = (bn_word)t;
                borrow = (bn_word)(t >> BN_WORD_BITS) & 1;
            }
            bn_dword t = (bn_dword)un[j + n] - carry - borrow;
            un[j + n] = (bn_word)t;
            qw[j] = (bn_word)qhat;

            if ((t >> BN_WORD_BITS) & 1) {
                /* The estimate was one too large: add the divisor back. */
                qw[j]--;
                carry = 0;
                for (int i = 0; i < n; i++) {
                    bn_dword s2 = (bn_dword)un[i + j] + vn[i] + carry;
                    un[i + j] = (bn_word)s2;
                    carry = (bn_word)(s2 >> BN_WORD_BITS);
                }
                un[j + n] += carry;
            }
        }

        /* Undo the normalization shift to get the remainder. */
        for (int i = 0; i < n; i++)
            rw[i] = (un[i] >> s) | (s ? un[i + 1] << (BN_WORD_BITS - s) : 0);
    }

    if (q != NULL) {
        for (int i = 0; i < BN_WORDS; i++)
            q->d[i] = qw[i];
        q->top = m;
        q->neg = 0;
        bn_normalize(q);
        q->neg = qneg && q->top;
    }
    if (r != NULL) {
        for (int i = 0; i < BN_WORDS; i++)
            r->d[i] = rw[i];
        r->top = n;
        r->neg = 0;
        bn_normalize(r);
        r->neg = rneg && r->top;
    }
    return 1;
}

/* r = a mod m with the sign of a (same as BN_mod). Returns 0 if m is zero. */
BN_HD int bn_mod(bn_t *r, const bn_t *a, const bn_t *m)
{
    return bn_div(NULL, r, a, m);
}

/* r = a mod m in [0, |m|) (same as BN_nnmod). Returns 0 if m is zero. */
BN_HD int bn_nnmod(bn_t *r, const bn_t *a, const bn_t *m)
{
    if (!bn_mod(r, a, m))
        return 0;
    if (!r->neg)
        return 1;
    return m->neg ? bn_sub(r, r, m) : bn_add(r, r, m);
}

/* ---- modular arithmetic ------------------------------------------------ */

/* r = (a + b) mod m. */
BN_HD int bn_mod_add(bn_t *r, const bn_t *a, const bn_t *b, const bn_t *m)
{
    bn_t t;
    if (!bn_add(&t, a, b))
        return 0;
    return bn_nnmod(r, &t, m);
}

/* r = (a - b) mod m. */
BN_HD int bn_mod_sub(bn_t *r, const bn_t *a, const bn_t *b, const bn_t *m)
{
    bn_t t;
    if (!bn_sub(&t, a, b))
        return 0;
    return bn_nnmod(r, &t, m);
}

/* r = (a * b) mod m. */
BN_HD int bn_mod_mul(bn_t *r, const bn_t *a, const bn_t *b, const bn_t *m)
{
    bn_t t;
    if (!bn_mul(&t, a, b))
        return 0;
    return bn_nnmod(r, &t, m);
}

BN_HD int bn_mod_sqr(bn_t *r, const bn_t *a, const bn_t *m) { return bn_mod_mul(r, a, a, m); }

/*
 * The *_quick variants assume 0 <= a, b < m and m > 0, and avoid division.
 * This is the form used by the field arithmetic.
 */
BN_HD void bn_mod_add_quick(bn_t *r, const bn_t *a, const bn_t *b, const bn_t *m)
{
    bn_uadd(r, a, b);
    if (bn_ucmp(r, m) >= 0)
        bn_usub(r, r, m);
}

BN_HD void bn_mod_sub_quick(bn_t *r, const bn_t *a, const bn_t *b, const bn_t *m)
{
    if (bn_ucmp(a, b) >= 0) {
        bn_usub(r, a, b);
    } else {
        bn_usub(r, b, a);
        bn_usub(r, m, r);
    }
}

BN_HD void bn_mod_lshift1_quick(bn_t *r, const bn_t *a, const bn_t *m)
{
    bn_lshift1(r, a);
    if (bn_ucmp(r, m) >= 0)
        bn_usub(r, r, m);
}

BN_HD void bn_mod_lshift_quick(bn_t *r, const bn_t *a, int n, const bn_t *m)
{
    bn_copy(r, a);
    for (int i = 0; i < n; i++)
        bn_mod_lshift1_quick(r, r, m);
}

/*
 * r = a^-1 mod n with 0 <= r < n, by the extended Euclidean algorithm.
 * Returns 0 if n <= 1 or the inverse does not exist (r is then zero).
 */
BN_HD int bn_mod_inverse(bn_t *r, const bn_t *a, const bn_t *n)
{
    if (n->neg || n->top == 0 || bn_is_one(n))
        return 0;

    bn_t r0 = *n, r1, t0, t1, q, tmp;
    if (!bn_nnmod(&r1, a, n))
        return 0;
    bn_zero(&t0);
    bn_set_word(&t1, 1);

    while (!bn_is_zero(&r1)) {
        bn_div(&q, &tmp, &r0, &r1);
        r0 = r1;
        r1 = tmp;
        if (!bn_mul(&tmp, &q, &t1) || !bn_sub(&tmp, &t0, &tmp))
            return 0;
        t0 = t1;
        t1 = tmp;
    }
    if (!bn_is_one(&r0)) {
        bn_zero(r);
        return 0;
    }
    return bn_nnmod(r, &t0, n);
}

/* Swap a and b if the low bit of condition is set, without branching. */
BN_HD void bn_consttime_swap(bn_word condition, bn_t *a, bn_t *b)
{
    bn_word mask = (bn_word)0 - (condition & 1);
    for (int i = 0; i < BN_WORDS; i++) {
        bn_word t = mask & (a->d[i] ^ b->d[i]);
        a->d[i] ^= t;
        b->d[i] ^= t;
    }
    int imask = -(int)(condition & 1);
    int t = imask & (a->top ^ b->top);
    a->top ^= t;
    b->top ^= t;
    t = imask & (a->neg ^ b->neg);
    a->neg ^= t;
    b->neg ^= t;
}

/* ---- conversion -------------------------------------------------------- */

/*
 * Parse a hex string (optional leading '-', no 0x prefix) into r.
 * Returns 0 on an invalid character or a value that does not fit.
 */
BN_HD int bn_from_hex(bn_t *r, const char *hex)
{
    bn_t t;
    bn_zero(&t);
    int neg = 0;
    if (*hex == '-') {
        neg = 1;
        hex++;
    }
    int len = 0;
    while (hex[len] != '\0')
        len++;
    if (len == 0)
        return 0;
    int bit = 0;
    for (int i = len - 1; i >= 0; i--) {
        char c = hex[i];
        bn_word v;
        if (c >= '0' && c <= '9')
            v = (bn_word)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v = (bn_word)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v = (bn_word)(c - 'A' + 10);
        else
            return 0;
        if (v != 0 && bit >= BN_MAX_BITS)
            return 0;
        if (bit < BN_MAX_BITS)
            t.d[bit / BN_WORD_BITS] |= v << (bit % BN_WORD_BITS);
        bit += 4;
    }
    t.top = BN_WORDS;
    bn_normalize(&t);
    t.neg = neg && t.top;
    *r = t;
    return 1;
}

/* Write a as lowercase hex with no leading zeros ("0" for zero).
 * out must hold at least BN_WORDS * 16 + 2 bytes. */
BN_HD void bn_to_hex(const bn_t *a, char *out)
{
    static const char digits[] = "0123456789abcdef";
    int pos = 0;
    if (a->top == 0) {
        out[pos++] = '0';
        out[pos] = '\0';
        return;
    }
    if (a->neg)
        out[pos++] = '-';
    int started = 0;
    for (int i = a->top - 1; i >= 0; i--) {
        for (int s = BN_WORD_BITS - 4; s >= 0; s -= 4) {
            int v = (int)((a->d[i] >> s) & 0xf);
            if (v == 0 && !started)
                continue;
            started = 1;
            out[pos++] = digits[v];
        }
    }
    out[pos] = '\0';
}

/* Parse len big-endian bytes as an unsigned value. Returns 0 if it does not fit. */
BN_HD int bn_from_bytes_be(bn_t *r, const uint8_t *in, int len)
{
    if (len < 0 || len > BN_WORDS * 8)
        return 0;
    bn_t t;
    bn_zero(&t);
    for (int i = 0; i < len; i++) {
        int byte = len - 1 - i; /* position counted from the least significant end */
        t.d[byte / 8] |= (bn_word)in[i] << (8 * (byte % 8));
    }
    t.top = BN_WORDS;
    bn_normalize(&t);
    *r = t;
    return 1;
}

/* Write |a| as len big-endian bytes, zero padded. Returns 0 if it does not fit. */
BN_HD int bn_to_bytes_be(const bn_t *a, uint8_t *out, int len)
{
    if (len < 0 || bn_num_bits(a) > 8 * len)
        return 0;
    for (int i = 0; i < len; i++) {
        int byte = len - 1 - i;
        out[i] = (byte / 8 < BN_WORDS) ? (uint8_t)(a->d[byte / 8] >> (8 * (byte % 8))) : 0;
    }
    return 1;
}

BN_HD void bn_print(const char *label, const bn_t *a)
{
    char buf[BN_WORDS * 16 + 2];
    bn_to_hex(a, buf);
    printf("%s%s\n", label, buf);
}

#endif /* CUDA_SECP256K1_BN_CUH */
