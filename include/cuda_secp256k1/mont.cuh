/*
 * mont.cuh - Montgomery modular arithmetic on bn_t.
 *
 * A bn_mont_t describes an odd modulus n of nl words and R = 2^(64*nl).
 * The Montgomery form of x is x*R mod n. Multiplying two values in
 * Montgomery form with bn_mont_mul gives the product in Montgomery form,
 * which is what makes the field arithmetic in ec.cuh fast: no division.
 *
 * The reduction is the word-by-word algorithm of OpenSSL's
 * bn_from_montgomery_word (crypto/bn/bn_mont.c), so results match
 * BN_mod_mul_montgomery / BN_to_montgomery / BN_from_montgomery exactly.
 */
#ifndef CUDA_SECP256K1_MONT_CUH
#define CUDA_SECP256K1_MONT_CUH

#include "bn.cuh"

typedef struct {
    bn_t n;      /* the modulus; odd, positive, at most BN_WORDS/2 words */
    bn_t rr;     /* R^2 mod n, used to convert into Montgomery form */
    bn_t one;    /* R mod n, the Montgomery form of 1 */
    bn_word n0;  /* -n^-1 mod 2^64 */
    int nl;      /* number of words in n, so R = 2^(64*nl) */
} bn_mont_t;

/* Set up a context for the modulus n. Returns 0 if n is not a positive odd
 * number of at most BN_WORDS/2 words. */
BN_HD int bn_mont_init(bn_mont_t *m, const bn_t *n)
{
    if (n->top == 0 || n->neg || !(n->d[0] & 1) || n->top > BN_WORDS / 2)
        return 0;
    m->n = *n;
    m->nl = n->top;

    /*
     * n0 = -n^-1 mod 2^64 by Newton's iteration. For odd n, x = n satisfies
     * n*x == 1 (mod 8); each step doubles the number of correct bits, so five
     * steps give 96 > 64 bits.
     */
    bn_word x = n->d[0];
    for (int i = 0; i < 5; i++)
        x *= 2 - n->d[0] * x;
    m->n0 = (bn_word)0 - x;

    /* one = R mod n and rr = R^2 mod n, by repeated doubling of 1 mod n. */
    int bits = BN_WORD_BITS * m->nl;
    bn_t t;
    bn_set_word(&t, 1);
    bn_nnmod(&t, &t, n);
    for (int i = 0; i < bits; i++)
        bn_mod_lshift1_quick(&t, &t, n);
    m->one = t;
    for (int i = 0; i < bits; i++)
        bn_mod_lshift1_quick(&t, &t, n);
    m->rr = t;
    return 1;
}

/*
 * r = t * R^-1 mod n for 0 <= t < n*R (so t has at most 2*nl words).
 * Returns 0 if t is negative or too large.
 */
BN_HD int bn_mont_reduce(bn_t *r, const bn_t *t, const bn_mont_t *m)
{
    int nl = m->nl;
    if (t->neg || t->top > 2 * nl)
        return 0;

    bn_word w[BN_WORDS];
    for (int i = 0; i < BN_WORDS; i++)
        w[i] = t->d[i];

    /* Add multiples of n until the low nl words are zero; then divide by R
     * by dropping those words. The value stays below 2n, so one conditional
     * subtraction finishes the reduction. */
    bn_word carry = 0;
    for (int i = 0; i < nl; i++) {
        bn_word u = w[i] * m->n0;
        bn_word c = bn_mul_add_words(&w[i], m->n.d, nl, u);
        bn_dword s = (bn_dword)w[i + nl] + c + carry;
        w[i + nl] = (bn_word)s;
        carry = (bn_word)(s >> BN_WORD_BITS);
    }

    bn_t res;
    bn_zero(&res);
    bn_word borrow = bn_sub_words(res.d, &w[nl], m->n.d, nl);
    /* Keep the subtracted value if the result overflowed R or is >= n. */
    bn_word mask = (bn_word)0 - (carry | (borrow ^ 1));
    for (int i = 0; i < nl; i++)
        res.d[i] = (mask & res.d[i]) | (~mask & w[nl + i]);
    res.top = nl;
    bn_normalize(&res);
    *r = res;
    return 1;
}

/* r = a * b * R^-1 mod n for non-negative a, b with a*b < n*R. */
BN_HD int bn_mont_mul(bn_t *r, const bn_t *a, const bn_t *b, const bn_mont_t *m)
{
    bn_t t;
    if (a->neg || b->neg || !bn_mul(&t, a, b))
        return 0;
    return bn_mont_reduce(r, &t, m);
}

BN_HD int bn_mont_sqr(bn_t *r, const bn_t *a, const bn_mont_t *m)
{
    return bn_mont_mul(r, a, a, m);
}

/* r = a * R mod n (convert into Montgomery form). a may be any value. */
BN_HD int bn_to_mont(bn_t *r, const bn_t *a, const bn_mont_t *m)
{
    bn_t t;
    if (a->neg || bn_ucmp(a, &m->n) >= 0) {
        if (!bn_nnmod(&t, a, &m->n))
            return 0;
    } else {
        t = *a;
    }
    return bn_mont_mul(r, &t, &m->rr, m);
}

/* r = a * R^-1 mod n (convert out of Montgomery form); requires 0 <= a < R. */
BN_HD int bn_from_mont(bn_t *r, const bn_t *a, const bn_mont_t *m)
{
    if (a->neg || a->top > m->nl)
        return 0;
    return bn_mont_reduce(r, a, m);
}

/*
 * r = a^e in Montgomery form, for a already in Montgomery form and e >= 0.
 * Plain left-to-right square-and-multiply; e is treated as public.
 */
BN_HD int bn_mont_exp(bn_t *r, const bn_t *a, const bn_t *e, const bn_mont_t *m)
{
    if (e->neg)
        return 0;
    bn_t base = *a;
    bn_t acc = m->one;
    for (int i = bn_num_bits(e) - 1; i >= 0; i--) {
        bn_mont_sqr(&acc, &acc, m);
        if (bn_is_bit_set(e, i))
            bn_mont_mul(&acc, &acc, &base, m);
    }
    *r = acc;
    return 1;
}

/* r = a^e mod n for e >= 0, with a and r in normal form (same as BN_mod_exp). */
BN_HD int bn_mod_exp_mont(bn_t *r, const bn_t *a, const bn_t *e, const bn_mont_t *m)
{
    bn_t t;
    return bn_to_mont(&t, a, m) && bn_mont_exp(&t, &t, e, m) && bn_from_mont(r, &t, m);
}

#endif /* CUDA_SECP256K1_MONT_CUH */
