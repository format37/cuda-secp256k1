/*
 * ec.cuh - elliptic curve arithmetic over a prime field, for secp256k1.
 *
 * Points are kept in Jacobian coordinates (X, Y, Z) with every coordinate in
 * Montgomery form, like OpenSSL's EC_GFp_mont_method. The point at infinity
 * has Z == 0. ec_affine_t is the external representation: plain (x, y) in
 * normal form, as found in keys and signatures.
 *
 * The formulas are ported from OpenSSL (crypto/ec/ecp_smpl.c and
 * crypto/ec/ec_mult.c):
 *   - ec_point_dbl / ec_point_add: ossl_ec_GFp_simple_dbl / _add
 *   - ec_point_mul: the Montgomery ladder (ossl_ec_scalar_mul_ladder with
 *     ossl_ec_GFp_simple_ladder_pre / _step / _post)
 *
 * The ladder does the same sequence of field operations for every scalar,
 * but there is no coordinate blinding and no other side-channel hardening:
 * this library is meant for bulk computation, not for protecting secrets
 * on shared hardware.
 */
#ifndef CUDA_SECP256K1_EC_CUH
#define CUDA_SECP256K1_EC_CUH

#include "mont.cuh"

typedef struct {
    bn_t x, y;    /* normal (non-Montgomery) form */
    int infinity;
} ec_affine_t;

typedef struct {
    bn_t X, Y, Z; /* Jacobian coordinates, Montgomery form; Z == 0 is infinity */
    int z_is_one; /* Z is known to be the Montgomery form of 1 */
} ec_point_t;

typedef struct {
    bn_mont_t field; /* Montgomery context for the field prime p */
    bn_t a, b;       /* curve coefficients, Montgomery form */
    int a_is_zero;   /* lets the formulas skip the a*Z^4 terms (true for secp256k1) */
    bn_t order;      /* order of the generator; the cofactor must be 1 */
    ec_affine_t g;   /* generator */
} ec_group_t;

/* ---- field element helpers (Montgomery form, values in [0, p)) --------- */

BN_HD void fe_mul(const ec_group_t *g, bn_t *r, const bn_t *a, const bn_t *b)
{
    bn_mont_mul(r, a, b, &g->field);
}

BN_HD void fe_sqr(const ec_group_t *g, bn_t *r, const bn_t *a)
{
    bn_mont_mul(r, a, a, &g->field);
}

BN_HD void fe_add(const ec_group_t *g, bn_t *r, const bn_t *a, const bn_t *b)
{
    bn_mod_add_quick(r, a, b, &g->field.n);
}

BN_HD void fe_sub(const ec_group_t *g, bn_t *r, const bn_t *a, const bn_t *b)
{
    bn_mod_sub_quick(r, a, b, &g->field.n);
}

BN_HD void fe_dbl(const ec_group_t *g, bn_t *r, const bn_t *a)
{
    bn_mod_lshift1_quick(r, a, &g->field.n);
}

/* r = a * 2^n */
BN_HD void fe_lshift(const ec_group_t *g, bn_t *r, const bn_t *a, int n)
{
    bn_mod_lshift_quick(r, a, n, &g->field.n);
}

/* r = -a */
BN_HD void fe_neg(const ec_group_t *g, bn_t *r, const bn_t *a)
{
    if (bn_is_zero(a))
        bn_zero(r);
    else
        bn_usub(r, &g->field.n, a);
}

/* r = a^-1 by Fermat's little theorem: a^(p-2). Gives 0 for a == 0. */
BN_HD void fe_inv(const ec_group_t *g, bn_t *r, const bn_t *a)
{
    bn_t e, two;
    bn_set_word(&two, 2);
    bn_usub(&e, &g->field.n, &two);
    bn_mont_exp(r, a, &e, &g->field);
}

/* Convert into / out of Montgomery form. */
BN_HD void fe_encode(const ec_group_t *g, bn_t *r, const bn_t *a)
{
    bn_to_mont(r, a, &g->field);
}

BN_HD void fe_decode(const ec_group_t *g, bn_t *r, const bn_t *a)
{
    bn_from_mont(r, a, &g->field);
}

/* ---- group setup ------------------------------------------------------- */

/*
 * Initialize a group for the curve y^2 = x^3 + a*x + b over GF(p) with
 * generator (gx, gy) of the given prime order. The cofactor must be 1.
 * Returns 0 if p is not usable as a Montgomery modulus.
 */
BN_HD int ec_group_init(ec_group_t *g, const bn_t *p, const bn_t *a, const bn_t *b,
                        const bn_t *gx, const bn_t *gy, const bn_t *order)
{
    if (!bn_mont_init(&g->field, p))
        return 0;
    fe_encode(g, &g->a, a);
    fe_encode(g, &g->b, b);
    g->a_is_zero = bn_is_zero(a);
    g->order = *order;
    g->g.x = *gx;
    g->g.y = *gy;
    g->g.infinity = 0;
    return 1;
}

/* secp256k1 parameters as published in SEC 2, section 2.4.1. */
BN_HD int ec_group_init_secp256k1(ec_group_t *g)
{
    bn_t p, a, b, gx, gy, n;
    bn_from_hex(&p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    bn_zero(&a);
    bn_set_word(&b, 7);
    bn_from_hex(&gx, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    bn_from_hex(&gy, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
    bn_from_hex(&n, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    return ec_group_init(g, &p, &a, &b, &gx, &gy, &n);
}

/* ---- point representation ---------------------------------------------- */

BN_HD void ec_point_set_infinity(ec_point_t *p)
{
    bn_zero(&p->X);
    bn_zero(&p->Y);
    bn_zero(&p->Z);
    p->z_is_one = 0;
}

BN_HD int ec_point_is_infinity(const ec_point_t *p)
{
    return bn_is_zero(&p->Z);
}

BN_HD void ec_point_from_affine(const ec_group_t *g, ec_point_t *p, const ec_affine_t *a)
{
    if (a->infinity) {
        ec_point_set_infinity(p);
        return;
    }
    fe_encode(g, &p->X, &a->x);
    fe_encode(g, &p->Y, &a->y);
    p->Z = g->field.one;
    p->z_is_one = 1;
}

/* Convert to affine coordinates; a->infinity is set for the point at infinity. */
BN_HD void ec_point_to_affine(const ec_group_t *g, ec_affine_t *a, const ec_point_t *p)
{
    if (ec_point_is_infinity(p)) {
        bn_zero(&a->x);
        bn_zero(&a->y);
        a->infinity = 1;
        return;
    }
    a->infinity = 0;
    if (p->z_is_one) {
        fe_decode(g, &a->x, &p->X);
        fe_decode(g, &a->y, &p->Y);
        return;
    }
    /* (x, y) = (X / Z^2, Y / Z^3) */
    bn_t zi, zi2, zi3, t;
    fe_inv(g, &zi, &p->Z);
    fe_sqr(g, &zi2, &zi);
    fe_mul(g, &zi3, &zi2, &zi);
    fe_mul(g, &t, &p->X, &zi2);
    fe_decode(g, &a->x, &t);
    fe_mul(g, &t, &p->Y, &zi3);
    fe_decode(g, &a->y, &t);
}

/* p = -p */
BN_HD void ec_point_invert(const ec_group_t *g, ec_point_t *p)
{
    if (!ec_point_is_infinity(p))
        fe_neg(g, &p->Y, &p->Y);
}

/* Check y^2 == x^3 + a*x + b for an affine point. */
BN_HD int ec_affine_is_on_curve(const ec_group_t *g, const ec_affine_t *a)
{
    if (a->infinity)
        return 1;
    if (a->x.neg || a->y.neg || bn_ucmp(&a->x, &g->field.n) >= 0 || bn_ucmp(&a->y, &g->field.n) >= 0)
        return 0;
    bn_t x, y, lhs, rhs;
    fe_encode(g, &x, &a->x);
    fe_encode(g, &y, &a->y);
    fe_sqr(g, &lhs, &y);
    fe_sqr(g, &rhs, &x);
    if (!g->a_is_zero)
        fe_add(g, &rhs, &rhs, &g->a);
    fe_mul(g, &rhs, &rhs, &x);
    fe_add(g, &rhs, &rhs, &g->b);
    return bn_ucmp(&lhs, &rhs) == 0;
}

/*
 * Serialize an affine point as SEC 1 octets: 33 bytes (02/03 prefix) when
 * compressed, 65 bytes (04 prefix) otherwise. Returns the length written,
 * or 0 for the point at infinity.
 */
BN_HD int ec_affine_to_bytes(const ec_affine_t *a, uint8_t *out, int compressed)
{
    if (a->infinity)
        return 0;
    if (compressed) {
        out[0] = bn_is_odd(&a->y) ? 0x03 : 0x02;
        bn_to_bytes_be(&a->x, out + 1, 32);
        return 33;
    }
    out[0] = 0x04;
    bn_to_bytes_be(&a->x, out + 1, 32);
    bn_to_bytes_be(&a->y, out + 33, 32);
    return 65;
}

/* ---- doubling and addition (ossl_ec_GFp_simple_dbl / _add) ------------- */

BN_HD void ec_point_dbl(const ec_group_t *g, ec_point_t *r, const ec_point_t *a)
{
    if (ec_point_is_infinity(a)) {
        ec_point_set_infinity(r);
        return;
    }
    ec_point_t res;
    bn_t n0, n1, n2, n3;

    /* n1 = 3 * X^2 + a * Z^4 */
    fe_sqr(g, &n0, &a->X);
    fe_dbl(g, &n1, &n0);
    fe_add(g, &n0, &n0, &n1);
    if (g->a_is_zero) {
        n1 = n0;
    } else if (a->z_is_one) {
        fe_add(g, &n1, &n0, &g->a);
    } else {
        fe_sqr(g, &n1, &a->Z);
        fe_sqr(g, &n1, &n1);
        fe_mul(g, &n1, &n1, &g->a);
        fe_add(g, &n1, &n1, &n0);
    }

    /* Z_r = 2 * Y * Z */
    if (a->z_is_one)
        n0 = a->Y;
    else
        fe_mul(g, &n0, &a->Y, &a->Z);
    fe_dbl(g, &res.Z, &n0);
    res.z_is_one = 0;

    /* n2 = 4 * X * Y^2 */
    fe_sqr(g, &n3, &a->Y);
    fe_mul(g, &n2, &a->X, &n3);
    fe_lshift(g, &n2, &n2, 2);

    /* X_r = n1^2 - 2 * n2 */
    fe_dbl(g, &n0, &n2);
    fe_sqr(g, &res.X, &n1);
    fe_sub(g, &res.X, &res.X, &n0);

    /* n3 = 8 * Y^4 */
    fe_sqr(g, &n0, &n3);
    fe_lshift(g, &n3, &n0, 3);

    /* Y_r = n1 * (n2 - X_r) - n3 */
    fe_sub(g, &n0, &n2, &res.X);
    fe_mul(g, &n0, &n1, &n0);
    fe_sub(g, &res.Y, &n0, &n3);

    *r = res;
}

BN_HD void ec_point_add(const ec_group_t *g, ec_point_t *r, const ec_point_t *a, const ec_point_t *b)
{
    if (a == b) {
        ec_point_dbl(g, r, a);
        return;
    }
    if (ec_point_is_infinity(a)) {
        *r = *b;
        return;
    }
    if (ec_point_is_infinity(b)) {
        *r = *a;
        return;
    }
    ec_point_t res;
    bn_t n0, n1, n2, n3, n4, n5, n6;

    /* n1 = X_a * Z_b^2, n2 = Y_a * Z_b^3 */
    if (b->z_is_one) {
        n1 = a->X;
        n2 = a->Y;
    } else {
        fe_sqr(g, &n0, &b->Z);
        fe_mul(g, &n1, &a->X, &n0);
        fe_mul(g, &n0, &n0, &b->Z);
        fe_mul(g, &n2, &a->Y, &n0);
    }

    /* n3 = X_b * Z_a^2, n4 = Y_b * Z_a^3 */
    if (a->z_is_one) {
        n3 = b->X;
        n4 = b->Y;
    } else {
        fe_sqr(g, &n0, &a->Z);
        fe_mul(g, &n3, &b->X, &n0);
        fe_mul(g, &n0, &n0, &a->Z);
        fe_mul(g, &n4, &b->Y, &n0);
    }

    /* n5 = n1 - n3, n6 = n2 - n4 */
    fe_sub(g, &n5, &n1, &n3);
    fe_sub(g, &n6, &n2, &n4);

    if (bn_is_zero(&n5)) {
        if (bn_is_zero(&n6))
            ec_point_dbl(g, r, a); /* same point */
        else
            ec_point_set_infinity(r); /* a == -b */
        return;
    }

    /* n7 = n1 + n3, n8 = n2 + n4 (reusing n1, n2) */
    fe_add(g, &n1, &n1, &n3);
    fe_add(g, &n2, &n2, &n4);

    /* Z_r = Z_a * Z_b * n5 */
    if (a->z_is_one && b->z_is_one) {
        res.Z = n5;
    } else {
        if (a->z_is_one)
            n0 = b->Z;
        else if (b->z_is_one)
            n0 = a->Z;
        else
            fe_mul(g, &n0, &a->Z, &b->Z);
        fe_mul(g, &res.Z, &n0, &n5);
    }
    res.z_is_one = 0;

    /* X_r = n6^2 - n5^2 * n7 */
    fe_sqr(g, &n0, &n6);
    fe_sqr(g, &n4, &n5);
    fe_mul(g, &n3, &n1, &n4);
    fe_sub(g, &res.X, &n0, &n3);

    /* n9 = n5^2 * n7 - 2 * X_r */
    fe_dbl(g, &n0, &res.X);
    fe_sub(g, &n0, &n3, &n0);

    /* Y_r = (n6 * n9 - n8 * n5^3) / 2 */
    fe_mul(g, &n0, &n0, &n6);
    fe_mul(g, &n5, &n4, &n5);
    fe_mul(g, &n1, &n2, &n5);
    fe_sub(g, &n0, &n0, &n1);
    if (bn_is_odd(&n0))
        bn_uadd(&n0, &n0, &g->field.n); /* now even and still below 2p */
    bn_rshift1(&res.Y, &n0);

    *r = res;
}

/* ---- scalar multiplication --------------------------------------------- */

BN_HD void ec_point_cswap(bn_word condition, ec_point_t *a, ec_point_t *b)
{
    bn_consttime_swap(condition, &a->X, &b->X);
    bn_consttime_swap(condition, &a->Y, &b->Y);
    bn_consttime_swap(condition, &a->Z, &b->Z);
    int t = (a->z_is_one ^ b->z_is_one) & (int)(condition & 1);
    a->z_is_one ^= t;
    b->z_is_one ^= t;
}

/*
 * Ladder setup: s = p, r = 2p in homogeneous (X : Z) coordinates, with p
 * affine. Doubling formula from Izu-Takagi (EFD dbl-2002-it-2) with Z1 = 1.
 */
BN_HD void ec_ladder_pre(const ec_group_t *g, ec_point_t *r, ec_point_t *s, const ec_point_t *p)
{
    bn_t t1, t2, t3, t4, t5;

    fe_sqr(g, &t3, &p->X);                 /* t3 = X^2 */
    if (g->a_is_zero)
        t4 = t3;
    else
        fe_sub(g, &t4, &t3, &g->a);        /* t4 = X^2 - a */
    fe_sqr(g, &t4, &t4);                   /* t4 = (X^2 - a)^2 */
    fe_mul(g, &t5, &p->X, &g->b);
    fe_lshift(g, &t5, &t5, 3);             /* t5 = 8 * b * X */
    fe_sub(g, &r->X, &t4, &t5);            /* r.X = (X^2 - a)^2 - 8 * b * X */
    if (g->a_is_zero)
        t1 = t3;
    else
        fe_add(g, &t1, &t3, &g->a);        /* t1 = X^2 + a */
    fe_mul(g, &t2, &p->X, &t1);
    fe_add(g, &t2, &g->b, &t2);            /* t2 = X^3 + a * X + b */
    fe_lshift(g, &r->Z, &t2, 2);           /* r.Z = 4 * (X^3 + a * X + b) */
    bn_zero(&r->Y);
    r->z_is_one = 0;

    s->X = p->X;
    bn_zero(&s->Y);
    s->Z = g->field.one;
    s->z_is_one = 0;
}

/*
 * One ladder step: s = r + s, r = 2r, with p affine and r, s in homogeneous
 * (X : Z) coordinates. Izu-Takagi differential addition-and-doubling
 * (EFD ladd-2002-it-4), in the operation order of OpenSSL.
 */
BN_HD void ec_ladder_step(const ec_group_t *g, ec_point_t *r, ec_point_t *s, const ec_point_t *p)
{
    bn_t t0, t1, t2, t3, t4, t5, t6;

    fe_mul(g, &t6, &r->X, &s->X);
    fe_mul(g, &t0, &r->Z, &s->Z);
    fe_mul(g, &t4, &r->X, &s->Z);
    fe_mul(g, &t3, &r->Z, &s->X);
    if (g->a_is_zero)
        bn_zero(&t5);
    else
        fe_mul(g, &t5, &g->a, &t0);
    fe_add(g, &t5, &t6, &t5);
    fe_add(g, &t6, &t3, &t4);
    fe_mul(g, &t5, &t6, &t5);
    fe_sqr(g, &t0, &t0);
    fe_lshift(g, &t2, &g->b, 2);
    fe_mul(g, &t0, &t2, &t0);
    fe_dbl(g, &t5, &t5);
    fe_sub(g, &t3, &t4, &t3);
    /* s.Z */
    fe_sqr(g, &s->Z, &t3);
    fe_mul(g, &t4, &s->Z, &p->X);
    fe_add(g, &t0, &t0, &t5);
    /* s.X */
    fe_sub(g, &s->X, &t0, &t4);
    fe_sqr(g, &t4, &r->X);
    fe_sqr(g, &t5, &r->Z);
    if (g->a_is_zero)
        bn_zero(&t6);
    else
        fe_mul(g, &t6, &t5, &g->a);
    fe_add(g, &t1, &r->X, &r->Z);
    fe_sqr(g, &t1, &t1);
    fe_sub(g, &t1, &t1, &t4);
    fe_sub(g, &t1, &t1, &t5);
    fe_sub(g, &t3, &t4, &t6);
    fe_sqr(g, &t3, &t3);
    fe_mul(g, &t0, &t5, &t1);
    fe_mul(g, &t0, &t2, &t0);
    /* r.X */
    fe_sub(g, &r->X, &t3, &t0);
    fe_add(g, &t3, &t4, &t6);
    fe_sqr(g, &t4, &t5);
    fe_mul(g, &t4, &t4, &t2);
    fe_mul(g, &t1, &t1, &t3);
    fe_dbl(g, &t1, &t1);
    /* r.Z */
    fe_add(g, &r->Z, &t4, &t1);
}

/*
 * Ladder finish: recover the y coordinate of r (Brier-Joye, eq. 8, in mixed
 * coordinates) and return r as an affine point with Z = 1.
 */
BN_HD void ec_ladder_post(const ec_group_t *g, ec_point_t *r, ec_point_t *s, const ec_point_t *p)
{
    if (bn_is_zero(&r->Z)) {
        ec_point_set_infinity(r);
        return;
    }
    if (bn_is_zero(&s->Z)) {
        *r = *p;
        ec_point_invert(g, r);
        return;
    }
    bn_t t0, t1, t2, t3, t4, t5, t6;

    fe_dbl(g, &t4, &p->Y);
    fe_mul(g, &t6, &r->X, &t4);
    fe_mul(g, &t6, &s->Z, &t6);
    fe_mul(g, &t5, &r->Z, &t6);
    fe_dbl(g, &t1, &g->b);
    fe_mul(g, &t1, &s->Z, &t1);
    fe_sqr(g, &t3, &r->Z);
    fe_mul(g, &t2, &t3, &t1);
    if (g->a_is_zero)
        bn_zero(&t6);
    else
        fe_mul(g, &t6, &r->Z, &g->a);
    fe_mul(g, &t1, &p->X, &r->X);
    fe_add(g, &t1, &t1, &t6);
    fe_mul(g, &t1, &s->Z, &t1);
    fe_mul(g, &t0, &p->X, &r->Z);
    fe_add(g, &t6, &r->X, &t0);
    fe_mul(g, &t6, &t6, &t1);
    fe_add(g, &t6, &t6, &t2);
    fe_sub(g, &t0, &t0, &r->X);
    fe_sqr(g, &t0, &t0);
    fe_mul(g, &t0, &t0, &s->X);
    fe_sub(g, &t0, &t6, &t0);
    fe_mul(g, &t1, &s->Z, &t4);
    fe_mul(g, &t1, &t3, &t1);
    fe_inv(g, &t1, &t1);
    fe_mul(g, &r->X, &t5, &t1);
    fe_mul(g, &r->Y, &t0, &t1);
    r->Z = g->field.one;
    r->z_is_one = 1;
}

/*
 * r = k * p by the Montgomery ladder, following ossl_ec_scalar_mul_ladder.
 * The result has Z = 1 (or is the point at infinity).
 */
BN_HD void ec_point_mul_ladder(const ec_group_t *g, ec_point_t *r, const bn_t *scalar, const ec_affine_t *p)
{
    if (p->infinity) {
        ec_point_set_infinity(r);
        return;
    }
    ec_point_t pp, s;
    ec_point_from_affine(g, &pp, p);

    /*
     * Pad the scalar to a fixed bit length: use k + order or k + 2*order,
     * whichever has the bit just above the order's top bit set. Both give
     * the same multiple of p since order * p is the point at infinity.
     */
    int cardinality_bits = bn_num_bits(&g->order);
    bn_t k = *scalar, lambda;
    if (k.neg || bn_num_bits(&k) > cardinality_bits)
        bn_nnmod(&k, &k, &g->order);
    bn_add(&lambda, &k, &g->order);
    bn_add(&k, &lambda, &g->order);
    int kbit = bn_is_bit_set(&lambda, cardinality_bits);
    bn_consttime_swap((bn_word)kbit, &k, &lambda);

    ec_ladder_pre(g, r, &s, &pp);

    /* The top bit is a 1 in a fixed position; the swaps of consecutive
     * iterations are merged by tracking the previous bit in pbit. */
    int pbit = 1;
    for (int i = cardinality_bits - 1; i >= 0; i--) {
        kbit = bn_is_bit_set(&k, i) ^ pbit;
        ec_point_cswap((bn_word)kbit, r, &s);
        ec_ladder_step(g, r, &s, &pp);
        pbit ^= kbit;
    }
    ec_point_cswap((bn_word)pbit, r, &s);

    ec_ladder_post(g, r, &s, &pp);
}

/* r = k * p by plain double-and-add. Faster than the ladder, but the work
 * depends on the bits of k. */
BN_HD void ec_point_mul_vartime(const ec_group_t *g, ec_point_t *r, const bn_t *scalar, const ec_affine_t *p)
{
    bn_t k = *scalar;
    if (k.neg || bn_ucmp(&k, &g->order) >= 0)
        bn_nnmod(&k, &k, &g->order);
    ec_point_t pp, acc;
    ec_point_from_affine(g, &pp, p);
    ec_point_set_infinity(&acc);
    for (int i = bn_num_bits(&k) - 1; i >= 0; i--) {
        ec_point_dbl(g, &acc, &acc);
        if (bn_is_bit_set(&k, i))
            ec_point_add(g, &acc, &acc, &pp);
    }
    *r = acc;
}

/* r = k * p in affine coordinates, using the ladder. */
BN_HD void ec_point_mul(const ec_group_t *g, ec_affine_t *r, const bn_t *scalar, const ec_affine_t *p)
{
    ec_point_t t;
    ec_point_mul_ladder(g, &t, scalar, p);
    ec_point_to_affine(g, r, &t);
}

/*
 * Public key for a private scalar: pub = priv * G.
 * Returns 0 if priv is not in [1, order - 1].
 */
BN_HD int ec_pubkey_create(const ec_group_t *g, ec_affine_t *pub, const bn_t *priv)
{
    if (priv->neg || bn_is_zero(priv) || bn_ucmp(priv, &g->order) >= 0)
        return 0;
    ec_point_mul(g, pub, priv, &g->g);
    return 1;
}

#endif /* CUDA_SECP256K1_EC_CUH */
