/* test_bn.cu - big-number arithmetic checked against OpenSSL BN_*. */
#include "test_util.h"

struct BnIn {
    bn_t a, b;
    int shift;
};

struct BnOut {
    bn_t add, sub, mul, q, r, nnmod, lsh, rsh, lsh1, rsh1;
    bn_t inv_p, inv_b;
    bn_t qadd, qsub, qlsh1, qlsh;
    int add_ok, sub_ok, mul_ok, div_ok, lsh_ok, inv_p_ok, inv_b_ok;
    int cmp, ucmp, bits, hex_ok, bytes_ok;
};

__host__ __device__ static void compute(const BnIn *inp, BnOut *out, const bn_t &p)
{
    const bn_t a = inp->a, b = inp->b;
    BnOut o;
    memset(&o, 0, sizeof(o));

    o.add_ok = bn_add(&o.add, &a, &b);
    o.sub_ok = bn_sub(&o.sub, &a, &b);
    o.mul_ok = bn_mul(&o.mul, &a, &b);
    o.div_ok = bn_div(&o.q, &o.r, &a, &b);
    if (o.div_ok)
        bn_nnmod(&o.nnmod, &a, &b);
    o.lsh_ok = bn_lshift(&o.lsh, &a, inp->shift);
    bn_rshift(&o.rsh, &a, inp->shift);
    bn_lshift1(&o.lsh1, &a);
    bn_rshift1(&o.rsh1, &a);
    o.inv_p_ok = bn_mod_inverse(&o.inv_p, &a, &p);
    o.inv_b_ok = bn_mod_inverse(&o.inv_b, &a, &b);
    o.cmp = bn_cmp(&a, &b);
    o.ucmp = bn_ucmp(&a, &b);
    o.bits = bn_num_bits(&a);

    /* *_quick helpers on inputs reduced mod p */
    bn_t ap, bp;
    bn_nnmod(&ap, &a, &p);
    bn_nnmod(&bp, &b, &p);
    bn_mod_add_quick(&o.qadd, &ap, &bp, &p);
    bn_mod_sub_quick(&o.qsub, &ap, &bp, &p);
    bn_mod_lshift1_quick(&o.qlsh1, &ap, &p);
    bn_mod_lshift_quick(&o.qlsh, &ap, inp->shift % 8, &p);

    /* conversions must round-trip */
    char hexbuf[BN_WORDS * 16 + 2];
    bn_t back;
    bn_to_hex(&a, hexbuf);
    o.hex_ok = bn_from_hex(&back, hexbuf) && bn_cmp(&back, &a) == 0;
    uint8_t bytes[BN_WORDS * 8];
    bn_to_bytes_be(&a, bytes, sizeof(bytes));
    bn_from_bytes_be(&back, bytes, sizeof(bytes));
    o.bytes_ok = bn_ucmp(&back, &a) == 0;

    /* in-place forms must give the same answers */
    bn_t t = a;
    if (o.add_ok) {
        bn_add(&t, &t, &b);
        o.hex_ok &= bn_cmp(&t, &o.add) == 0;
    }
    if (o.sub_ok) {
        t = a;
        bn_sub(&t, &t, &b);
        o.hex_ok &= bn_cmp(&t, &o.sub) == 0;
        t = b;
        bn_sub(&t, &a, &t);
        o.hex_ok &= bn_cmp(&t, &o.sub) == 0;
    }
    if (o.mul_ok) {
        t = a;
        bn_mul(&t, &t, &b);
        o.hex_ok &= bn_cmp(&t, &o.mul) == 0;
    }
    if (o.div_ok) {
        t = a;
        bn_t t2 = b;
        bn_div(&t, &t2, &t, &t2);
        o.hex_ok &= bn_cmp(&t, &o.q) == 0 && bn_cmp(&t2, &o.r) == 0;
    }

    *out = o;
}

/*
 * Divisions whose quotient-word estimate in algorithm D takes a branch that
 * random inputs almost never reach. For each divisor length n, the top words
 * of the dividend and divisor make the first estimate either one too large,
 * so the divisor has to be added back, or equal to the word base 2^64, with
 * and without the remainder overflowing a word while it is corrected.
 */
static void add_division_corner_cases(std::vector<BnIn> &in)
{
    const bn_word H = (bn_word)1 << 63, M = ~(bn_word)0;
    struct Shape {
        bn_word a_top, a_next, b_top, b_next, b_low;
    } shapes[] = {
        {H - 1, H, H, 0, 1}, /* estimate one too large: add back */
        {H, H - 1, H, H, 0}, /* estimate equals 2^64 */
        {M, 1, M, M, 0},     /* estimate equals 2^64 and the remainder overflows */
    };
    for (int n = 2; n < BN_WORDS; n++)
        for (const Shape &sh : shapes)
            for (int signs = 0; signs < 4; signs++) {
                BnIn x;
                bn_zero(&x.a);
                bn_zero(&x.b);
                x.a.d[n] = sh.a_top;
                x.a.d[n - 1] = sh.a_next;
                x.a.top = n + 1;
                x.b.d[n - 1] = sh.b_top;
                x.b.d[n - 2] |= sh.b_next;
                x.b.d[0] |= sh.b_low;
                x.b.top = n;
                x.a.neg = signs & 1;
                x.b.neg = (signs >> 1) & 1;
                x.shift = 0;
                in.push_back(x);
            }
}

__global__ void bn_kernel(const BnIn *in, BnOut *out, int n, bn_t p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        compute(&in[i], &out[i], p);
}


int main()
{
    Ossl ossl;
    Stats st;
    bn_t p = from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f");

    /* Inputs: every pair of edge values with every sign combination, division
     * corner cases, then random pairs. */
    std::vector<BnIn> in;
    std::vector<bn_t> edges = edge_values();
    for (const bn_t &a : edges)
        for (const bn_t &b : edges)
            for (int signs = 0; signs < 4; signs++) {
                BnIn x;
                x.a = a;
                x.b = b;
                x.a.neg = (signs & 1) && a.top;
                x.b.neg = (signs & 2) && b.top;
                x.shift = (int)(rng()() % (BN_MAX_BITS + 8));
                in.push_back(x);
            }
    add_division_corner_cases(in);
    const int half = BN_MAX_BITS / 2;
    int nrand = batch_size();
    for (int i = 0; i < nrand; i++) {
        BnIn x;
        if (i % 4 == 0) {
            /* dividend that may use all the words */
            x.a = rand_upto(BN_MAX_BITS);
            x.b = rand_upto(half);
        } else if (i % 4 == 1) {
            /* divisor longer than half the width, dividend at least as long */
            int bbits = half + 1 + (int)(rng()() % half);
            x.b = rand_bits(bbits);
            x.a = rand_bits(bbits + (int)(rng()() % (BN_MAX_BITS - bbits + 1)));
        } else {
            x.a = rand_upto(half);
            x.b = rand_upto(half);
        }
        x.a.neg = (rng()() & 1) && x.a.top;
        x.b.neg = (rng()() & 1) && x.b.top;
        x.shift = (int)(rng()() % (BN_MAX_BITS + 8));
        in.push_back(x);
    }

    std::vector<BnOut> out = run_on_device<BnIn, BnOut>(in, [&](BnIn *d_in, BnOut *d_out, int n) {
        bn_kernel<<<(n + 127) / 128, 128>>>(d_in, d_out, n, p);
    }, [&](const BnIn *x, BnOut *o) { compute(x, o, p); });

    BIGNUM *a = BN_new(), *b = BN_new(), *P = BN_new(), *x = BN_new(), *y = BN_new(), *z = BN_new();
    to_ossl(P, &p);
    for (size_t i = 0; i < in.size(); i++) {
        const BnOut &o = out[i];
        to_ossl(a, &in[i].a);
        to_ossl(b, &in[i].b);
        std::string ha = hex(a), hb = hex(b);
        const char *A = ha.c_str(), *B = hb.c_str();

        BN_add(x, a, b);
        bool fits = BN_num_bits(x) <= BN_MAX_BITS;
        st.check(o.add_ok == (int)fits, "add_ok %s + %s: got %d", A, B, o.add_ok);
        if (o.add_ok)
            st.check(equals(&o.add, x), "add %s + %s: got %s want %s", A, B, hex(&o.add).c_str(), hex(x).c_str());
        BN_sub(x, a, b);
        fits = BN_num_bits(x) <= BN_MAX_BITS;
        st.check(o.sub_ok == (int)fits, "sub_ok %s - %s: got %d", A, B, o.sub_ok);
        if (o.sub_ok)
            st.check(equals(&o.sub, x), "sub %s - %s: got %s want %s", A, B, hex(&o.sub).c_str(), hex(x).c_str());

        BN_mul(x, a, b, ossl.ctx);
        fits = BN_num_bits(x) <= BN_MAX_BITS;
        st.check(o.mul_ok == (int)fits, "mul_ok %s * %s: got %d", A, B, o.mul_ok);
        if (o.mul_ok)
            st.check(equals(&o.mul, x), "mul %s * %s: got %s want %s", A, B, hex(&o.mul).c_str(), hex(x).c_str());

        st.check(o.div_ok == !BN_is_zero(b), "div_ok %s / %s: got %d", A, B, o.div_ok);
        if (o.div_ok) {
            BN_div(x, y, a, b, ossl.ctx);
            st.check(equals(&o.q, x), "div %s / %s: got q %s want %s", A, B, hex(&o.q).c_str(), hex(x).c_str());
            st.check(equals(&o.r, y), "div %s / %s: got r %s want %s", A, B, hex(&o.r).c_str(), hex(y).c_str());
            BN_nnmod(x, a, b, ossl.ctx);
            st.check(equals(&o.nnmod, x), "nnmod %s mod %s: got %s want %s", A, B, hex(&o.nnmod).c_str(), hex(x).c_str());
        }

        int sh = in[i].shift;
        bool lsh_fits = in[i].a.top == 0 || bn_num_bits(&in[i].a) + sh <= BN_MAX_BITS;
        st.check(o.lsh_ok == (int)lsh_fits, "lshift_ok %s << %d: got %d", A, sh, o.lsh_ok);
        if (o.lsh_ok) {
            BN_lshift(x, a, sh);
            st.check(equals(&o.lsh, x), "lshift %s << %d: got %s want %s", A, sh, hex(&o.lsh).c_str(), hex(x).c_str());
        }
        BN_rshift(x, a, sh);
        st.check(equals(&o.rsh, x), "rshift %s >> %d: got %s want %s", A, sh, hex(&o.rsh).c_str(), hex(x).c_str());
        if (bn_num_bits(&in[i].a) < BN_MAX_BITS) {
            BN_lshift1(x, a);
            st.check(equals(&o.lsh1, x), "lshift1 %s: got %s want %s", A, hex(&o.lsh1).c_str(), hex(x).c_str());
        }
        BN_rshift1(x, a);
        st.check(equals(&o.rsh1, x), "rshift1 %s: got %s want %s", A, hex(&o.rsh1).c_str(), hex(x).c_str());

        bool has_inv = BN_mod_inverse(x, a, P, ossl.ctx) != NULL;
        st.check(o.inv_p_ok == (int)has_inv, "inverse mod p of %s: got ok=%d want %d", A, o.inv_p_ok, (int)has_inv);
        if (has_inv && o.inv_p_ok)
            st.check(equals(&o.inv_p, x), "inverse mod p of %s: got %s want %s", A, hex(&o.inv_p).c_str(), hex(x).c_str());
        if (!BN_is_negative(b) && BN_cmp(b, BN_value_one()) > 0) {
            has_inv = BN_mod_inverse(x, a, b, ossl.ctx) != NULL;
            st.check(o.inv_b_ok == (int)has_inv, "inverse of %s mod %s: got ok=%d want %d", A, B, o.inv_b_ok, (int)has_inv);
            if (has_inv && o.inv_b_ok)
                st.check(equals(&o.inv_b, x), "inverse of %s mod %s: got %s want %s", A, B, hex(&o.inv_b).c_str(), hex(x).c_str());
        } else {
            st.check(o.inv_b_ok == 0, "inverse mod %s must fail", B);
        }

        st.check(o.cmp == BN_cmp(a, b), "cmp %s %s: got %d", A, B, o.cmp);
        int uc = BN_ucmp(a, b); /* OpenSSL returns any negative/positive value */
        st.check(o.ucmp == (uc > 0) - (uc < 0), "ucmp %s %s: got %d want %d", A, B, o.ucmp, uc);
        st.check(o.bits == BN_num_bits(a), "num_bits %s: got %d want %d", A, o.bits, BN_num_bits(a));

        BN_nnmod(x, a, P, ossl.ctx);
        BN_nnmod(y, b, P, ossl.ctx);
        BN_mod_add(z, x, y, P, ossl.ctx);
        st.check(equals(&o.qadd, z), "mod_add_quick %s %s: got %s want %s", A, B, hex(&o.qadd).c_str(), hex(z).c_str());
        BN_mod_sub(z, x, y, P, ossl.ctx);
        st.check(equals(&o.qsub, z), "mod_sub_quick %s %s: got %s want %s", A, B, hex(&o.qsub).c_str(), hex(z).c_str());
        BN_mod_lshift1(z, x, P, ossl.ctx);
        st.check(equals(&o.qlsh1, z), "mod_lshift1_quick %s: got %s want %s", A, hex(&o.qlsh1).c_str(), hex(z).c_str());
        BN_mod_lshift(z, x, sh % 8, P, ossl.ctx);
        st.check(equals(&o.qlsh, z), "mod_lshift_quick %s << %d: got %s want %s", A, sh % 8, hex(&o.qlsh).c_str(), hex(z).c_str());

        st.check(o.hex_ok, "hex round trip / in-place forms for %s", A);
        st.check(o.bytes_ok, "bytes round trip for %s", A);
    }
    BN_free(a); BN_free(b); BN_free(P); BN_free(x); BN_free(y); BN_free(z);
    return st.finish("test_bn");
}
