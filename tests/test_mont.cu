/* test_mont.cu - Montgomery arithmetic checked against OpenSSL's BN_MONT_CTX. */
#include "test_util.h"

struct MontIn {
    bn_t m, a, b, e;
};

struct MontOut {
    bn_mont_t ctx;
    int ctx_ok;
    bn_t am, bm, back, prod, prod_back, powm;
};

__global__ void mont_kernel(const MontIn *in, MontOut *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    MontOut o;
    memset(&o, 0, sizeof(o));
    o.ctx_ok = bn_mont_init(&o.ctx, &in[i].m);
    if (o.ctx_ok) {
        bn_to_mont(&o.am, &in[i].a, &o.ctx);
        bn_to_mont(&o.bm, &in[i].b, &o.ctx);
        bn_from_mont(&o.back, &o.am, &o.ctx);
        bn_mont_mul(&o.prod, &o.am, &o.bm, &o.ctx);
        bn_from_mont(&o.prod_back, &o.prod, &o.ctx);
        bn_mod_exp_mont(&o.powm, &in[i].a, &in[i].e, &o.ctx);
    }
    out[i] = o;
}

int main()
{
    Ossl ossl;
    Stats st;

    std::vector<bn_t> moduli = {
        from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"), /* p */
        from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"), /* n */
        from_hex("1"), from_hex("3"), from_hex("ffffffffffffffff"), from_hex("10000000000000001"),
        from_hex("8000000000000000000000000000000000000000000000000000000000000001"),
        from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
        from_hex("2"),                                                                 /* even: rejected */
        from_hex("10000000000000000000000000000000000000000000000000000000000000001"), /* 5 words: rejected */
    };
    for (int bits : {64, 65, 100, 127, 128, 192, 255, 256}) {
        bn_t m = rand_bits(bits);
        m.d[0] |= 1;
        moduli.push_back(m);
    }

    bn_t one = from_hex("1");
    std::vector<MontIn> in;
    int per_modulus = batch_size() / (int)moduli.size() + 1;
    for (const bn_t &m : moduli)
        for (int i = 0; i < per_modulus; i++) {
            MontIn x;
            x.m = m;
            x.a = rand_below(&m);
            x.b = rand_below(&m);
            x.e = (i == 0) ? from_hex("0") : (i == 1) ? from_hex("1") : rand_upto(256);
            if (i == 2) { x.a = from_hex("0"); }
            if (i == 3) { x.a = from_hex("1"); x.b = from_hex("1"); }
            if (i == 4) { x.a = m; bn_usub(&x.a, &x.a, &one); x.b = x.a; } /* m-1 squared */
            in.push_back(x);
        }

    std::vector<MontOut> out = run_on_device<MontIn, MontOut>(in, [&](MontIn *d_in, MontOut *d_out, int n) {
        mont_kernel<<<(n + 127) / 128, 128>>>(d_in, d_out, n);
    });

    BIGNUM *m = BN_new(), *a = BN_new(), *b = BN_new(), *e = BN_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *t = BN_new(), *R = BN_new();
    BN_MONT_CTX *mont = BN_MONT_CTX_new();
    for (size_t i = 0; i < in.size(); i++) {
        const MontOut &o = out[i];
        to_ossl(m, &in[i].m);
        to_ossl(a, &in[i].a);
        to_ossl(b, &in[i].b);
        to_ossl(e, &in[i].e);
        std::string hm = hex(m), ha = hex(a), hb = hex(b), he = hex(e);
        const char *M = hm.c_str(), *A = ha.c_str(), *B = hb.c_str(), *E = he.c_str();

        bool usable = BN_is_odd(m) && in[i].m.top <= BN_WORDS / 2;
        st.check(o.ctx_ok == (int)usable, "mont_init %s: got ok=%d want %d", M, o.ctx_ok, (int)usable);
        if (!o.ctx_ok)
            continue;

        /* context: n0, one = R mod m, rr = R^2 mod m, with R = 2^(64 * words) */
        st.check(in[i].m.d[0] * ((bn_word)0 - o.ctx.n0) == 1, "n0 for %s", M);
        BN_zero(R);
        BN_set_bit(R, 64 * in[i].m.top);
        BN_nnmod(x, R, m, ossl.ctx);
        st.check(equals(&o.ctx.one, x), "one for %s: got %s want %s", M, hex(&o.ctx.one).c_str(), hex(x).c_str());
        BN_mod_sqr(x, x, m, ossl.ctx);
        st.check(equals(&o.ctx.rr, x), "rr for %s: got %s want %s", M, hex(&o.ctx.rr).c_str(), hex(x).c_str());

        /* conversions and products against OpenSSL's own Montgomery code */
        BN_MONT_CTX_set(mont, m, ossl.ctx);
        BN_to_montgomery(x, a, mont, ossl.ctx);
        st.check(equals(&o.am, x), "to_mont %s mod %s: got %s want %s", A, M, hex(&o.am).c_str(), hex(x).c_str());
        BN_to_montgomery(y, b, mont, ossl.ctx);
        st.check(equals(&o.bm, y), "to_mont %s mod %s: got %s want %s", B, M, hex(&o.bm).c_str(), hex(y).c_str());
        BN_nnmod(t, a, m, ossl.ctx);
        st.check(equals(&o.back, t), "from_mont(to_mont(%s)) mod %s: got %s want %s", A, M, hex(&o.back).c_str(), hex(t).c_str());
        BN_mod_mul_montgomery(x, x, y, mont, ossl.ctx);
        st.check(equals(&o.prod, x), "mont_mul %s * %s mod %s: got %s want %s", A, B, M, hex(&o.prod).c_str(), hex(x).c_str());
        BN_mod_mul(x, a, b, m, ossl.ctx);
        st.check(equals(&o.prod_back, x), "from_mont(mont_mul) %s * %s mod %s: got %s want %s", A, B, M, hex(&o.prod_back).c_str(), hex(x).c_str());
        BN_mod_exp(x, a, e, m, ossl.ctx);
        st.check(equals(&o.powm, x), "mod_exp %s ^ %s mod %s: got %s want %s", A, E, M, hex(&o.powm).c_str(), hex(x).c_str());
    }
    BN_MONT_CTX_free(mont);
    BN_free(m); BN_free(a); BN_free(b); BN_free(e); BN_free(x); BN_free(y); BN_free(t); BN_free(R);
    return st.finish("test_mont");
}
