/* test_ec.cu - secp256k1 point arithmetic checked against OpenSSL EC_POINT_*. */
#include "test_util.h"

struct EcIn {
    bn_t k;           /* scalar for multiplication (any sign / size) */
    ec_affine_t P, Q; /* points on the curve */
};

struct EcOut {
    ec_affine_t kP_ladder, kP_vartime, kP_vartime_k2;
    ec_affine_t PplusQ, twoP, PplusP_copy, PminusP, infP, Pinf;
    ec_affine_t A_plus_B, A_plus_Q, P_plus_B, A_plus_Acopy, A_minus_A, twoA;
    ec_affine_t pub;
    int pub_ok, on_curve, comp_len;
    uint8_t comp[33];
};

__constant__ ec_group_t c_group;

__host__ __device__ static void compute(const ec_group_t *g, const EcIn *inp, EcOut *out)
{
    const bn_t k = inp->k;
    const ec_affine_t P = inp->P, Q = inp->Q;
    EcOut o;
    memset(&o, 0, sizeof(o));
    ec_point_t t, pj, qj, A, B, C;

    ec_point_mul(g, &o.kP_ladder, &k, &P);
    ec_point_mul_vartime(g, &t, &k, &P);
    ec_point_to_affine(g, &o.kP_vartime, &t);

    ec_point_from_affine(g, &pj, &P);
    ec_point_from_affine(g, &qj, &Q);

    ec_point_add(g, &t, &pj, &qj);
    ec_point_to_affine(g, &o.PplusQ, &t);
    ec_point_dbl(g, &t, &pj);
    ec_point_to_affine(g, &o.twoP, &t);
    C = pj; /* same point in a different object: exercises the add-as-double path */
    ec_point_add(g, &t, &pj, &C);
    ec_point_to_affine(g, &o.PplusP_copy, &t);
    C = pj;
    ec_point_invert(g, &C);
    ec_point_add(g, &t, &pj, &C);
    ec_point_to_affine(g, &o.PminusP, &t);
    ec_point_set_infinity(&C);
    ec_point_add(g, &t, &C, &pj);
    ec_point_to_affine(g, &o.infP, &t);
    ec_point_add(g, &t, &pj, &C);
    ec_point_to_affine(g, &o.Pinf, &t);

    /* points with Z != 1: A = 2P, B = 2Q as produced by doubling */
    ec_point_dbl(g, &A, &pj);
    ec_point_dbl(g, &B, &qj);
    ec_point_add(g, &t, &A, &B);
    ec_point_to_affine(g, &o.A_plus_B, &t);
    ec_point_add(g, &t, &A, &qj);
    ec_point_to_affine(g, &o.A_plus_Q, &t);
    ec_point_add(g, &t, &pj, &B);
    ec_point_to_affine(g, &o.P_plus_B, &t);
    C = A;
    ec_point_add(g, &t, &A, &C);
    ec_point_to_affine(g, &o.A_plus_Acopy, &t);
    C = A;
    ec_point_invert(g, &C);
    ec_point_add(g, &t, &A, &C);
    ec_point_to_affine(g, &o.A_minus_A, &t);
    ec_point_dbl(g, &t, &A);
    ec_point_to_affine(g, &o.twoA, &t);

    /* vartime multiplication starting from a Z != 1 point: k * (2P) via affine 2P */
    ec_point_mul_vartime(g, &t, &k, &o.twoP);
    ec_point_to_affine(g, &o.kP_vartime_k2, &t);

    o.pub_ok = ec_pubkey_create(g, &o.pub, &k);
    o.on_curve = ec_affine_is_on_curve(g, &P);
    o.comp_len = ec_affine_to_bytes(&P, o.comp, 1);
    *out = o;
}

__global__ void ec_kernel(const EcIn *in, EcOut *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        compute(&c_group, &in[i], &out[i]);
}


/* OpenSSL helpers */
static EC_POINT *point_from_affine(const Ossl &o, const ec_affine_t *a)
{
    EC_POINT *p = EC_POINT_new(o.group);
    if (a->infinity) {
        EC_POINT_set_to_infinity(o.group, p);
        return p;
    }
    BIGNUM *x = BN_new(), *y = BN_new();
    to_ossl(x, &a->x);
    to_ossl(y, &a->y);
    if (!EC_POINT_set_affine_coordinates(o.group, p, x, y, o.ctx)) {
        fprintf(stderr, "point not on curve: %s %s\n", hex(&a->x).c_str(), hex(&a->y).c_str());
        exit(2);
    }
    BN_free(x);
    BN_free(y);
    return p;
}

static ec_affine_t affine_from_point(const Ossl &o, const EC_POINT *p)
{
    ec_affine_t a;
    bn_zero(&a.x);
    bn_zero(&a.y);
    a.infinity = EC_POINT_is_at_infinity(o.group, p);
    if (!a.infinity) {
        BIGNUM *x = BN_new(), *y = BN_new();
        EC_POINT_get_affine_coordinates(o.group, p, x, y, o.ctx);
        from_ossl(&a.x, x);
        from_ossl(&a.y, y);
        BN_free(x);
        BN_free(y);
    }
    return a;
}

static ec_affine_t mul_G(const Ossl &o, const bn_t *k)
{
    BIGNUM *kk = BN_new();
    to_ossl(kk, k);
    EC_POINT *r = EC_POINT_new(o.group);
    EC_POINT_mul(o.group, r, kk, NULL, NULL, o.ctx);
    ec_affine_t a = affine_from_point(o, r);
    EC_POINT_free(r);
    BN_free(kk);
    return a;
}

static bool same_point(const ec_affine_t *a, const ec_affine_t *b)
{
    if (a->infinity || b->infinity)
        return a->infinity == b->infinity;
    return bn_cmp(&a->x, &b->x) == 0 && bn_cmp(&a->y, &b->y) == 0;
}

static std::string show(const ec_affine_t *a)
{
    if (a->infinity)
        return "infinity";
    return "(" + hex(&a->x) + ", " + hex(&a->y) + ")";
}

int main()
{
    Ossl ossl;
    Stats st;

    ec_group_t group;
    if (!ec_group_init_secp256k1(&group)) {
        fprintf(stderr, "group init failed\n");
        return 2;
    }
    if (have_cuda_device())
        CUDA_CHECK(cudaMemcpyToSymbol(c_group, &group, sizeof(group)));

    bn_t order = from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
    bn_t one = from_hex("1");

    /* scalars: edge cases then random, including values outside [0, order) */
    std::vector<bn_t> scalars = {
        from_hex("0"), from_hex("1"), from_hex("2"), from_hex("3"), from_hex("4"), from_hex("7"),
        from_hex("ffffffffffffffff"), from_hex("8000000000000000000000000000000000000000000000000000000000000000"),
        from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
        from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"), /* n */
        from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140"), /* n - 1 */
        from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd036413f"), /* n - 2 */
        from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364142"), /* n + 1 */
        from_hex("7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0"), /* (n - 1) / 2 */
        from_hex("1fffffffffffffffffffffffffffffffd755db9cd5e9140777fa4bd19a06c8282"), /* 2n */
        from_hex("-1"), from_hex("-2"), from_hex("-fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140"),
        from_hex("aa5e28d6a97a2479a65527f7290311a3624d4cc0fa1578598ee3c2613bf99522"),
    };
    int nrand = batch_size();
    for (int i = (int)scalars.size(); i < nrand; i++) {
        bn_t k = (i % 8 == 0) ? rand_upto(512) : rand_below(&order);
        if (i % 16 == 1)
            k.neg = k.top != 0;
        scalars.push_back(k);
    }

    /* base points: small multiples of G first, then random multiples */
    std::vector<EcIn> in;
    for (size_t i = 0; i < scalars.size(); i++) {
        EcIn x;
        x.k = scalars[i];
        bn_t k1, k2;
        if (i < 8) {
            bn_set_word(&k1, (bn_word)i + 1);
            bn_set_word(&k2, (bn_word)i + 2);
        } else if (i == 8) {
            bn_usub(&k1, &order, &one); /* -G */
            bn_set_word(&k2, 1);
        } else {
            k1 = rand_below(&order);
            k2 = rand_below(&order);
            if (k1.top == 0) bn_set_word(&k1, 1);
            if (k2.top == 0) bn_set_word(&k2, 2);
        }
        x.P = mul_G(ossl, &k1);
        x.Q = mul_G(ossl, &k2);
        in.push_back(x);
    }

    std::vector<EcOut> out = run_on_device<EcIn, EcOut>(in, [&](EcIn *d_in, EcOut *d_out, int n) {
        ec_kernel<<<(n + 63) / 64, 64>>>(d_in, d_out, n);
    }, [&](const EcIn *x, EcOut *o) { compute(&group, x, o); });

    BIGNUM *k = BN_new(), *ord = BN_new();
    to_ossl(ord, &order);
    for (size_t i = 0; i < in.size(); i++) {
        const EcOut &o = out[i];
        std::string hk = hex(&in[i].k), sP = show(&in[i].P), sQ = show(&in[i].Q);
        const char *K = hk.c_str();
        EC_POINT *P = point_from_affine(ossl, &in[i].P);
        EC_POINT *Q = point_from_affine(ossl, &in[i].Q);
        EC_POINT *R = EC_POINT_new(ossl.group), *A = EC_POINT_new(ossl.group), *B = EC_POINT_new(ossl.group);
        ec_affine_t want;

        /* k * P, with k reduced mod n on the OpenSSL side */
        to_ossl(k, &in[i].k);
        BN_nnmod(k, k, ord, ossl.ctx);
        EC_POINT_mul(ossl.group, R, NULL, P, k, ossl.ctx);
        want = affine_from_point(ossl, R);
        st.check(same_point(&o.kP_ladder, &want), "ladder %s * %s: got %s want %s", K, sP.c_str(), show(&o.kP_ladder).c_str(), show(&want).c_str());
        st.check(same_point(&o.kP_vartime, &want), "vartime %s * %s: got %s want %s", K, sP.c_str(), show(&o.kP_vartime).c_str(), show(&want).c_str());

        EC_POINT_add(ossl.group, R, P, Q, ossl.ctx);
        want = affine_from_point(ossl, R);
        st.check(same_point(&o.PplusQ, &want), "P + Q for %s + %s: got %s want %s", sP.c_str(), sQ.c_str(), show(&o.PplusQ).c_str(), show(&want).c_str());

        EC_POINT_dbl(ossl.group, A, P, ossl.ctx); /* A = 2P */
        want = affine_from_point(ossl, A);
        st.check(same_point(&o.twoP, &want), "2P for %s: got %s want %s", sP.c_str(), show(&o.twoP).c_str(), show(&want).c_str());
        st.check(same_point(&o.PplusP_copy, &want), "P + P for %s: got %s want %s", sP.c_str(), show(&o.PplusP_copy).c_str(), show(&want).c_str());
        st.check(o.PminusP.infinity, "P - P must be infinity for %s", sP.c_str());
        st.check(same_point(&o.infP, &in[i].P), "inf + P for %s: got %s", sP.c_str(), show(&o.infP).c_str());
        st.check(same_point(&o.Pinf, &in[i].P), "P + inf for %s: got %s", sP.c_str(), show(&o.Pinf).c_str());

        EC_POINT_dbl(ossl.group, B, Q, ossl.ctx); /* B = 2Q */
        EC_POINT_add(ossl.group, R, A, B, ossl.ctx);
        want = affine_from_point(ossl, R);
        st.check(same_point(&o.A_plus_B, &want), "2P + 2Q for %s, %s: got %s want %s", sP.c_str(), sQ.c_str(), show(&o.A_plus_B).c_str(), show(&want).c_str());
        EC_POINT_add(ossl.group, R, A, Q, ossl.ctx);
        want = affine_from_point(ossl, R);
        st.check(same_point(&o.A_plus_Q, &want), "2P + Q for %s, %s: got %s want %s", sP.c_str(), sQ.c_str(), show(&o.A_plus_Q).c_str(), show(&want).c_str());
        EC_POINT_add(ossl.group, R, P, B, ossl.ctx);
        want = affine_from_point(ossl, R);
        st.check(same_point(&o.P_plus_B, &want), "P + 2Q for %s, %s: got %s want %s", sP.c_str(), sQ.c_str(), show(&o.P_plus_B).c_str(), show(&want).c_str());
        EC_POINT_dbl(ossl.group, R, A, ossl.ctx);
        want = affine_from_point(ossl, R);
        st.check(same_point(&o.A_plus_Acopy, &want), "2P + 2P for %s: got %s want %s", sP.c_str(), show(&o.A_plus_Acopy).c_str(), show(&want).c_str());
        st.check(same_point(&o.twoA, &want), "2(2P) for %s: got %s want %s", sP.c_str(), show(&o.twoA).c_str(), show(&want).c_str());
        st.check(o.A_minus_A.infinity, "2P - 2P must be infinity for %s", sP.c_str());

        EC_POINT_mul(ossl.group, R, NULL, A, k, ossl.ctx);
        want = affine_from_point(ossl, R);
        st.check(same_point(&o.kP_vartime_k2, &want), "vartime %s * 2P for %s: got %s want %s", K, sP.c_str(), show(&o.kP_vartime_k2).c_str(), show(&want).c_str());

        /* public key: only for 0 < k < n */
        to_ossl(k, &in[i].k);
        bool valid_priv = !BN_is_negative(k) && !BN_is_zero(k) && BN_cmp(k, ord) < 0;
        st.check(o.pub_ok == (int)valid_priv, "pubkey_create ok for %s: got %d want %d", K, o.pub_ok, (int)valid_priv);
        if (valid_priv) {
            want = mul_G(ossl, &in[i].k);
            st.check(same_point(&o.pub, &want), "pubkey for %s: got %s want %s", K, show(&o.pub).c_str(), show(&want).c_str());
        }

        st.check(o.on_curve == 1, "on_curve for %s", sP.c_str());
        unsigned char buf[33];
        size_t len = EC_POINT_point2oct(ossl.group, P, POINT_CONVERSION_COMPRESSED, buf, sizeof(buf), ossl.ctx);
        st.check(o.comp_len == (int)len && memcmp(o.comp, buf, len) == 0, "compressed encoding for %s", sP.c_str());

        EC_POINT_free(P); EC_POINT_free(Q); EC_POINT_free(R); EC_POINT_free(A); EC_POINT_free(B);
    }

    /* a point that is not on the curve must be rejected */
    ec_affine_t bad = in[0].P;
    bad.y.d[0] ^= 1;
    st.check(!ec_affine_is_on_curve(&group, &bad), "off-curve point accepted");

    BN_free(k); BN_free(ord);
    return st.finish("test_ec");
}
