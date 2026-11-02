/* One multiple-shooting segment in MPFR: series(), step_size() and sweep() of oracle_ns.py, operation for operation
   (same correctly rounded MPFR operations in the same order), so the C and Python sweeps agree bit for bit and the
   Python one stays the reference. Numbers cross the boundary exactly, as hex "mantissa p exponent" tokens.
   Built and loaded by oracle_ns.py (ctypes). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <mpfr.h>

#define RN MPFR_RNDN

typedef struct {
    int n, npairs, npots, nterms, nsame, nJ, maxpow;
    long Z;
    int *l, *pi, *pj, *lq;          /* pairs (i, j), l_i + l_j */
    int *pq, *pk, *nu, *self;       /* pots (pair, k), is a nu (not (aa, 0)); self[a] = pot (aa, 0) */
    int *t0, *tb, *tm, *tp;         /* terms of orbital a: t0[a] .. t0[a+1]-1, each (b, power m, pot) */
    int ngroups, *g0, *gb, *gx0, *gx; /* the terms of a grouped by the orbital b they multiply: groups g0[a] ..
                                         g0[a+1]-1, group G multiplies f_gb[G] and holds terms gx[gx0[G] .. gx0[G+1]-1];
                                         each a's first group is b = a and also carries the direct term r D */
    mpq_t *te;                      /* and its coefficient e */
    int *same, *jq, *jp, *jk;       /* same-l pairs; J = (pair q, pot of (p, k), k) */
} sys_t;

static int *copy(const int *a, int n) {
    int *b = malloc((n ? n : 1) * sizeof(int));
    memcpy(b, a, n * sizeof(int));
    return b;
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void *frost_sys(int n, long Z, const int *l, int npairs, const int *pi, const int *pj, int npots, const int *pq,
                const int *pk, int nterms, const int *ta, const int *tb, const int *tm, const int *tp, const char *te,
                int nsame, const int *same, int nJ, const int *jq, const int *jp, const int *jk) {
    sys_t *s = calloc(1, sizeof(sys_t));
    s->n = n, s->Z = Z, s->npairs = npairs, s->npots = npots, s->nterms = nterms, s->nsame = nsame, s->nJ = nJ;
    s->l = copy(l, n), s->pi = copy(pi, npairs), s->pj = copy(pj, npairs), s->pq = copy(pq, npots), s->pk = copy(pk, npots);
    s->tb = copy(tb, nterms), s->tm = copy(tm, nterms), s->tp = copy(tp, nterms), s->same = copy(same, nsame);
    s->jq = copy(jq, nJ), s->jp = copy(jp, nJ), s->jk = copy(jk, nJ);
    s->lq = malloc(npairs * sizeof(int));
    int mp = 1;
    for (int q = 0; q < npairs; q++) s->lq[q] = l[pi[q]] + l[pj[q]];
    s->nu = malloc(npots * sizeof(int)), s->self = malloc(n * sizeof(int));
    for (int x = 0; x < npots; x++) {
        int q = pq[x], diag = pi[q] == pj[q];
        s->nu[x] = !(diag && pk[x] == 0);
        if (diag && pk[x] == 0) s->self[pi[q]] = x;
        mp = MAX(mp, s->lq[q] + 1 - pk[x]), mp = MAX(mp, s->lq[q] + pk[x] + 2);
    }
    s->t0 = calloc(n + 1, sizeof(int));
    for (int x = 0; x < nterms; x++) s->t0[ta[x] + 1]++, mp = MAX(mp, tm[x]);
    for (int a = 0; a < n; a++) s->t0[a + 1] += s->t0[a], mp = MAX(mp, 2 * l[a] + 2);
    s->maxpow = mp;
    s->g0 = malloc((n + 1) * sizeof(int)), s->gb = malloc(n * n * sizeof(int));
    s->gx0 = malloc((n * n + 1) * sizeof(int)), s->gx = malloc((nterms ? nterms : 1) * sizeof(int));
    int G = 0, X = 0;
    for (int a = 0; a < n; a++) {
        s->g0[a] = G;
        for (int bb = -1; bb < n; bb++) {
            int b = bb < 0 ? a : bb, start = X;
            if (bb >= 0 && b == a) continue;
            for (int x = s->t0[a]; x < s->t0[a + 1]; x++)
                if (s->tb[x] == b) s->gx[X++] = x;
            if (b != a && X == start) continue;
            s->gb[G] = b, s->gx0[G] = start, G++;
        }
    }
    s->g0[n] = G, s->gx0[G] = X, s->ngroups = G;
    s->te = malloc((nterms ? nterms : 1) * sizeof(mpq_t));
    const char *c = te;
    for (int x = 0; x < nterms; x++) {
        char buf[256];
        int len = 0;
        while (*c == ' ') c++;
        while (*c && *c != ' ' && len < 255) buf[len++] = *c++;
        buf[len] = 0;
        mpq_init(s->te[x]);
        mpq_set_str(s->te[x], buf, 10);
        mpq_canonicalize(s->te[x]);
    }
    return s;
}

/* growable output text */
typedef struct { char *s; size_t len, cap; } buf_t;

static void puts_(buf_t *b, const char *t, size_t n) {
    if (b->len + n + 1 > b->cap) {
        b->cap = 2 * (b->len + n + 1);
        b->s = realloc(b->s, b->cap);
    }
    memcpy(b->s + b->len, t, n);
    b->len += n;
    b->s[b->len] = 0;
}

static mpz_t Zt;
static char *zs;
static size_t zcap;

static void put(buf_t *b, mpfr_srcptr x) {
    if (mpfr_zero_p(x)) {
        puts_(b, "0p0 ", 4);
        return;
    }
    if (!mpfr_number_p(x)) {  /* the Python side checks the residual for these */
        const char *t = mpfr_nan_p(x) ? "@NaN@ " : mpfr_signbit(x) ? "-@Inf@ " : "@Inf@ ";
        puts_(b, t, strlen(t));
        return;
    }
    long e = mpfr_get_z_2exp(Zt, x);
    size_t need = mpz_sizeinbase(Zt, 16) + 32;
    if (need > zcap) zcap = 2 * need, zs = realloc(zs, zcap);
    mpz_get_str(zs, 16, Zt);
    size_t k = strlen(zs);
    k += sprintf(zs + k, "p%ld ", e);
    puts_(b, zs, k);
}

/* next token of the input as an exact mpfr (precision large enough for its mantissa, at least bits) */
static const char *get(mpfr_t x, const char *c, long bits) {
    while (*c == ' ') c++;
    const char *e = c;
    while (*e && *e != ' ') e++;
    long need = 4 * (long)(e - c) + 8;
    static char *tok;  /* mpfr_strtofr runs strlen over its whole argument: give it the token alone, not the rest */
    static size_t cap;
    if ((size_t)(e - c) + 1 > cap) cap = 2 * (e - c) + 64, tok = realloc(tok, cap);
    memcpy(tok, c, e - c), tok[e - c] = 0;
    mpfr_set_prec(x, MAX(bits, need));
    mpfr_strtofr(x, tok, NULL, 16, RN);
    return e;
}

static mpfr_t *vec(size_t len, long bits) {
    mpfr_t *v = malloc((len ? len : 1) * sizeof(mpfr_t));
    for (size_t i = 0; i < len; i++) mpfr_init2(v[i], bits), mpfr_set_zero(v[i], 1);
    return v;
}

static void unvec(mpfr_t *v, size_t len) {
    for (size_t i = 0; i < len; i++) mpfr_clear(v[i]);
    free(v);
}

/* sum a[i] b[k-i] (r must not alias a or b), rounded once: every product is formed exactly from the limbs
   (value = M 2^(exp - 64 limbs)) and added into a fixed-point accumulator whose last bit is 128 below the largest
   product, so only bits that far down are cut (< 1 ulp of the result in all), against the 2k + 1 roundings of a
   mul/add loop. Positive and negative parts are summed apart; W limbs leave headroom for the carries. */
#define DOTW 260  /* accumulator limbs: 2 * 128 limbs of operand (8192 bits) + 3 */
#define DOTK 4096 /* terms */
static void dotr(mpfr_t r, mpfr_t *a, mpfr_t *b, int k, int lo, int hi, mpfr_t t) {  /* sum_{i=lo}^{hi} a[i] b[k-i] */
    static mp_limb_t pos[DOTW], neg[DOTW], prod[DOTW], sh[DOTW];
    static mpfr_exp_t ex[DOTK];
    mpfr_exp_t emax = 0;
    int any = 0, L = 0;
    if (hi - lo >= DOTK) goto slow;
    for (int i = lo; i <= hi; i++) {
        mpfr_srcptr x = a[i], y = b[k - i];
        if (mpfr_zero_p(x) || mpfr_zero_p(y)) {
            ex[i - lo] = __MPFR_EXP_ZERO;  /* marker: skipped (no sum of two exponents is this) */
            continue;
        }
        if (!mpfr_regular_p(x) || !mpfr_regular_p(y)) goto slow;
        mpfr_exp_t e = mpfr_get_exp(x) + mpfr_get_exp(y);
        int np = (int)((mpfr_get_prec(x) + 63) / 64 + (mpfr_get_prec(y) + 63) / 64);
        ex[i - lo] = e;
        if (!any || e > emax) emax = e;
        if (np > L) L = np;
        any = 1;
    }
    if (!any) {
        mpfr_set_zero(r, 1);
        return;
    }
    int W = L + 3;
    if (W > DOTW) goto slow;
    memset(pos, 0, W * sizeof(mp_limb_t)), memset(neg, 0, W * sizeof(mp_limb_t));
    for (int i = lo; i <= hi; i++) {
        if (ex[i - lo] == __MPFR_EXP_ZERO) continue;
        mpfr_srcptr x = a[i], y = b[k - i];
        int nx = (int)((mpfr_get_prec(x) + 63) / 64), ny = (int)((mpfr_get_prec(y) + 63) / 64), np = nx + ny;
        if (nx >= ny) mpn_mul(prod, x->_mpfr_d, nx, y->_mpfr_d, ny);
        else mpn_mul(prod, y->_mpfr_d, ny, x->_mpfr_d, nx);
        /* bits to shift the product left so that its last bit lands on the accumulator's (2^(emax - 64 L - 128)) */
        long s = 128 - (long)(emax - ex[i - lo]) + 64L * (L - np);
        mp_limb_t *acc = mpfr_signbit(x) != mpfr_signbit(y) ? neg : pos;
        if (s >= 0) {
            int q = (int)(s / 64), c = (int)(s % 64);
            if (c) sh[np] = mpn_lshift(sh, prod, np, c);
            else memcpy(sh, prod, np * sizeof(mp_limb_t)), sh[np] = 0;
            mpn_add(acc + q, acc + q, W - q, sh, np + 1);
        } else {
            int q = (int)(-s / 64), c = (int)(-s % 64);
            if (q >= np) continue;  /* entirely below the last bit kept */
            if (c) mpn_rshift(sh, prod + q, np - q, c);
            else memcpy(sh, prod + q, (np - q) * sizeof(mp_limb_t));
            mpn_add(acc, acc, W, sh, np - q);
        }
    }
    int sgn = mpn_cmp(pos, neg, W), n = W;
    if (sgn >= 0) mpn_sub_n(prod, pos, neg, W);
    else mpn_sub_n(prod, neg, pos, W);
    while (n > 0 && prod[n - 1] == 0) n--;
    if (!n) {
        mpfr_set_zero(r, 1);
        return;
    }
    mpz_t z;
    mpfr_set_z_2exp(r, mpz_roinit_n(z, prod, sgn >= 0 ? n : -n), emax - 64L * L - 128, RN);
    return;
slow:
    mpfr_set_zero(r, 1);
    for (int i = lo; i <= hi; i++) {
        mpfr_mul(t, a[i], b[k - i], RN);
        mpfr_add(r, r, t, RN);
    }
}

static void dot(mpfr_t r, mpfr_t *a, mpfr_t *b, int k, mpfr_t t) { dotr(r, a, b, k, 0, k, t); }

typedef struct {
    int origin;  /* r0 == 0 */
    mpfr_ptr r0;
    mpfr_t *Rm;  /* (r0 + x)^m coefficients, row m of stride maxpow + 1 */
    int stride;
} ctx_t;

/* (r^m g)_k about r0, or NULL for Python's int 0 (at the origin, k < m) */
static mpfr_ptr rmul(ctx_t *c, mpfr_t r, int m, mpfr_t *g, int k, mpfr_t t) {
    if (c->origin) return k >= m ? g[k - m] : NULL;
    if (m == 1) {
        mpfr_mul(r, c->r0, g[k], RN);
        if (k) mpfr_add(r, r, g[k - 1], RN);
        return r;
    }
    if (m == 0) return g[k];
    dotr(r, c->Rm + m * c->stride, g, k, 0, m < k ? m : k, t);
    return r;
}

/* next coefficient of r u'' + 2 (L + 1) u' = g (g NULL: Python's int 0) */
static void step(ctx_t *c, mpfr_t out, mpfr_t *u, mpfr_ptr g, int L, int k, mpfr_t t1, mpfr_t t2) {
    long m = (long)(k + 1) * (k + 2 * L + 2);
    if (c->origin) {
        if (g) mpfr_div_si(out, g, m, RN);
        else mpfr_set_zero(out, 1);
        return;
    }
    mpfr_mul_si(t1, u[k + 1], m, RN);
    mpfr_sub(t1, g, t1, RN);
    mpfr_mul_si(t2, c->r0, k + 1, RN);
    mpfr_mul_si(t2, t2, k + 2, RN);
    mpfr_div(out, t1, t2, RN);
}

static void value(mpfr_t v, mpfr_t *c, int len, mpfr_srcptr h) {
    mpfr_set_zero(v, 1);
    for (int i = len - 1; i >= 0; i--) {
        mpfr_mul(v, v, h, RN);
        mpfr_add(v, v, c[i], RN);
    }
}

static void slope(mpfr_t v, mpfr_t *c, int len, mpfr_srcptr h, mpfr_t t) {
    mpfr_set_zero(v, 1);
    for (int i = len - 1; i >= 1; i--) {
        mpfr_mul(v, v, h, RN);
        mpfr_mul_si(t, c[i], i, RN);
        mpfr_add(v, v, t, RN);
    }
}

/* in: r0 r1 c t eps[n] (f f')[n] (y y')[npots] mesh[nmesh]; mode 1 = energy integrals, 2 = also elements.
   out: (f f')[n] (y y')[npots] [T V (same) M (pots) J] | h[steps] | per step: r0 h f_a coefficients */
char *frost_sweep(void *sp, long bits, const char *in, int p, int nmesh, int mode) {
    sys_t *s = sp;
    int n = s->n, np = s->npots, energy = mode >= 1, L = p + 2;
    static int ready;
    if (!ready) mpz_init(Zt), ready = 1;
    mpfr_t r0in, r1, c, t, *eps = vec(n, bits), *mesh = vec(nmesh, bits);
    mpfr_inits2(bits, r0in, r1, c, t, (mpfr_ptr)0);
    mpfr_t *F = vec((size_t)n * L, bits), *Y = vec((size_t)np * L, bits);
    const char *q = in;
    q = get(r0in, q, bits), q = get(r1, q, bits), q = get(c, q, bits), q = get(t, q, bits);
    for (int a = 0; a < n; a++) q = get(eps[a], q, bits);
    for (int a = 0; a < n; a++) q = get(F[a * L], q, bits), q = get(F[a * L + 1], q, bits);
    for (int x = 0; x < np; x++) q = get(Y[x * L], q, bits), q = get(Y[x * L + 1], q, bits);
    for (int i = 0; i < nmesh; i++) q = get(mesh[i], q, bits);

    mpfr_t *te = vec(s->nterms, bits);  /* gmpy2's mpq * mpfr rounds the mpq to an mpfr first: so do we */
    for (int x = 0; x < s->nterms; x++) mpfr_set_q(te[x], s->te[x], RN);
    mpfr_t *ff = vec((size_t)s->npairs * p, bits), *W = vec((size_t)np * p, bits), *D = vec((size_t)n * p, bits),
           *rD = vec((size_t)n * p, bits), *V = vec((size_t)s->ngroups * p, bits), *nw = vec(n, bits);
    int ns = s->nsame, nJ = s->nJ, E1 = p + 1;
    mpfr_t *df = vec(energy ? (size_t)n * p : 0, bits), *dd = vec(energy ? (size_t)ns * p : 0, bits),
           *fy = vec(energy ? (size_t)nJ * p : 0, bits), *ET = vec(energy ? (size_t)ns * E1 : 0, bits),
           *EV = vec(energy ? (size_t)ns * E1 : 0, bits), *EM = vec(energy ? (size_t)np * E1 : 0, bits),
           *EJ = vec(energy ? (size_t)nJ * E1 : 0, bits);
    int st = s->maxpow + 1;
    mpfr_t *Rm = vec((size_t)st * st, bits), *sf = vec(2 * n, bits), *sy = vec(2 * np, bits);
    mpfr_t S, g, g2, ta, tb, tc, tv, h, rho, r0b, one;
    mpfr_inits2(bits, S, g, g2, ta, tb, tc, tv, h, rho, r0b, one, (mpfr_ptr)0);
    mpfr_ptr r0 = r0in;
    ctx_t cx = {0, 0, Rm, st};
    buf_t out = {0}, el = {0}, hs = {0};
    int steps = 0;
    double mn = 0, cap = INFINITY;  /* stiff_cap(): c p / (e kappa_max) in 2^-16 units, kappa = sqrt(-2 eps) */
    for (int a = 0; a < n; a++)
        if (mpfr_get_d(eps[a], RN) < mn) mn = mpfr_get_d(eps[a], RN);
    if (mn < 0) cap = floor(mpfr_get_d(c, RN) * p / (M_E * sqrt(-2 * mn)) * 65536.0);

    while (mpfr_less_p(r0, r1)) {
        cx.origin = mpfr_zero_p(r0), cx.r0 = r0;
        int off = cx.origin ? 1 : 2;
        if (cx.origin) {  /* the Frobenius start uses values only: slot 1 receives a result, at bits */
            for (int a = 0; a < n; a++) mpfr_set_prec(F[a * L + 1], bits);
            for (int x = 0; x < np; x++) mpfr_set_prec(Y[x * L + 1], bits);
        } else
            for (int m = 2; m <= s->maxpow; m++)
                for (int j = 0; j <= m; j++) {
                    mpfr_pow_si(ta, r0, m - j, RN);
                    mpz_t b;
                    mpz_init(b);
                    mpz_bin_uiui(b, m, j);
                    mpfr_mul_z(Rm[m * st + j], ta, b, RN);
                    mpz_clear(b);
                }
        for (int k = 0; k < p; k++) {
            for (int x = 0; x < s->npairs; x++)
                dot(ff[x * p + k], F + s->pi[x] * L, F + s->pj[x] * L, k, ta);
            for (int x = 0; x < np; x++) {
                int qq = s->pq[x], kk = s->pk[x];
                mpfr_ptr rm = rmul(&cx, tb, s->lq[qq] + 1 - kk, ff + qq * p, k, ta), gg = NULL;
                if (rm) mpfr_mul_si(g, rm, -(2 * kk + 1), RN), gg = g;
                step(&cx, Y[x * L + k + off], Y + x * L, gg, kk, k, ta, tb);
            }
            for (int x = 0; x < np; x++)
                if (s->nu[x]) mpfr_mul(W[x * p + k], t, Y[x * L + k], RN);
            mpfr_set_zero(S, 1);
            for (int b = 0; b < n; b++) {
                mpfr_mul_si(ta, Y[s->self[b] * L + k], 2 * s->l[b] + 1, RN);
                mpfr_add(S, S, ta, RN);
            }
            for (int a = 0; a < n; a++) {
                mpfr_t *fa = F + a * L;
                mpfr_mul_si(ta, S, 2, RN);
                mpfr_sub(ta, ta, Y[s->self[a] * L + k], RN);
                mpfr_mul(D[a * p + k], t, ta, RN);
                mpfr_ptr rd = rmul(&cx, rD[a * p + k], 1, D + a * p, k, ta);
                if (rd && rd != rD[a * p + k]) mpfr_set(rD[a * p + k], rd, RN);
                else if (!rd) mpfr_set_zero(rD[a * p + k], 1);
                /* sum_x e_x r^m_x (W_x f_b) = (sum_x e_x r^m_x W_x) f_b: one series per orbital b, one product */
                mpfr_set_zero(g, 1);
                for (int G = s->g0[a]; G < s->g0[a + 1]; G++) {
                    mpfr_t *v = V + G * p;
                    if (s->gb[G] == a) mpfr_set(v[k], rD[a * p + k], RN);
                    else mpfr_set_zero(v[k], 1);
                    for (int j = s->gx0[G]; j < s->gx0[G + 1]; j++) {
                        int x = s->gx[j];
                        mpfr_ptr rw = rmul(&cx, tb, s->tm[x], W + s->tp[x] * p, k, tc);
                        if (rw) {
                            mpfr_mul(ta, te[x], rw, RN);
                            mpfr_sub(v[k], v[k], ta, RN);
                        }
                    }
                    dot(tv, v, F + s->gb[G] * L, k, ta);
                    mpfr_add(g, g, tv, RN);
                }
                mpfr_mul_si(ta, fa[k], s->Z, RN);
                mpfr_sub(g, g, ta, RN);
                mpfr_ptr rf = rmul(&cx, tb, 1, fa, k, tc);
                if (rf) mpfr_mul(ta, eps[a], rf, RN);
                else mpfr_mul_si(ta, eps[a], 0, RN);
                mpfr_sub(g, g, ta, RN);
                mpfr_mul_si(g2, g, 2, RN);
                step(&cx, nw[a], fa, g2, s->l[a], k, ta, tb);
            }
            for (int a = 0; a < n; a++)  /* a swap moves precisions too: results are always rounded to bits */
                if (mpfr_swap(F[a * L + k + off], nw[a]), mpfr_get_prec(nw[a]) != bits) mpfr_set_prec(nw[a], bits);
            if (!energy) continue;
            for (int a = 0; a < n; a++) mpfr_mul_si(df[a * p + k], F[a * L + k + 1], k + 1, RN);
            for (int x = 0; x < ns; x++) {
                int qq = s->same[x], la = s->l[s->pi[qq]];
                dot(dd[x * p + k], df + s->pi[qq] * p, df + s->pj[qq] * p, k, ta);
                mpfr_ptr rt = rmul(&cx, tb, 2 * la + 2, dd + x * p, k, tc);
                if (rt) mpfr_div_si(ET[x * E1 + k + 1], rt, 2 * (k + 1), RN);
                else mpfr_set_zero(ET[x * E1 + k + 1], 1);
                mpfr_ptr rv = rmul(&cx, tb, 2 * la + 1, ff + qq * p, k, tc);
                if (rv) {
                    mpfr_mul_si(ta, rv, -s->Z, RN);
                    mpfr_div_si(EV[x * E1 + k + 1], ta, k + 1, RN);
                } else mpfr_set_zero(EV[x * E1 + k + 1], 1);
            }
            for (int x = 0; x < np; x++) {
                int qq = s->pq[x];
                mpfr_ptr rm = rmul(&cx, tb, s->lq[qq] + s->pk[x] + 2, ff + qq * p, k, tc);
                if (rm) mpfr_div_si(EM[x * E1 + k + 1], rm, k + 1, RN);
                else mpfr_set_zero(EM[x * E1 + k + 1], 1);
            }
            for (int x = 0; x < nJ; x++) {
                int qq = s->jq[x];
                dot(fy[x * p + k], ff + qq * p, Y + s->jp[x] * L, k, ta);
                mpfr_ptr rj = rmul(&cx, tb, s->lq[qq] + s->jk[x] + 2, fy + x * p, k, tc);
                if (rj) mpfr_div_si(EJ[x * E1 + k + 1], rj, k + 1, RN);
                else mpfr_set_zero(EJ[x * E1 + k + 1], 1);
            }
        }
        int len = p + off;
        if (steps < nmesh) mpfr_set(h, mesh[steps], RN);
        else {  /* step_size: c times the root-test radius (min over orbitals) on a 2^-16 grid, at most cap */
            int first = 1;
            mpfr_set_prec(h, bits);
            for (int a = 0; a < n; a++) {
                mpfr_t *fa = F + a * L;
                for (int k = len - 9; k < len; k++) {
                    if (mpfr_zero_p(fa[k])) continue;
                    mpfr_abs(ta, fa[0], RN), mpfr_abs(tb, fa[1], RN);
                    if (mpfr_greater_p(tb, ta)) mpfr_swap(ta, tb);
                    mpfr_abs(tb, fa[k], RN);
                    mpfr_div(ta, ta, tb, RN);
                    mpfr_set_ui(tc, 1, RN), mpfr_div_si(tc, tc, k, RN);
                    mpfr_pow(ta, ta, tc, RN);
                    if (first || mpfr_less_p(ta, rho)) mpfr_set(rho, ta, RN), first = 0;
                }
            }
            mpfr_mul(ta, c, rho, RN);
            double d = floor(mpfr_get_d(ta, RN) * 65536.0);
            mpfr_set_d(h, d < cap ? d : cap, RN), mpfr_div_ui(h, h, 65536, RN);
            mpfr_set_ui(tb, 1, RN), mpfr_div_ui(tb, tb, 65536, RN);
            if (mpfr_less_p(h, tb)) mpfr_set(h, tb, RN);
            mpfr_sub(tb, r1, r0, RN);
            if (mpfr_less_p(tb, h)) mpfr_set(h, tb, RN);
        }
        put(&hs, h);
        if (mode == 2) {
            put(&el, r0), put(&el, h);
            for (int a = 0; a < n; a++)
                for (int k = 0; k < len; k++) put(&el, F[a * L + k]);
        }
        for (int a = 0; a < n; a++) value(sf[2 * a], F + a * L, len, h), slope(sf[2 * a + 1], F + a * L, len, h, ta);
        for (int x = 0; x < np; x++) value(sy[2 * x], Y + x * L, len, h), slope(sy[2 * x + 1], Y + x * L, len, h, ta);
        for (int a = 0; a < n; a++)
            for (int i = 0; i < 2; i++) mpfr_swap(F[a * L + i], sf[2 * a + i]), mpfr_set_prec(sf[2 * a + i], bits);
        for (int x = 0; x < np; x++)
            for (int i = 0; i < 2; i++) mpfr_swap(Y[x * L + i], sy[2 * x + i]), mpfr_set_prec(sy[2 * x + i], bits);
        if (energy) {  /* the integrals so far: value of each integrated series at h */
            mpfr_t *Es[4] = {ET, EV, EM, EJ};
            int cnt[4] = {ns, ns, np, nJ};
            for (int e = 0; e < 4; e++)
                for (int x = 0; x < cnt[e]; x++) value(ta, Es[e] + x * E1, E1, h), mpfr_set(Es[e][x * E1], ta, RN);
        }
        mpfr_add(r0b, r0, h, RN);
        r0 = r0b;
        steps++;
    }
    for (int a = 0; a < n; a++) put(&out, F[a * L]), put(&out, F[a * L + 1]);
    for (int x = 0; x < np; x++) put(&out, Y[x * L]), put(&out, Y[x * L + 1]);
    if (energy) {
        mpfr_t *Es[4] = {ET, EV, EM, EJ};
        int cnt[4] = {ns, ns, np, nJ};
        for (int e = 0; e < 4; e++)
            for (int x = 0; x < cnt[e]; x++) put(&out, Es[e][x * E1]);
    }
    puts_(&out, "|", 1);
    if (hs.s) puts_(&out, hs.s, hs.len);
    puts_(&out, "|", 1);
    if (el.s) puts_(&out, el.s, el.len);
    free(hs.s), free(el.s);

    mpfr_clears(r0in, r1, c, t, S, g, g2, ta, tb, tc, tv, h, rho, r0b, one, (mpfr_ptr)0);
    unvec(te, s->nterms), unvec(eps, n), unvec(mesh, nmesh), unvec(F, (size_t)n * L), unvec(Y, (size_t)np * L);
    unvec(ff, (size_t)s->npairs * p), unvec(W, (size_t)np * p), unvec(D, (size_t)n * p), unvec(rD, (size_t)n * p);
    unvec(V, (size_t)s->ngroups * p), unvec(nw, n), unvec(Rm, (size_t)st * st), unvec(sf, 2 * n), unvec(sy, 2 * np);
    unvec(df, energy ? (size_t)n * p : 0), unvec(dd, energy ? (size_t)ns * p : 0), unvec(fy, energy ? (size_t)nJ * p : 0);
    unvec(ET, energy ? (size_t)ns * E1 : 0), unvec(EV, energy ? (size_t)ns * E1 : 0);
    unvec(EM, energy ? (size_t)np * E1 : 0), unvec(EJ, energy ? (size_t)nJ * E1 : 0);
    return out.s;
}

void frost_free(char *s) { free(s); }

/* Sparse LU of the multiple-shooting Jacobian, as factor() / lu_solve() of oracle_ns.py: row equilibration, then
   column by column partial pivoting among the free rows with a nonzero there. Row i is kept as a dense window of
   band columns [lo, hi) plus the ne dense columns at the end (the eps); fill only ever widens a window. */
typedef struct {
    int lo, hi;
    mpfr_t *v, *t;  /* window v[c - lo], tail t[c - (N - ne)] */
} row_t;

typedef struct {
    int N, ne, nops;
    long bits;
    row_t *r;
    mpfr_t *scale;
    int *piv, *opp, *opi;
    mpfr_t *opm;
} lu_t;

static mpfr_ptr at(lu_t *L, row_t *w, int c) {  /* NULL if outside the window */
    if (c >= L->N - L->ne) return w->t[c - (L->N - L->ne)];
    return c >= w->lo && c < w->hi ? w->v[c - w->lo] : NULL;
}

static void widen(lu_t *L, row_t *w, int hi) {
    if (hi <= w->hi) return;
    w->v = realloc(w->v, (hi - w->lo) * sizeof(mpfr_t));
    for (int c = w->hi; c < hi; c++) mpfr_init2(w->v[c - w->lo], L->bits), mpfr_set_zero(w->v[c - w->lo], 1);
    w->hi = hi;
}

void frost_lu_free(void *lp) {
    lu_t *L = lp;
    if (!L) return;
    for (int i = 0; i < L->N; i++) {
        unvec(L->r[i].v, L->r[i].hi - L->r[i].lo);
        unvec(L->r[i].t, L->ne);
    }
    free(L->r), unvec(L->scale, L->N), free(L->piv), free(L->opp), free(L->opi), unvec(L->opm, L->nops);
    free(L);
}

/* in: "row col value" triplets (any order); NULL if singular */
void *frost_lu(long bits, int N, int ne, const char *in) {
    lu_t *L = calloc(1, sizeof(lu_t));
    L->N = N, L->ne = ne, L->bits = bits;
    L->r = calloc(N, sizeof(row_t)), L->scale = vec(N, bits), L->piv = malloc(N * sizeof(int));
    int B = N - ne;
    for (int i = 0; i < N; i++) L->r[i].lo = B, L->r[i].hi = -1;
    char *q;
    for (const char *c = in;;) {  /* pass 1: each row's band window */
        long i = strtol(c, &q, 10);
        if (q == c) break;
        long col = strtol(q, &q, 10);
        while (*q == ' ') q++;
        while (*q && *q != ' ') q++;
        c = q;
        if (col < B) {
            if (col < L->r[i].lo) L->r[i].lo = col;
            if (col + 1 > L->r[i].hi) L->r[i].hi = col + 1;
        }
    }
    for (int i = 0; i < N; i++) {
        row_t *w = L->r + i;
        if (w->hi < 0) w->lo = w->hi = B;
        w->v = vec(w->hi - w->lo, bits), w->t = vec(ne, bits);
    }
    mpfr_t tok, mx, m, t;
    mpfr_inits2(bits, tok, mx, m, t, (mpfr_ptr)0);
    for (const char *c = in;;) {  /* pass 2: values */
        long i = strtol(c, &q, 10);
        if (q == c) break;
        long col = strtol(q, &q, 10);
        c = get(tok, q, bits);
        mpfr_set(at(L, L->r + i, col), tok, RN);
    }
    for (int i = 0; i < N; i++) {  /* row equilibration */
        row_t *w = L->r + i;
        mpfr_set_zero(mx, 1);
        for (int k = 0; k < w->hi - w->lo + ne; k++) {
            mpfr_abs(t, k < w->hi - w->lo ? w->v[k] : w->t[k - (w->hi - w->lo)], RN);
            if (mpfr_greater_p(t, mx)) mpfr_set(mx, t, RN);
        }
        mpfr_ui_div(L->scale[i], 1, mx, RN);
        for (int k = 0; k < w->hi - w->lo; k++) mpfr_mul(w->v[k], w->v[k], L->scale[i], RN);
        for (int k = 0; k < ne; k++) mpfr_mul(w->t[k], w->t[k], L->scale[i], RN);
    }
    mpfr_clear(tok);
    int cap = 1024, *act = malloc(N * sizeof(int)), nact = 0, next = 0, *ord = malloc(N * sizeof(int));
    L->opp = malloc(cap * sizeof(int)), L->opi = malloc(cap * sizeof(int)), L->opm = malloc(cap * sizeof(mpfr_t));
    for (int i = 0; i < N; i++) ord[i] = i;  /* rows enter the active set in order of their first column */
    for (int i = 1; i < N; i++)
        for (int j = i; j > 0 && L->r[ord[j - 1]].lo > L->r[ord[j]].lo; j--) {
            int s = ord[j];
            ord[j] = ord[j - 1], ord[j - 1] = s;
        }
    for (int col = 0; col < N; col++) {
        while (next < N && (L->r[ord[next]].lo <= col || col >= N - ne)) act[nact++] = ord[next++];
        int best = -1;
        for (int a = 0; a < nact; a++) {
            mpfr_ptr e = at(L, L->r + act[a], col);
            if (!e || mpfr_zero_p(e)) continue;
            mpfr_abs(t, e, RN);
            if (best < 0 || mpfr_greater_p(t, mx)) best = a, mpfr_set(mx, t, RN);
        }
        if (best < 0) {
            free(act), free(ord), mpfr_clears(mx, m, t, (mpfr_ptr)0);
            frost_lu_free(L);
            return NULL;
        }
        int p = act[best];
        act[best] = act[--nact];
        L->piv[col] = p;
        row_t *P = L->r + p;
        mpfr_ptr pc = at(L, P, col);
        for (int a = 0; a < nact; a++) {
            row_t *w = L->r + act[a];
            mpfr_ptr e = at(L, w, col);
            if (!e || mpfr_zero_p(e)) continue;
            mpfr_div(m, e, pc, RN);
            mpfr_set_zero(e, 1);
            if (col < N - ne) {
                widen(L, w, P->hi);
                for (int c = col + 1; c < P->hi; c++) {
                    mpfr_ptr pv = P->v[c - P->lo];
                    if (mpfr_zero_p(pv)) continue;
                    mpfr_mul(t, m, pv, RN);
                    mpfr_sub(w->v[c - w->lo], w->v[c - w->lo], t, RN);
                }
            }
            for (int k = (col < N - ne ? 0 : col - (N - ne) + 1); k < ne; k++) {
                if (mpfr_zero_p(P->t[k])) continue;
                mpfr_mul(t, m, P->t[k], RN);
                mpfr_sub(w->t[k], w->t[k], t, RN);
            }
            if (L->nops == cap) {
                cap *= 2;
                L->opp = realloc(L->opp, cap * sizeof(int)), L->opi = realloc(L->opi, cap * sizeof(int));
                L->opm = realloc(L->opm, cap * sizeof(mpfr_t));
            }
            L->opp[L->nops] = p, L->opi[L->nops] = act[a];
            mpfr_init2(L->opm[L->nops], bits), mpfr_set(L->opm[L->nops], m, RN), L->nops++;
        }
    }
    free(act), free(ord), mpfr_clears(mx, m, t, (mpfr_ptr)0);
    return L;
}

char *frost_lu_solve(void *lp, const char *in) {
    lu_t *L = lp;
    int N = L->N, ne = L->ne;
    mpfr_t *b = vec(N, L->bits), *x = vec(N, L->bits), tok, t;
    mpfr_inits2(L->bits, tok, t, (mpfr_ptr)0);
    const char *q = in;
    for (int i = 0; i < N; i++) q = get(tok, q, L->bits), mpfr_mul(b[i], tok, L->scale[i], RN);
    for (int k = 0; k < L->nops; k++) {
        mpfr_mul(t, L->opm[k], b[L->opp[k]], RN);
        mpfr_sub(b[L->opi[k]], b[L->opi[k]], t, RN);
    }
    for (int col = N - 1; col >= 0; col--) {
        row_t *w = L->r + L->piv[col];
        mpfr_set(x[col], b[L->piv[col]], RN);
        if (col < N - ne)
            for (int c = col + 1; c < w->hi; c++) mpfr_mul(t, w->v[c - w->lo], x[c], RN), mpfr_sub(x[col], x[col], t, RN);
        for (int k = (col < N - ne ? 0 : col - (N - ne) + 1); k < ne; k++)
            mpfr_mul(t, w->t[k], x[N - ne + k], RN), mpfr_sub(x[col], x[col], t, RN);
        mpfr_div(x[col], x[col], at(L, w, col), RN);
    }
    buf_t out = {0};
    for (int i = 0; i < N; i++) put(&out, x[i]);
    unvec(b, N), unvec(x, N), mpfr_clears(tok, t, (mpfr_ptr)0);
    return out.s;
}
