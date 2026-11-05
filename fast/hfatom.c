/* hfatom.c: Hartree-Fock limit of closed-shell atoms in double precision, built for speed.

   Discretisation. N right Gauss-Radau nodes t_j in (0, 2] (t_N = 2 is the wall), r = a (exp(S t / 2) - 1) with
   S = log(1 + R / a), and the regularised Lagrange basis (t / t_j) l_j(t), which vanishes at r = 0. Unknowns are
   the nodal values of P = r R_nl, and the multipole potentials Y^k (Y^k / r is the potential) are the discrete
   Galerkin solutions of Y'' - k(k+1) Y / r^2 = -(2k+1) P_a P_b / r with Y(0) = 0 and R Y'(R) + k Y(R) = 0 (the
   exact exterior solution). Every integral is Radau quadrature at the nodes, so the mass matrix is diagonal and
   the energy is a polynomial functional E[P] of the nodal values.

   Solver. The orbitals are the stationary point of E under the norm constraints; one common wall makes the
   canonical orbitals of each l eigenvectors of one Fock matrix, hence orthogonal, with the eps_a the multipliers.
   A Thomas-Fermi guess and a few SCF+DIIS steps bring the orbitals into Newton's basin; Newton on the Lagrangian
   (all orbitals and eps at once, exact Hessian) then converges quadratically. Because E is stationary, rounding
   noise in the orbitals enters the energy only at second order.

   Accuracy notes (measured). Anything rounded that defines the functional (differentiation matrix, nodes, weights,
   the mapped mesh) is a first-order perturbation of the energy, while the orbitals enter only at second order. So
   the solve runs on plain double operators, and the final energy is evaluated with the exact ones: the mesh,
   weights and differentiation matrix in double-double, every product formed exactly, and a first-order correction
   for the rounding-level norm defects of the orbitals. Rounding the kinetic matrix alone moved Ne by 5e-11, and the
   double mesh left a scatter of a few ulp in Xe. The Radau nodes are found by Newton with one double-double step:
   1e-14 node errors gave 1e-12 weight errors.

   cc -O3 -mcpu=native -o hfatom hfatom.c -framework Accelerate
   ./hfatom 54 "1s 2s 2p 3s 3p 3d 4s 4p 4d 5s 5p" [--tol 1e-12] [-v]
*/
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXORB 32
#define LMAX 4

/* ---------- double-double ---------- */
typedef struct { double hi, lo; } dd;
static inline dd qsum(double a, double b) { double s = a + b; return (dd){s, b - (s - a)}; }
static inline dd tsum(double a, double b) { double s = a + b, z = s - a; return (dd){s, (a - (s - z)) + (b - z)}; }
static inline dd ddadd(dd a, dd b) { dd s = tsum(a.hi, b.hi), t = tsum(a.lo, b.lo); s.lo += t.hi; s = qsum(s.hi, s.lo); s.lo += t.lo; return qsum(s.hi, s.lo); }
static inline dd ddsub(dd a, dd b) { return ddadd(a, (dd){-b.hi, -b.lo}); }
static inline dd ddmul(dd a, dd b) { double p = a.hi * b.hi; return qsum(p, fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi)); }
static inline dd ddmuld(dd a, double b) { double p = a.hi * b; return qsum(p, fma(a.hi, b, -p) + a.lo * b); }
static inline dd dddiv(dd a, dd b) { double q = a.hi / b.hi; dd r = ddsub(a, ddmuld(b, q)); double q2 = r.hi / b.hi; r = ddsub(r, ddmuld(b, q2)); return ddadd(qsum(q, q2), (dd){r.hi / b.hi, 0}); }
static inline dd ddd(double a) { return (dd){a, 0}; }
static inline dd ddsq(double a) { return ddmul(ddd(a), ddd(a)); }     /* exact */
/* expm1 in double-double: halve to |x| < 2^-10, Taylor, then double back exactly with e -> 2e + e^2 */
static dd ddexpm1(dd x) {
    int s = 0;
    while (fabs(x.hi) > 1.0 / 1024) { x = ddmuld(x, 0.5); s++; }
    dd e = ddd(0), term = x;
    for (int k = 2; k <= 12; k++) { e = ddadd(e, term); term = dddiv(ddmul(term, x), ddd(k)); }
    while (s--) e = ddadd(ddmuld(e, 2), ddmul(e, e));
    return e;
}

/* ---------- the atom ---------- */
typedef struct {
    int Z, n, N, lmax, nn[MAXORB], l[MAXORB], nocc[LMAX + 1], occ[LMAX + 1][MAXORB];
    double a, R, S;
    double *r, *W, *v, *c;          /* N: nodes, quadrature weights, W / r, kinetic weights w / r' */
    dd *rx, *Wx, *vx, *cx, *qx;     /* N: the same in double-double, and W / r^2, for the final energy only */
    double *Dh, *Dl;                /* N^2: d/dt of the basis at the nodes, double-double (hi, lo) */
    double *h[LMAX + 1];            /* (N-1)^2, one-electron matrix of each l (wall node dropped) */
    double *G[2 * LMAX + 1];        /* N^2: (2k+1) K_k^-1, potential of a density sampled at the nodes */
    double c3[LMAX + 1][LMAX + 1][2 * LMAX + 1];   /* (l k l'; 0 0 0)^2 */
} atom;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static double threej2(int a, int b, int c) {
    int J = a + b + c, g = J / 2;
    if (J % 2 || c > a + b || c < abs(a - b)) return 0;
    double f = tgamma(J - 2 * a + 1) * tgamma(J - 2 * b + 1) * tgamma(J - 2 * c + 1) / tgamma(J + 2);
    double q = tgamma(g + 1) / (tgamma(g - a + 1) * tgamma(g - b + 1) * tgamma(g - c + 1));
    return f * q * q;
}

/* Jacobi P^(1,0)_n(x) and its derivative, in double-double; Legendre P_n(x) */
static void jacobi10(int n, dd x, dd *p, dd *dp) {
    dd p0 = ddd(1), p1 = ddmuld(ddadd(ddmuld(x, 3), ddd(1)), 0.5), d0 = ddd(0), d1 = ddd(1.5);
    if (n == 0) { *p = p0; *dp = d0; return; }
    for (int k = 2; k <= n; k++) {
        double A = (2.0 * k + 1) * (2.0 * k - 1), B = (k - 1.0) * (2.0 * k + 1), C = (k + 1.0) * (2.0 * k - 1);
        dd c = ddadd(ddmuld(x, A), ddd(1));
        dd p2 = dddiv(ddsub(ddmul(c, p1), ddmuld(p0, B)), ddd(C));
        dd d2 = dddiv(ddadd(ddsub(ddmul(c, d1), ddmuld(d0, B)), ddmuld(p1, A)), ddd(C));
        p0 = p1; p1 = p2; d0 = d1; d1 = d2;
    }
    *p = p1; *dp = d1;
}
static dd legendre(int n, dd x) {
    dd p0 = ddd(1), p1 = x;
    if (n == 0) return p0;
    for (int k = 2; k <= n; k++) {
        dd p2 = dddiv(ddsub(ddmuld(ddmul(x, p1), 2.0 * k - 1), ddmuld(p0, k - 1.0)), ddd(k));
        p0 = p1; p1 = p2;
    }
    return p1;
}

/* N right Gauss-Radau nodes as t = 1 + x (ascending, t[N-1] = 2) and weights: the interior nodes are the zeros
   of P^(1,0)_(N-1). Newton in double from cos(2 pi j / (2N - 1)), then one step with the residual in
   double-double (the root is simple, so one step squares the 1e-16 error). */
static void radau(int N, dd *t, dd *w) {
    for (int j = 1; j < N; j++) {
        double x = cos(2 * M_PI * j / (2 * N - 1)), dx = 1;
        for (int it = 0; it < 50 && fabs(dx) > 1e-15; it++) {
            double p0 = 1, p1 = (3 * x + 1) / 2, d0 = 0, d1 = 1.5;
            for (int k = 2; k < N; k++) {
                double A = (2.0 * k + 1) * (2.0 * k - 1), B = (k - 1.0) * (2.0 * k + 1), C = (k + 1.0) * (2.0 * k - 1);
                double p2 = ((A * x + 1) * p1 - B * p0) / C, d2 = ((A * x + 1) * d1 - B * d0 + A * p1) / C;
                p0 = p1; p1 = p2; d0 = d1; d1 = d2;
            }
            dx = p1 / d1;
            x -= dx;
        }
        dd X = ddd(x), p, dp;
        jacobi10(N - 1, X, &p, &dp);
        X = ddsub(X, ddd(p.hi / dp.hi));
        int i = N - 1 - j;                          /* ascending in x */
        t[i] = ddadd(X, ddd(1));
        dd q = legendre(N - 1, X);
        w[i] = dddiv(t[i], ddmuld(ddmul(q, q), (double)N * N));
    }
    t[N - 1] = ddd(2);
    w[N - 1] = dddiv(ddd(2), ddd((double)N * N));
}

static void setup(atom *A) {
    int N = A->N, M = N - 1;
    dd *t = malloc(N * sizeof(dd)), *te = malloc((N + 1) * sizeof(dd)), *lam = malloc((N + 1) * sizeof(dd));
    dd *w = malloc(N * sizeof(dd));
    double *B = malloc(sizeof(double) * N * N), *T = malloc(sizeof(double) * N * N);
    radau(N, t, w);
    /* the regularised basis is the Lagrange basis on {0} u nodes: barycentric differentiation there, rows and
       columns of the r = 0 node dropped (negative-sum diagonal over all N + 1 nodes) */
    te[0] = ddd(0);
    for (int i = 0; i < N; i++) te[i + 1] = t[i];
    for (int j = 0; j <= N; j++) {
        dd p = ddd(1);
        for (int k = 0; k <= N; k++) if (k != j) p = ddmul(p, ddmuld(ddsub(te[j], te[k]), 2));
        lam[j] = dddiv(ddd(1), p);
    }
    for (int i = 1; i <= N; i++) {
        dd s = ddd(0);
        for (int j = 0; j <= N; j++) {
            if (j == i) continue;
            dd d = dddiv(dddiv(lam[j], lam[i]), ddsub(te[i], te[j]));
            s = ddadd(s, d);
            if (j) { A->Dh[(i - 1) * N + j - 1] = d.hi; A->Dl[(i - 1) * N + j - 1] = d.lo; }
        }
        A->Dh[(i - 1) * N + i - 1] = -s.hi; A->Dl[(i - 1) * N + i - 1] = -s.lo;
    }
    /* the mesh in double-double (S as stored defines the map; the wall node is put at R exactly), rounded once
       for the solve */
    A->S = log1p(A->R / A->a);
    for (int i = 0; i < N; i++) {
        dd r = i < N - 1 ? ddmuld(ddexpm1(ddmuld(t[i], A->S / 2)), A->a) : ddd(A->R);
        dd dr = ddmuld(ddadd(r, ddd(A->a)), A->S / 2);
        A->rx[i] = r;
        A->Wx[i] = ddmul(w[i], dr);
        A->vx[i] = dddiv(A->Wx[i], r);
        A->qx[i] = dddiv(A->vx[i], r);
        A->cx[i] = dddiv(w[i], dr);         /* int P'^2 dr = sum_m w_m / r'_m (dP/dt)_m^2 */
        A->r[i] = r.hi; A->W[i] = A->Wx[i].hi; A->v[i] = A->vx[i].hi; A->c[i] = A->cx[i].hi;
    }
    /* T = D^T diag(c) D in plain double: Newton only needs an operator close to the exact one, and the energy is
       evaluated with the exact one (double-double), which leaves the error second order */
    for (int m = 0; m < N; m++)
        for (int j = 0; j < N; j++) B[m * N + j] = sqrt(A->c[m]) * A->Dh[m * N + j];
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, N, N, 1.0, B, N, 0.0, T, N);
    for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) T[j * N + i] = T[i * N + j];
    for (int k = 0; k <= 2 * A->lmax; k++) {
        double *K = A->G[k];
        memcpy(K, T, sizeof(double) * N * N);
        for (int i = 0; i < N; i++) K[i * N + i] += k * (k + 1.0) * A->W[i] / (A->r[i] * A->r[i]);
        K[N * N - 1] += k / A->R;
        int n = N, info;
        dpotrf_("L", &n, K, &n, &info);
        dpotri_("L", &n, K, &n, &info);
        if (info) { fprintf(stderr, "Poisson matrix k=%d not positive definite\n", k); exit(1); }
        for (int i = 0; i < N; i++)
            for (int j = 0; j <= i; j++) K[j * N + i] = K[i * N + j] = (2 * k + 1) * K[j * N + i];  /* lower (col-major) */
    }
    for (int L = 0; L <= A->lmax; L++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < M; j++)
                A->h[L][i * M + j] = 0.5 * T[i * N + j] + (i == j ? A->W[i] * (L * (L + 1) / (2 * A->r[i] * A->r[i]) - A->Z / A->r[i]) : 0);
    for (int x = 0; x <= LMAX; x++)
        for (int y = 0; y <= LMAX; y++)
            for (int k = 0; k <= 2 * LMAX; k++) A->c3[x][y][k] = threej2(x, k, y);
    free(t); free(te); free(lam); free(w); free(B); free(T);
}

/* Y = G_k (v P_a P_b), all N nodes */
static void potential(const atom *A, int k, const double *Pa, const double *Pb, double *Y, double *tmp) {
    int N = A->N;
    for (int i = 0; i < N; i++) tmp[i] = A->v[i] * Pa[i] * Pb[i];
    cblas_dsymv(CblasRowMajor, CblasLower, N, 1.0, A->G[k], N, tmp, 1, 0.0, Y, 1);
}

/* Fock matrix of angular momentum L (N-1)^2: h + diag(v J) - sum_b,k e (vP_b)(vP_b)^T o G_k */
static void fock(const atom *A, const double *P, int L, const double *Dd, double *F) {
    int N = A->N, M = N - 1;
    memcpy(F, A->h[L], sizeof(double) * M * M);
    for (int i = 0; i < M; i++) F[i * M + i] += A->v[i] * Dd[i];
    double *u = malloc(N * sizeof(double));
    for (int b = 0; b < A->n; b++) {
        for (int i = 0; i < N; i++) u[i] = A->v[i] * P[b * N + i];
        for (int k = abs(L - A->l[b]); k <= L + A->l[b]; k += 2) {
            double e = (2 * A->l[b] + 1) * A->c3[L][A->l[b]][k];
            const double *G = A->G[k];
            for (int i = 0; i < M; i++) {
                double ui = e * u[i];
                for (int j = 0; j < M; j++) F[i * M + j] -= ui * G[i * N + j] * u[j];
            }
        }
    }
    free(u);
}

static void direct(const atom *A, const double *P, double *Dd) {
    int N = A->N;
    double *Y = malloc(N * sizeof(double)), *tmp = malloc(N * sizeof(double));
    memset(Dd, 0, N * sizeof(double));
    for (int b = 0; b < A->n; b++) {
        potential(A, 0, P + b * N, P + b * N, Y, tmp);
        for (int i = 0; i < N; i++) Dd[i] += 2 * (2 * A->l[b] + 1) * Y[i];
    }
    free(Y); free(tmp);
}

/* lowest eigenpairs of W^-1/2 F W^-1/2 (F is (N-1)^2) into the orbitals of l = L */
static void diag_into(const atom *A, int L, const double *F, double *P, double *eps) {
    int M = A->N - 1, N = A->N;
    double *B = malloc(sizeof(double) * M * M), *ev = malloc(M * sizeof(double));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) B[i * M + j] = F[i * M + j] / sqrt(A->W[i] * A->W[j]);
    int n = M, il = 1, iu = A->nocc[L], nf, lw = -1, liw = -1, info, iw0, *isup = malloc(2 * M * sizeof(int));
    double w0, vl = 0, vu = 0, abstol = 0, *Z = malloc(sizeof(double) * M * iu);
    dsyevr_("V", "I", "L", &n, B, &n, &vl, &vu, &il, &iu, &abstol, &nf, ev, Z, &n, isup, &w0, &lw, &iw0, &liw, &info);
    lw = (int)w0; liw = iw0;
    double *work = malloc(lw * sizeof(double)); int *iwork = malloc(liw * sizeof(int));
    dsyevr_("V", "I", "L", &n, B, &n, &vl, &vu, &il, &iu, &abstol, &nf, ev, Z, &n, isup, work, &lw, iwork, &liw, &info);
    memcpy(B, Z, sizeof(double) * M * iu);
    free(Z); free(isup);
    for (int q = 0; q < A->nocc[L]; q++) {
        int a = A->occ[L][q];
        double *Pa = P + a * N, s = 0;
        for (int i = 0; i < M; i++) s += B[q * M + i] / sqrt(A->W[i]) * A->W[i] * Pa[i];
        for (int i = 0; i < M; i++) Pa[i] = (s < 0 ? -1 : 1) * B[q * M + i] / sqrt(A->W[i]);
        Pa[M] = 0;
        eps[a] = ev[q];
    }
    free(B); free(ev); free(work); free(iwork);
}

/* Thomas-Fermi (Tietz) screened potential with the Latter tail */
static void guess(const atom *A, double *P, double *eps) {
    int N = A->N, M = N - 1;
    double b = 0.8853 * pow(A->Z, -1.0 / 3), *F = malloc(sizeof(double) * M * M);
    for (int L = 0; L <= A->lmax; L++) {
        if (!A->nocc[L]) continue;
        memcpy(F, A->h[L], sizeof(double) * M * M);
        for (int i = 0; i < M; i++) {
            double x = A->r[i] / b, s = sqrt(x);
            double phi = 1 / (1 + 0.02747 * s + 1.243 * x - 0.1486 * x * s + 0.2302 * x * x + 0.007298 * x * x * s + 0.006944 * x * x * x);
            double Zeff = fmax(A->Z * phi, 1.0);
            F[i * M + i] += A->W[i] * (A->Z - Zeff) / A->r[i];
        }
        diag_into(A, L, F, P, eps);
    }
    free(F);
}

/* SCF with Pulay DIIS on the commutator W^-1/2 (F D W - W D F) W^-1/2, until its max-norm is below tol */
static int scf(const atom *A, double *P, double *eps, double tol, int maxit) {
    enum { MD = 8 };
    int N = A->N, M = N - 1, nl = A->lmax + 1, sz = nl * M * M, it;
    double *Fh = malloc(sizeof(double) * MD * sz), *Eh = malloc(sizeof(double) * MD * sz), *Dd = malloc(N * sizeof(double));
    double *Fs = malloc(sizeof(double) * sz), *C = malloc(sizeof(double) * M * M);
    int nh = 0;
    for (it = 0; it < maxit; it++) {
        direct(A, P, Dd);
        int slot = it % MD;
        double *F = Fh + slot * sz, *E = Eh + slot * sz, err = 0;
        for (int L = 0; L < nl; L++) {
            double *FL = F + L * M * M, *EL = E + L * M * M;
            if (!A->nocc[L]) { memset(FL, 0, sizeof(double) * M * M); memset(EL, 0, sizeof(double) * M * M); continue; }
            fock(A, P, L, Dd, FL);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < M; j++) FL[i * M + j] /= sqrt(A->W[i] * A->W[j]);
            /* D~ = sum_occ c c^T with c = W^1/2 P; error = F~ D~ - D~ F~ */
            memset(C, 0, sizeof(double) * M * M);
            for (int q = 0; q < A->nocc[L]; q++) {
                const double *Pa = P + A->occ[L][q] * N;
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < M; j++) C[i * M + j] += sqrt(A->W[i] * A->W[j]) * Pa[i] * Pa[j];
            }
            cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, M, M, 1.0, FL, M, C, M, 0.0, EL, M);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < i; j++) {
                    double e = EL[i * M + j] - EL[j * M + i];
                    EL[i * M + j] = e * sqrt(2 * L + 1); EL[j * M + i] = -e * sqrt(2 * L + 1);
                    if (fabs(e) > err) err = fabs(e);
                }
            for (int i = 0; i < M; i++) EL[i * M + i] = 0;
        }
        if (err < tol) break;
        nh = nh < MD ? nh + 1 : MD;
        /* DIIS: min |sum c_i e_i| with sum c_i = 1 */
        double B[(MD + 1) * (MD + 1)], rhs[MD + 1];
        int m = nh, mm = m + 1, ipiv[MD + 1], one = 1, info;
        for (int i = 0; i < m; i++) {
            int si = (it - i + MD) % MD;
            for (int j = 0; j <= i; j++) {
                int sj = (it - j + MD) % MD;
                B[i * mm + j] = B[j * mm + i] = cblas_ddot(sz, Eh + si * sz, 1, Eh + sj * sz, 1);
            }
            B[i * mm + m] = B[m * mm + i] = -1; rhs[i] = 0;
        }
        B[m * mm + m] = 0; rhs[m] = -1;
        dgesv_(&mm, &one, B, &mm, ipiv, rhs, &mm, &info);
        if (info) { memset(rhs, 0, sizeof rhs); rhs[0] = 1; }
        memset(Fs, 0, sizeof(double) * sz);
        for (int i = 0; i < m; i++) cblas_daxpy(sz, rhs[i], Fh + ((it - i + MD) % MD) * sz, 1, Fs, 1);
        for (int L = 0; L < nl; L++) {
            if (!A->nocc[L]) continue;
            double *FL = Fs + L * M * M;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < M; j++) FL[i * M + j] *= sqrt(A->W[i] * A->W[j]);
            diag_into(A, L, FL, P, eps);
        }
    }
    free(Fh); free(Eh); free(Dd); free(Fs); free(C);
    return it;
}

/* Newton on the Lagrangian: unknowns P_a at the M = N - 1 inner nodes and eps_a. Orbital rows are scaled by
   w_a = 2 l_a + 1 and norm rows by -w_a, which makes the matrix symmetric (it is the Hessian of the Lagrangian);
   Jacobi-scaled, it is factored by LAPACK. Stops after a step that changes every eps_a by less than tol
   (relatively): Newton is quadratic, so the error left is ~tol^2. Returns the iterations. */
static int newton(const atom *A, double *P, double *eps, double tol, int maxit) {
    int N = A->N, M = N - 1, n = A->n, D = n * M + n, it;
    double *H = malloc(sizeof(double) * D * D), *g = malloc(sizeof(double) * D), *s = malloc(sizeof(double) * D);
    double *Dd = malloc(N * sizeof(double)), *F = malloc(sizeof(double) * (A->lmax + 1) * M * M);
    double *Y = malloc(sizeof(double) * N), *tmp = malloc(N * sizeof(double)), *u = malloc(sizeof(double) * n * N);
    int *ipiv = malloc(D * sizeof(int));
#define HH(i, j) H[(size_t)(i) * D + (j)]
    for (it = 1; it <= maxit; it++) {
        direct(A, P, Dd);
        for (int L = 0; L <= A->lmax; L++) if (A->nocc[L]) fock(A, P, L, Dd, F + L * M * M);
        for (int a = 0; a < n; a++)
            for (int i = 0; i < N; i++) u[a * N + i] = A->v[i] * P[a * N + i];
        memset(H, 0, sizeof(double) * D * D);
        for (int a = 0; a < n; a++) {
            int la = A->l[a];
            double wa = 2 * la + 1;
            const double *FL = F + la * M * M, *Pa = P + a * N, *ua = u + a * N;
            /* gradient w_a (F P_a - eps W P_a) (F holds the exchange of every b, a included) and -w_a (|P_a|^2 - 1) / 2 */
            for (int i = 0; i < M; i++) {
                double x = 0;
                for (int j = 0; j < M; j++) x += FL[i * M + j] * Pa[j];
                g[a * M + i] = wa * (x - eps[a] * A->W[i] * Pa[i]);
            }
            double nrm = 0;
            for (int i = 0; i < M; i++) nrm += A->W[i] * Pa[i] * Pa[i];
            g[n * M + a] = -wa * 0.5 * (nrm - 1);
            for (int i = 0; i < M; i++) {
                double *row = H + (size_t)(a * M + i) * D + a * M;
                for (int j = 0; j < M; j++) row[j] = wa * FL[i * M + j];
                row[i] -= wa * eps[a] * A->W[i];
            }
            for (int b = 0; b < n; b++) {
                int lb = A->l[b];
                const double *ub = u + b * N;
                double c0 = 4.0 * wa * (2 * lb + 1);
                for (int k = abs(la - lb); k <= la + lb; k += 2) {
                    double e = wa * (2 * lb + 1) * A->c3[la][lb][k];
                    const double *G = A->G[k];
                    potential(A, k, Pa, P + b * N, Y, tmp);
                    /* local exchange and its density response; for b = a, F's own copy makes the 2 P_a dP_a */
                    for (int i = 0; i < M; i++) {
                        double *row = H + (size_t)(a * M + i) * D + b * M, x = e * ub[i];
                        const double *Gi = G + i * N;
                        if (k == 0) { double y = c0 * ua[i]; for (int j = 0; j < M; j++) row[j] += (y * ub[j] - x * ua[j]) * Gi[j]; }
                        else for (int j = 0; j < M; j++) row[j] -= x * Gi[j] * ua[j];
                        row[i] -= e * A->v[i] * Y[i];
                    }
                }
                if (abs(la - lb) % 2 || la != lb)   /* no k = 0 term: Coulomb response alone */
                    for (int i = 0; i < M; i++) {
                        double *row = H + (size_t)(a * M + i) * D + b * M, y = c0 * ua[i];
                        const double *Gi = A->G[0] + i * N;
                        for (int j = 0; j < M; j++) row[j] += y * Gi[j] * ub[j];
                    }
            }
            for (int i = 0; i < M; i++) HH(a * M + i, n * M + a) = HH(n * M + a, a * M + i) = -wa * A->W[i] * Pa[i];
        }
        for (int i = 0; i < D; i++) { double d = fabs(HH(i, i)); s[i] = d > 0 ? 1 / sqrt(d) : 1; }
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) HH(i, j) *= s[i] * s[j];
        for (int i = 0; i < D; i++) g[i] = -g[i] * s[i];
        int one = 1, info, DD_ = D;
        if (getenv("SYSV")) {
            int lw = -1; double w0;
            dsysv_("U", &DD_, &one, H, &DD_, ipiv, g, &DD_, &w0, &lw, &info);
            lw = (int)w0; double *wk = malloc(lw * sizeof(double));
            dsysv_("U", &DD_, &one, H, &DD_, ipiv, g, &DD_, wk, &lw, &info);
            free(wk);
        } else dgesv_(&DD_, &one, H, &DD_, ipiv, g, &DD_, &info);
        if (info) { fprintf(stderr, "singular Newton matrix\n"); break; }
        double st = 0;
        for (int a = 0; a < n; a++) {
            for (int i = 0; i < M; i++) P[a * N + i] += s[a * M + i] * g[a * M + i];
            double de = s[n * M + a] * g[n * M + a];
            eps[a] += de;
            if (fabs(de / eps[a]) > st) st = fabs(de / eps[a]);
        }
        if (st < tol) break;
    }
    free(H); free(g); free(s); free(Dd); free(F); free(Y); free(tmp); free(u); free(ipiv);
    return it;
}

/* (D y)_m in double-double, D = Dh + Dl */
static void ddmv(const atom *A, const double *y, dd *out) {
    int N = A->N;
    for (int m = 0; m < N; m++) {
        const double *dh = A->Dh + m * N, *dl = A->Dl + m * N;
        double hi = 0, lo = 0;
        for (int j = 0; j < N; j++) {
            double p = dh[j] * y[j], e = fma(dh[j], y[j], -p), s = hi + p, z = s - hi;
            lo += ((hi - (s - z)) + (p - z)) + e + dl[j] * y[j];
            hi = s;
        }
        out[m] = qsum(hi, lo);
    }
}

/* Y^T K_k Y with the exact (double-double) operator */
static dd quadK(const atom *A, int k, const double *Y, dd *tmp) {
    int N = A->N;
    ddmv(A, Y, tmp);
    dd s = ddd(0);
    for (int m = 0; m < N; m++) s = ddadd(s, ddmul(ddmul(tmp[m], tmp[m]), A->cx[m]));
    if (!k) return s;
    for (int i = 0; i < N; i++) s = ddadd(s, ddmuld(ddmul(A->qx[i], ddsq(Y[i])), k * (k + 1.0)));
    return ddadd(s, dddiv(ddmuld(ddsq(Y[N - 1]), k), ddd(A->R)));
}

/* B_k(rho, rho) = rho^T G_k rho, variationally: 2 rho^T Y - Y^T K_k Y / (2k+1) with Y = G_k rho, so the error of
   the double-precision G_k enters at second order; rho is exact (double-double), rh is scratch */
static dd selfB(const atom *A, int k, const dd *rho, double *rh, double *Y, dd *tmp) {
    int N = A->N;
    for (int i = 0; i < N; i++) rh[i] = rho[i].hi;
    cblas_dsymv(CblasRowMajor, CblasLower, N, 1.0, A->G[k], N, rh, 1, 0.0, Y, 1);
    dd s = ddd(0);
    for (int i = 0; i < N; i++) s = ddadd(s, ddmuld(rho[i], 2 * Y[i]));
    return ddsub(s, dddiv(quadK(A, k, Y, tmp), ddd(2 * k + 1)));
}

/* E = sum_a 2 w_a h_aa + 2 B_0(rho, rho) - sum_ab w_a w_b sum_k c^k B_k(rho_ab, rho_ab), rho = sum w_a v P_a^2,
   with the exact operators and mesh in double-double and every product exact: P is the stationary point of the
   double-precision functional, so this energy differs from the exact discrete one only at second order. The
   orbitals are orthonormal to rounding, and that defect is first order, so it is corrected with the norms in
   double-double: dE = occ_a eps_a (<a|a> - 1). Also the virial ratio -V/T. */
static double energy(const atom *A, const double *P, const double *eps, double *virial) {
    int N = A->N, n = A->n;
    double *Y = malloc(N * sizeof(double)), *rh = malloc(N * sizeof(double));
    dd *tmp = malloc(N * sizeof(dd)), *rho = calloc(N, sizeof(dd)), E = ddd(0), Tk = ddd(0);
    for (int a = 0; a < n; a++) {
        const double *Pa = P + a * N;
        int L = A->l[a];
        double wa = 2.0 * (2 * L + 1);
        ddmv(A, Pa, tmp);
        dd t = ddd(0), u = ddd(0), nrm = ddd(0);
        for (int m = 0; m < N; m++) t = ddadd(t, ddmul(ddmul(tmp[m], tmp[m]), A->cx[m]));
        t = ddmuld(t, 0.5);
        for (int i = 0; i < N; i++) {
            dd p2 = ddsq(Pa[i]), vp = ddmul(A->vx[i], p2);
            t = ddadd(t, ddmuld(ddmul(A->qx[i], p2), L * (L + 1) / 2.0));
            u = ddsub(u, ddmuld(vp, A->Z));
            rho[i] = ddadd(rho[i], ddmuld(vp, 2 * L + 1));
            nrm = ddadd(nrm, ddmul(A->Wx[i], p2));
        }
        E = ddadd(E, ddmuld(ddadd(t, u), wa));
        Tk = ddadd(Tk, ddmuld(t, wa));
        E = ddsub(E, ddmuld(ddsub(nrm, ddd(1)), wa * eps[a]));
    }
    E = ddadd(E, ddmuld(selfB(A, 0, rho, rh, Y, tmp), 2));
    for (int a = 0; a < n; a++)
        for (int b = a; b < n; b++) {
            int la = A->l[a], lb = A->l[b];
            double w = (2 * la + 1) * (2 * lb + 1) * (a == b ? 1 : 2);
            for (int i = 0; i < N; i++) rho[i] = ddmul(A->vx[i], ddmul(ddd(P[a * N + i]), ddd(P[b * N + i])));
            for (int k = abs(la - lb); k <= la + lb; k += 2)
                E = ddsub(E, ddmuld(selfB(A, k, rho, rh, Y, tmp), w * A->c3[la][lb][k]));
        }
    free(Y); free(rh); free(rho); free(tmp);
    *virial = -(E.hi - Tk.hi) / Tk.hi;
    return E.hi + E.lo;
}

static const char *SYM[] = {"", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"};
/* closed-shell neutral ground states (HF): Z -> occupied subshells */
static const char *closed(int Z) {
    static const struct { int Z; const char *o; } t[] = {
        {2, "1s"}, {4, "1s 2s"}, {10, "1s 2s 2p"}, {12, "1s 2s 2p 3s"}, {18, "1s 2s 2p 3s 3p"}, {20, "1s 2s 2p 3s 3p 4s"},
        {30, "1s 2s 2p 3s 3p 3d 4s"}, {36, "1s 2s 2p 3s 3p 3d 4s 4p"}, {38, "1s 2s 2p 3s 3p 3d 4s 4p 5s"},
        {46, "1s 2s 2p 3s 3p 3d 4s 4p 4d"}, {48, "1s 2s 2p 3s 3p 3d 4s 4p 4d 5s"},
        {54, "1s 2s 2p 3s 3p 3d 4s 4p 4d 5s 5p"}, {56, "1s 2s 2p 3s 3p 3d 4s 4p 4d 5s 5p 6s"},
        {70, "1s 2s 2p 3s 3p 3d 4s 4p 4d 4f 5s 5p 6s"}, {80, "1s 2s 2p 3s 3p 3d 4s 4p 4d 4f 5s 5p 5d 6s"},
        {86, "1s 2s 2p 3s 3p 3d 4s 4p 4d 4f 5s 5p 5d 6s 6p"}};
    for (unsigned i = 0; i < sizeof t / sizeof *t; i++) if (t[i].Z == Z) return t[i].o;
    return NULL;
}

static void atom_init(atom *A, int Z, const char *orbitals, int N, double a, double R) {
    memset(A, 0, sizeof *A);
    A->Z = Z; A->N = N; A->a = a; A->R = R;
    char buf[512]; strncpy(buf, orbitals, sizeof buf - 1);
    for (char *tok = strtok(buf, " "); tok; tok = strtok(NULL, " ")) {
        int nn = atoi(tok), L = strchr("spdfg", tok[strlen(tok) - 1]) - "spdfg";
        A->nn[A->n] = nn; A->l[A->n] = L;
        A->occ[L][A->nocc[L]++] = A->n;     /* listed in increasing n within each l */
        if (L > A->lmax) A->lmax = L;
        A->n++;
    }
    A->r = malloc(N * sizeof(double)); A->W = malloc(N * sizeof(double)); A->v = malloc(N * sizeof(double));
    A->c = malloc(N * sizeof(double)); A->Dh = malloc(sizeof(double) * N * N); A->Dl = malloc(sizeof(double) * N * N);
    A->rx = malloc(N * sizeof(dd)); A->Wx = malloc(N * sizeof(dd)); A->vx = malloc(N * sizeof(dd));
    A->cx = malloc(N * sizeof(dd)); A->qx = malloc(N * sizeof(dd));
    for (int L = 0; L <= A->lmax; L++) A->h[L] = malloc(sizeof(double) * (N - 1) * (N - 1));
    for (int k = 0; k <= 2 * A->lmax; k++) A->G[k] = malloc(sizeof(double) * N * N);
}

/* Loewdin-orthonormalise the orbitals of each l in the W metric. The energy formula assumes orthonormal orbitals; a
   Newton step meets the norm constraint only to first order and leaves same-l overlaps of the order of its square,
   which would enter the energy at first order. Afterwards E is the determinant's energy, second order in the error. */
static void orthonormalize(const atom *A, double *P) {
    int N = A->N;
    for (int L = 0; L <= A->lmax; L++) {
        int m = A->nocc[L];
        if (!m) continue;
        double S[MAXORB * MAXORB], ev[MAXORB], X[MAXORB * MAXORB], Q[MAXORB * MAXORB] = {0};
        for (int p = 0; p < m; p++)
            for (int q = 0; q < m; q++) {
                const double *a = P + A->occ[L][p] * N, *b = P + A->occ[L][q] * N;
                double x = 0;
                for (int i = 0; i < N; i++) x += A->W[i] * a[i] * b[i];
                S[p * m + q] = x;
            }
        int n = m, lw = 64 * MAXORB, info; double wk[64 * MAXORB];
        dsyev_("V", "L", &n, S, &n, ev, wk, &lw, &info);       /* S = V diag(ev) V^T, V in columns of S */
        for (int p = 0; p < m; p++)
            for (int q = 0; q < m; q++) {
                double x = 0;
                for (int k = 0; k < m; k++) x += S[k * m + p] * S[k * m + q] / sqrt(ev[k]);
                Q[p * m + q] = x;                               /* S^-1/2 */
            }
        for (int i = 0; i < N; i++) {
            for (int q = 0; q < m; q++) {
                double x = 0;
                for (int p = 0; p < m; p++) x += P[A->occ[L][p] * N + i] * Q[p * m + q];
                X[q] = x;
            }
            for (int q = 0; q < m; q++) P[A->occ[L][q] * N + i] = X[q];
        }
    }
}

/* One solve on the grid (N, a, R): returns the energy, fills eps; *iters = SCF and Newton steps */
static double solve(atom *A, double *eps, double *virial, int iters[2], double tm[4]) {
    double t0 = now();
    setup(A);
    double *P = calloc((size_t)A->n * A->N, sizeof(double)), t1 = now();
    guess(A, P, eps);
    iters[0] = scf(A, P, eps, 1e-6, 200);
    double t2 = now();
    iters[1] = newton(A, P, eps, 1e-7, 30);   /* one step from SCF(1e-6) left Zn 2 ulp off; the second puts it at the floor */
    orthonormalize(A, P);
    double t3 = now(), E = energy(A, P, eps, virial);
    tm[0] = t1 - t0; tm[1] = t2 - t1; tm[2] = t3 - t2; tm[3] = now() - t3;
    free(P);
    return E;
}

static void atom_free(atom *A) {
    free(A->r); free(A->W); free(A->v); free(A->c); free(A->Dh); free(A->Dl);
    free(A->rx); free(A->Wx); free(A->vx); free(A->cx); free(A->qx);
    for (int L = 0; L <= A->lmax; L++) free(A->h[L]);
    for (int k = 0; k <= 2 * A->lmax; k++) free(A->G[k]);
}

/* Library entry point. orbitals NULL: the neutral closed-shell ground state; N, a, R <= 0: automatic.
   out[0] E, [1] -V/T - 2, [2] N, [3] a, [4] R, [5] SCF steps, [6] Newton steps, [7] seconds, [8 ...] eps in the
   order of the orbital list (at most 32). Returns the number of orbitals, or -1 for bad input. */
const char *hfatom_config(int Z) { return Z > 0 ? closed(Z) : NULL; }
const char *hfatom_symbol(int Z) { return Z > 0 && Z <= 86 ? SYM[Z] : NULL; }

int hfatom_run(int Z, const char *orbitals, double tol, int N, double a, double R, double *out) {
    const char *orb = orbitals ? orbitals : closed(Z);
    if (Z < 1 || Z > 120 || !orb || !(tol > 0)) return -1;
    double T0 = now();
    /* grid: a = 3/Z; N from the calibration on He-Xe against the certified limits (N = 10 ln Z + 2 digits + 3);
       the wall from E_R - E_HF ~ 3e4 exp(-2 kappa R), checked after the solve with kappa of the highest orbital */
    double d = fmin(-log10(tol), 15);
    if (!(a > 0)) a = 3.0 / Z;
    int autoN = !(N > 0), autoR = !(R > 0);
    if (autoN) N = (int)ceil(10 * log(Z) + 2 * d + 3 + fmax(0, Z - 54) / 2.5);   /* beyond Xe: Hg needed 80 */
    if (autoR) R = 40;
    atom A;
    double eps[MAXORB], vir, tm[4], E;
    int it[2];
    for (;;) {
        atom_init(&A, Z, orb, N, a, R);
        if (A.n < 1 || A.n > MAXORB || A.lmax > LMAX) { atom_free(&A); return -1; }
        E = solve(&A, eps, &vir, it, tm);
        atom_free(&A);
        double emax = -1e300;
        for (int i = 0; i < A.n; i++) emax = fmax(emax, eps[i]);
        double need = (log(3e4) + d * log(10)) / (2 * sqrt(-2 * emax));
        if (autoR && emax < 0 && need > R) { R = ceil(need); continue; }
        /* the virial theorem is exact for the HF limit and holds to ~1e-14 on a resolved grid; an under-resolved grid
           fails abruptly (Hg at N = 71: a wrong solution, -V/T - 2 = 1e-4), so it is the canary for a larger N */
        if (autoN && !(fabs(vir - 2) < 1e-9) && N < 400) { N = (int)ceil(1.15 * N); continue; }
        break;
        /* ponytail: each retry rebuilds from scratch; interpolate the old solution if retries ever become common */
    }
    double o[8] = {E, vir - 2, N, a, R, it[0], it[1], now() - T0};
    memcpy(out, o, sizeof o);
    memcpy(out + 8, eps, A.n * sizeof(double));
    return A.n;
}

#ifndef HFATOM_LIB
int main(int argc, char **argv) {
    const char *usage = "usage: hfatom Z|symbol [\"1s 2s 2p ...\"] [--tol 1e-12] [--N n --a a --R R] [-v]\n"
        "  closed-shell restricted HF limit; orbitals default to the neutral closed-shell ground state\n";
    int Z = 0, N = 0, verbose = 0;
    double tol = 1e-12, a = 0, R = 0, out[8 + MAXORB];
    const char *orb = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--a") && i + 1 < argc) a = atof(argv[++i]);
        else if (!strcmp(argv[i], "--R") && i + 1 < argc) R = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!Z) {
            Z = atoi(argv[i]);
            for (int z = 1; !Z && z <= 86; z++) if (!strcmp(argv[i], SYM[z])) Z = z;
        } else orb = argv[i];
    }
    if (!orb && Z) orb = closed(Z);
    int n = hfatom_run(Z, orb, tol, N, a, R, out);
    if (n < 0) { fputs(usage, stderr); return 1; }
    char buf[512]; strncpy(buf, orb, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    printf("%s  Z=%d  %s\n", Z <= 86 ? SYM[Z] : "", Z, orb);
    printf("E      = %.17g\n", out[0]);
    printf("virial = %.3e  (-V/T - 2)\n", out[1]);
    int i = 0;
    for (char *tok = strtok(buf, " "); tok; tok = strtok(NULL, " ")) printf("eps %-3s = %.13f\n", tok, out[8 + i++]);
    if (verbose) printf("N=%d a=%.4g R=%g  scf=%d newton=%d\n", (int)out[2], out[3], out[4], (int)out[5], (int)out[6]);
    printf("time   = %.2f ms\n", 1e3 * out[7]);
    return 0;
}
#endif
