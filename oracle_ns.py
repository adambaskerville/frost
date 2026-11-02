"""Hartree-Fock limit of closed-shell atoms by Taylor-series multiple shooting: any set of closed subshells nl^(2(2l+1)).
"1s" is the He-like problem of oracle.py, "1s 2s" the Be-like 1s^2 2s^2, "1s 2s 2p" the Ne-like 1s^2 2s^2 2p^6.

Orbital a (angular momentum l_a) has radial function R_a = r^l_a f_a (phi_am = R_a Y_lm, P_a = r^(l_a+1) f_a). The
pair (a, b) has multipole potentials Y^k_ab(r) = r int P_a P_b(s) r_<^k / r_>^(k+1) ds for k = |l_a - l_b|, ...,
l_a + l_b in steps of 2, kept as y^k_ab = Y^k_ab / r^(k+1). In these variables every unknown is analytic at r = 0
and obeys r u'' + 2 (L + 1) u' = rhs, so one Frobenius/Taylor recurrence serves all. The canonical Fock equations:

    r f_a'' + 2 (l_a + 1) f_a' = 2 [(r (2 sum_b (2 l_b + 1) y^0_bb - y^0_aa) - Z - eps_a r) f_a
                                   - sum_{(b,k) != (a,0)} e^k_ab r^(k + 1 + l_b - l_a) y^k_ab f_b]
    r y^k_ab'' + 2 (k + 1) y^k_ab' = -(2k + 1) r^(l_a + l_b + 1 - k) f_a f_b,      e^k_ab = (2 l_b + 1) (l_a k l_b; 0 0 0)^2

All orbitals share one wall R: f_a(R) = 0, and beyond the density R y' + (2k + 1) y = 0 holds exactly, so
R y^0_aa(R) = int P_a^2 is the norm. With one wall the canonical orbitals are eigenfunctions of one self-adjoint
Fock operator, hence orthogonal, and the walled problem is the exact HF minimiser on [0, R]: E_R - E_HF ~
exp(-2 kappa R) with kappa the outermost orbital's. A common wall is also necessary: exchange with k >= 1 gives
every canonical orbital the outermost one's exp(-kappa r) tail (Handy, Marron & Silverstone 1969), which a
wall at an inner orbital's own exp(-2 kappa_i R_i) would cut.

Multiple shooting. Inner orbitals are stiff on [0, R]: their own mode exp(kappa_1 r) grows by exp(kappa_1 R), which
single shooting can only absorb with ~kappa_1 / kappa extra precision and an eigenvalue resolved to
exp(-(kappa_1 + kappa) R). So [0, R] is cut into segments over which exp(kappa_1 L) <= 10^g. The unknowns are
f_a(0) and y(0) (the Frobenius start), the full state (f, f', y, y') at every interior knot, and the eps_a; the
equations are continuity at the knots and, at R, f_a = 0, the far condition, and the norms. Its Jacobian is
block-bidiagonal plus the eps columns, factored by sparse elimination with partial pivoting (fill stays banded)
and reused across Newton iterations (chord). Working precision is ~ target + 2g, whatever kappa_1 / kappa.
The start is a homotopy in the electron-electron coupling t from hydrogenic orbitals, exact at t = 0.

    python oracle_ns.py --Z 4 --digits 100 --out be100.json
    python oracle_ns.py --Z 10 --orbitals "1s 2s 2p" --digits 50 --out ne50.json
"""
import argparse
import contextlib
import ctypes
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from math import comb, factorial
from types import SimpleNamespace

import gmpy2
from gmpy2 import mpfr, mpq

from oracle import LN10, dot, slope, value


def threej2(a, b, c):
    """(a b c; 0 0 0)^2, exactly."""
    J = a + b + c
    if J % 2 or c > a + b or c < abs(a - b):
        return mpq(0)
    g = J // 2
    return mpq(factorial(J - 2 * a) * factorial(J - 2 * b) * factorial(J - 2 * c), factorial(J + 1)) * mpq(
        factorial(g), factorial(g - a) * factorial(g - b) * factorial(g - c)) ** 2


def system(Z, orbitals):
    """Index sets for the orbitals ("1s 2s 2p"): pairs, multipole potentials (q, k), the constants that are
    unknowns (nu), the exchange terms of each orbital's equation, and the energy integrals."""
    nl = [(int(o[:-1]), "spdfg".index(o[-1])) for o in orbitals.split()]
    l, n = [x[1] for x in nl], len(nl)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    lq = {q: l[q[0]] + l[q[1]] for q in pairs}
    K = {q: list(range(abs(l[q[0]] - l[q[1]]), lq[q] + 1, 2)) for q in pairs}
    pots = [(q, k) for q in pairs for k in K[q]]
    nus = [(q, k) for q, k in pots if not (q[0] == q[1] and k == 0)]
    key = lambda i, j: (min(i, j), max(i, j))  # noqa: E731
    # e = (2 l_b + 1) (l_a k l_b; 0 0 0)^2 stays an exact rational: an mpfr made here would freeze the precision of
    # the first (low-precision homotopy) call into every later stage
    terms = [[(b, k + 1 + l[b] - l[a], (2 * l[b] + 1) * threej2(l[a], k, l[b]), (key(a, b), k))
              for b in range(n) for k in K[key(a, b)] if (b, k) != (a, 0)] for a in range(n)]
    same = [q for q in pairs if l[q[0]] == l[q[1]]]
    ltype = lambda q: sorted((l[q[0]], l[q[1]]))  # noqa: E731
    J = sorted({(q, p, 0) for q in same for p in same}
               | {(q, p, k) for q in pairs for p in pairs if ltype(q) == ltype(p) for k in K[q]})
    powers = ({1} | {lq[q] + 1 - k for q, k in pots} | {m for t in terms for _, m, _, _ in t} | {2 * x + 2 for x in l}
              | {2 * x + 1 for x in l} | {lq[q] + k + 2 for q, k in pots})
    return SimpleNamespace(Z=Z, orbitals=orbitals, nl=nl, l=l, n=n, pairs=pairs, lq=lq, K=K, pots=pots, nus=nus, terms=terms, same=same,
                           J=J, powers=powers, key=key)


def series(r0, st, mu, nu, sy, p, active, t, energy=False):
    """Order-p Taylor coefficients about r0 of the f_a and y~^k_q, and if energy of T_q = 1/2 int r^(2l+2) f_i' f_j',
    V_q = -Z int r^(2l+1) f_i f_j (same-l pairs q), M^k_q = int r^(l_q+k+2) f_i f_j and
    J^k_(q,p) = int r^(l_q+k+2) f_i f_j y~^k_p. st = state at r0; inactive orbitals (beyond their wall) are zero."""
    n, Z, l = sy.n, sy.Z, sy.l
    f = [None if not active[i] else [st["f"][i][0]] if r0 == 0 else list(st["f"][i]) for i in range(n)]
    y = {pk: [st["y"][pk][0]] if r0 == 0 else list(st["y"][pk]) for pk in sy.pots}
    E = {key: {q: [v] for q, v in st[key].items()} for key in ("T", "V", "M", "J")}
    Rm = {m: [comb(m, j) * r0 ** (m - j) for j in range(m + 1)] for m in sy.powers}  # (r0 + x)^m

    def rmul(m, g, k):
        """(r^m g)_k about r0; m <= 1 and r0 = 0 are the hot cases."""
        if r0 == 0:
            return g[k - m] if k >= m else 0
        if m == 1:
            return r0 * g[k] + g[k - 1] if k else r0 * g[0]
        if m == 0:
            return g[k]
        c = Rm[m]
        return sum(c[j] * g[k - j] for j in range(min(m, k) + 1))
    ff, W, D, rD = {q: [] for q in sy.pairs}, {pk: [] for pk in sy.nus}, [[] for _ in range(n)], [[] for _ in range(n)]
    prod = [[[] for _ in sy.terms[a]] for a in range(n)]
    df, dd, fy = [[] for _ in range(n)], {q: [] for q in sy.same}, {x: [] for x in sy.J}

    def step(u, g, L, k):
        """Next coefficient of r u'' + 2 (L + 1) u' = g (Frobenius at r0 = 0)."""
        m = (k + 1) * (k + 2 * L + 2)
        return g / m if r0 == 0 else (g - m * u[k + 1]) / (r0 * (k + 1) * (k + 2))

    for k in range(p):
        for q in sy.pairs:
            fi, fj = f[q[0]], f[q[1]]
            ff[q].append(dot(fi[: k + 1], fj[: k + 1]) if fi and fj else mpfr(0))
        for q, kk in sy.pots:
            y[(q, kk)].append(step(y[(q, kk)], -(2 * kk + 1) * rmul(sy.lq[q] + 1 - kk, ff[q], k), kk, k))
        for pk in sy.nus:
            W[pk].append(t * y[pk][k] + (nu[pk] if k == 0 else 0))
        new = []
        for a in range(n):
            D[a].append(t * (2 * sum((2 * l[b] + 1) * y[((b, b), 0)][k] for b in range(n)) - y[((a, a), 0)][k]))
            rD[a].append(rmul(1, D[a], k))
            if not f[a]:
                new.append(None)
                continue
            g = dot(rD[a], f[a][: k + 1]) - Z * f[a][k] - mu[a] * rmul(1, f[a], k)
            for x, (b, m, e, pk) in enumerate(sy.terms[a]):
                prod[a][x].append(dot(W[pk], f[b][: k + 1]) if f[b] else mpfr(0))
                g -= e * rmul(m, prod[a][x], k)
            new.append(step(f[a], 2 * g, l[a], k))
        for a in range(n):
            if f[a]:
                f[a].append(new[a])
                if energy:
                    df[a].append((k + 1) * f[a][k + 1])
        if energy:
            for q in sy.same:
                both = f[q[0]] and f[q[1]]
                dd[q].append(dot(df[q[0]], df[q[1]]) if both else mpfr(0))
                E["T"][q].append(rmul(2 * l[q[0]] + 2, dd[q], k) / (2 * (k + 1)))
                E["V"][q].append(-Z * rmul(2 * l[q[0]] + 1, ff[q], k) / (k + 1))
            for q, kk in sy.pots:
                E["M"][(q, kk)].append(rmul(sy.lq[q] + kk + 2, ff[q], k) / (k + 1))
            for x in sy.J:
                q, pq, kk = x
                fy[x].append(dot(ff[q], y[(pq, kk)][: k + 1]))
                E["J"][x].append(rmul(sy.lq[q] + kk + 2, fy[x], k) / (k + 1))
    return f, y, E


def step_size(fs, c, r0, R, cap=math.inf):
    """c times the root-test radius of convergence (min over orbitals), on a 2^-16 grid, at most cap grid units. The
    scale is max(|f_0|, |f_1|) so that an expansion point at the 2s node does not fake a tiny radius."""
    rho = min(
        (max(abs(f[0]), abs(f[1])) / abs(f[k])) ** (mpfr(1) / k)
        for f in fs if f for k in range(len(f) - 9, len(f)) if f[k]
    )
    h = mpfr(min(math.floor(float(c * rho) * 65536), cap)) / 65536
    return min(max(h, mpfr(1) / 65536), R - r0)


def stiff_cap(eps, c, p):
    """The root test sees only the solution, which far out is every orbital riding the outermost one's slow tail. The
    modes exp(+-kappa r), kappa = sqrt(-2 eps), that Newton steps and Jacobian columns are made of are entire, with
    order-p root-test radius p / (e kappa): cap the step at c times that, in 2^-16 units (without it a whole segment
    near the wall is one step and the Jacobian is good to ~15 digits)."""
    m = float(min(eps))
    return math.floor(float(c) * p / (math.e * math.sqrt(-2 * m)) * 65536) if m < 0 else math.inf


def sweep(st, r0, r1, eps, sy, p, c, t, mesh=None, energy=False):
    """Integrate one segment r0 -> r1 from st = {"f": [(f, f')], "y": {pk: (y, y')}} (at r0 = 0 only the values
    are used: the Frobenius start). Returns (end state with the segment's energy integrals, mesh, elements)."""
    z = mpfr(0)
    acc = ({"T": {q: z for q in sy.same}, "V": {q: z for q in sy.same}, "M": {pk: z for pk in sy.pots},
            "J": {x: z for x in sy.J}} if energy else {"T": {}, "V": {}, "M": {}, "J": {}})
    st, nu, on, hs, elems = {**st, **acc}, {pk: z for pk in sy.nus}, [True] * sy.n, [], []
    cap = stiff_cap(eps, c, p)
    while r0 < r1:
        f, y, E = series(r0, st, eps, nu, sy, p, on, t, energy)  # full y: no constants to absorb, nu = 0
        h = step_size(f, c, r0, r1, cap) if mesh is None else mesh[len(hs)]
        hs.append(h)
        elems.append((r0, h, f, y))
        st = {"f": [(value(u, h), slope(u, h)) for u in f], "y": {pk: (value(u, h), slope(u, h)) for pk, u in y.items()},
              **{key: {q: value(u, h) for q, u in E[key].items()} for key in E}}
        r0 += h
    return st, hs, elems


def _kernel():
    """taylor.c: sweep() in C, operation for operation (bit-identical). Compiled against the env's MPFR when missing or
    older than its source, so a stale binary is never loaded; None (Python sweep) if that fails."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taylor.c")
    lib = src[:-2] + ".so"
    try:
        if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
            inc, libdir = os.path.join(sys.prefix, "include"), os.path.join(sys.prefix, "lib")
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", src, "-o", f"{lib}.{os.getpid()}", f"-I{inc}", f"-L{libdir}",
                            "-lmpfr", "-lgmp", f"-Wl,-rpath,{libdir}"], check=True)
            os.replace(f"{lib}.{os.getpid()}", lib)
        k = ctypes.CDLL(lib)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"taylor.c unavailable ({e}); using the Python sweep", file=sys.stderr)
        return None
    P, I = ctypes.POINTER(ctypes.c_int), ctypes.c_int
    k.frost_sys.restype, k.frost_sweep.restype = ctypes.c_void_p, ctypes.c_void_p
    k.frost_sys.argtypes = [I, ctypes.c_long, P, I, P, P, I, P, P, I, P, P, P, P, ctypes.c_char_p, I, P, I, P, P, P]
    k.frost_sweep.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_char_p, I, I, I]
    k.frost_free.argtypes = [ctypes.c_void_p]
    k.frost_lu.restype, k.frost_lu_solve.restype = ctypes.c_void_p, ctypes.c_void_p
    k.frost_lu.argtypes = [ctypes.c_long, I, I, ctypes.c_char_p]
    k.frost_lu_solve.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    k.frost_lu_free.argtypes = [ctypes.c_void_p]
    return k


_K = _kernel()


def _hex(v):
    """v exactly, as the hex token "mantissa p exponent" (powers of 2) that taylor.c reads and writes."""
    v = v if isinstance(v, gmpy2.mpfr) else mpfr(v)
    if not gmpy2.is_finite(v):
        return "@NaN@" if gmpy2.is_nan(v) else "-@Inf@" if v < 0 else "@Inf@"
    return "{:x}p{}".format(*v.as_mantissa_exp())


def csweep(st, r0, r1, eps, sy, p, c, t, mesh, mode):
    """sweep() in C: (end state with the segment's energy integrals, mesh, elements (r0, h, f) if mode is
    "elements", else None)."""
    if not hasattr(sy, "c"):
        pr, ix = {q: i for i, q in enumerate(sy.pairs)}, {pk: i for i, pk in enumerate(sy.pots)}
        T = [(a, b, m, e, ix[pk]) for a in range(sy.n) for b, m, e, pk in sy.terms[a]]
        arr = lambda xs: (ctypes.c_int * max(len(xs), 1))(*xs)  # noqa: E731
        sy.c = _K.frost_sys(sy.n, sy.Z, arr(sy.l), len(sy.pairs), arr([i for i, _ in sy.pairs]), arr([j for _, j in sy.pairs]),
                            len(sy.pots), arr([pr[q] for q, _ in sy.pots]), arr([k for _, k in sy.pots]), len(T),
                            *(arr([x[i] for x in T]) for i in (0, 1, 2, 4)),
                            " ".join(f"{x[3].numerator}/{x[3].denominator}" for x in T).encode(), len(sy.same),
                            arr([pr[q] for q in sy.same]), len(sy.J), arr([pr[q] for q, _, _ in sy.J]),
                            arr([ix[(pp, k)] for _, pp, k in sy.J]), arr([k for _, _, k in sy.J]))
    bits, mesh = gmpy2.get_context().precision, mesh or []
    xs = [r0, r1, c, t, *eps] + [v for fa in st["f"] for v in fa] + [v for pk in sy.pots for v in st["y"][pk]] + mesh
    ptr = _K.frost_sweep(sy.c, bits, " ".join(map(_hex, xs)).encode(), p, len(mesh),
                         {"": 0, "energy": 1, "elements": 2}[mode])
    out, hs, el = ctypes.string_at(ptr).decode().split("|")
    _K.frost_free(ptr)
    v = iter([mpfr(s, bits, 16) for s in out.split()])
    n = sy.n
    st = {"f": [(next(v), next(v)) for _ in range(n)], "y": {pk: (next(v), next(v)) for pk in sy.pots}}
    keys = (("T", sy.same), ("V", sy.same), ("M", sy.pots), ("J", sy.J))
    st.update({key: ({q: next(v) for q in qs} if mode else {}) for key, qs in keys})
    els, e = [], iter([mpfr(s, bits, 16) for s in el.split()])
    for u in e:
        h, ln = next(e), p + (1 if u == 0 else 2)
        els.append((u, h, [[next(e) for _ in range(ln)] for _ in range(n)]))
    return st, [mpfr(s, bits, 16) for s in hs.split()], els if mode == "elements" else None


def _out(st, sy, R=None):
    """A segment's contribution to the residual (before subtracting the next knot's state); R: the last segment's,
    at the wall: f_a(R), the far conditions and the norms."""
    if R is None:
        return pack(st, sy)
    return ([v for v, _ in st["f"]] + [R * st["y"][pk][1] + (2 * pk[1] + 1) * st["y"][pk][0] for pk in sy.pots]
            + [R * st["y"][((a, a), 0)][0] - 1 for a in range(sy.n)])


def pack(st, sy):
    return [v for fa in st["f"] for v in fa] + [v for pk in sy.pots for v in st["y"][pk]]


def unpack(x, sy):
    n = sy.n
    return {"f": [(x[2 * a], x[2 * a + 1]) for a in range(n)],
            "y": {pk: (x[2 * n + 2 * i], x[2 * n + 2 * i + 1]) for i, pk in enumerate(sy.pots)}}


class _CLU:
    """An LU held by taylor.c (frost_lu), freed with this object."""
    def __init__(self, ptr, bits):
        self.ptr, self.bits = ptr, bits

    def __del__(self):
        _K.frost_lu_free(self.ptr)


def factor(rows, ne=0):
    """LU of a square matrix given as sparse rows (dicts col -> value): row equilibration, then elimination with
    partial pivoting column by column. The multiple-shooting structure (block-bidiagonal plus a few dense columns)
    keeps the fill inside the band. In taylor.c when built: each row a dense window plus the last ne columns."""
    if _K:  # rows: dicts, or ("row col value" text, N) as differences() hands it over
        txt, N = rows if isinstance(rows, tuple) else (" ".join(f"{i} {c} {_hex(v)}" for i, r in enumerate(rows)
                                                                for c, v in r.items()), len(rows))
        bits = gmpy2.get_context().precision
        ptr = _K.frost_lu(bits, N, ne, txt.encode())
        if not ptr:
            raise ZeroDivisionError("singular Jacobian")
        return _CLU(ptr, bits)
    scale = [1 / max(map(abs, r.values())) for r in rows]
    rows = [{c: v * s for c, v in r.items()} for r, s in zip(rows, scale)]
    where = {}
    for i, r in enumerate(rows):
        for c in r:
            where.setdefault(c, set()).add(i)
    free, piv, ops = set(range(len(rows))), [], []
    for col in range(len(rows)):
        cand = [i for i in where.get(col, ()) if i in free and rows[i].get(col)]
        if not cand:
            raise ZeroDivisionError(f"singular Jacobian at column {col}")
        p = max(cand, key=lambda i: abs(rows[i][col]))
        free.discard(p)
        piv.append(p)
        P = rows[p]
        for i in cand:
            if i != p:
                m = rows[i].pop(col) / P[col]
                ops.append((p, i, m))
                ri = rows[i]
                for c, v in P.items():
                    if c != col:
                        if c in ri:
                            ri[c] -= m * v
                        else:
                            ri[c] = -m * v
                            where.setdefault(c, set()).add(i)
    return rows, piv, ops, scale


def lu_solve(lu, b):
    if isinstance(lu, _CLU):
        ptr = _K.frost_lu_solve(lu.ptr, " ".join(map(_hex, b)).encode())
        out = ctypes.string_at(ptr).decode()
        _K.frost_free(ptr)
        return [mpfr(s, lu.bits, 16) for s in out.split()]
    rows, piv, ops, scale = lu
    b = [v * s for v, s in zip(b, scale)]
    for p, i, m in ops:
        b[i] -= m * b[p]
    x = [mpfr(0)] * len(b)
    for col in reversed(range(len(b))):
        r = rows[piv[col]]
        x[col] = (b[piv[col]] - sum(v * x[c] for c, v in r.items() if c != col)) / r[col]
    return x


def solve_lin(A, b):
    """Dense Gaussian elimination with partial pivoting (small systems)."""
    return lu_solve(factor([{j: v for j, v in enumerate(row) if v} for row in A]), b)


_SYS = {}


def _task(a):
    """One segment integration (run in a worker process): returns (end state, mesh, f-elements if asked).
    mode "energy" also accumulates the segment's energy integrals; "elements" additionally returns its f pieces."""
    Z, orbitals, bits, start, r0, r1, eps, p, c, t, mesh, mode = a
    gmpy2.get_context().precision = bits
    sy = _SYS.get((Z, orbitals)) or _SYS.setdefault((Z, orbitals), system(Z, orbitals))
    if isinstance(mode, tuple):  # a Jacobian column on this segment: its nonzeros (u - v) / h, done here in parallel
        base, row0, col, h, R = mode
        st = (csweep if _K else sweep)(start, r0, r1, eps, sy, p, c, t, mesh, "" if _K else False)[0]
        d = [(row0 + k, (u - v) / h) for k, (u, v) in enumerate(zip(_out(st, sy, R), base)) if u != v]
        return " ".join(f"{i} {col} {_hex(v)}" for i, v in d) if _K else [(i, col, v) for i, v in d]
    if _K:
        return csweep(start, r0, r1, eps, sy, p, c, t, mesh, mode)
    st, hs, el = sweep(start, r0, r1, eps, sy, p, c, t, mesh, mode in ("energy", "elements"))
    return st, hs, [(u, h, f) for u, h, f, _ in el] if mode == "elements" else None


class Shooting:
    """Multiple shooting at fixed knots 0 = r_0 < ... < r_J = R, mesh, precision and coupling t.
    x = [f_a(0), y(0) | state at r_1 | ... | state at r_(J-1) | eps]; residual = [end of segment j - state at
    r_(j+1), j < J - 1 | f_a(R), R y' + (2k + 1) y at R, R y^0_aa(R) - 1]. Segments are integrated independently,
    so every residual and Jacobian evaluation is farmed out to the process pool in one batch."""

    def __init__(self, sy, knots, p, c, t, rel, pool, jbits, jp):
        self.sy, self.knots, self.p, self.c, self.t, self.rel, self.pool = sy, knots, p, c, t, rel, pool
        self.jbits, self.jp = jbits, jp  # the Jacobian's precision and order (rel is its difference step)
        self.full = None  # (bits, order, rel) to escalate to when the cheap Jacobian does not contract
        self.n0, self.m = sy.n + len(sy.pots), 2 * (sy.n + len(sy.pots))
        self.N = self.n0 + (len(knots) - 2) * self.m + sy.n
        self.mesh, self.sweeps = [None] * (len(knots) - 1), 0

    def cols(self, j):
        return range(self.n0) if j == 0 else range(self.n0 + (j - 1) * self.m, self.n0 + j * self.m)

    def start(self, x, j):
        sy, z = self.sy, mpfr(0)
        if j == 0:
            return {"f": [(v, z) for v in x[:sy.n]], "y": {pk: (v, z) for pk, v in zip(sy.pots, x[sy.n:self.n0])}}
        return unpack([x[i] for i in self.cols(j)], self.sy)

    def run(self, items, raw=False):
        """Integrate (x, j, mode) items, each segment j from x's state at r_j to r_(j+1), on the pool."""
        bits, n = gmpy2.get_context().precision, self.sy.n
        args = [(self.sy.Z, self.sy.orbitals, bits, self.start(x, j), self.knots[j], self.knots[j + 1], x[-n:],
                 self.p, self.c, self.t, self.mesh[j], mode) for x, j, mode in items]
        res = self.pool.map(_task, args) if self.pool else list(map(_task, args))
        if raw:
            return res
        for (_, j, _), (_, hs, _) in zip(items, res):
            self.mesh[j] = self.mesh[j] or hs
        return res

    def out(self, st, j):
        """Segment j's contribution to the residual (before subtracting the next knot's state)."""
        return _out(st, self.sy, self.knots[-1] if j == len(self.knots) - 2 else None)

    def assemble(self, x, sts):
        r, J = [], len(self.knots) - 1
        for j, st in enumerate(sts):
            o = self.out(st, j)
            r += [u - x[i] for u, i in zip(o, self.cols(j + 1))] if j < J - 1 else o
        return r

    def residual(self, x):
        self.sweeps += 1
        return self.assemble(x, [st for st, _, _ in self.run([(x, j, "") for j in range(len(self.knots) - 1)])])

    def jacobian(self, x):
        """At the Jacobian's own precision and order (call inside its context): chord Newton needs it only accurate
        enough to contract, ~2g digits, not to the target, so its cost does not grow with the digits asked for."""
        p, self.p = self.p, self.jp
        try:
            x = [mpfr(v, self.jbits) for v in x]
            return self.differences(x, self.residual(x))
        finally:
            self.p = p

    def differences(self, x, r):
        """Forward differences, all columns in one parallel batch: a knot-state column needs only its own segment,
        an eps column every segment."""
        J, n, m = len(self.knots) - 1, self.sy.n, self.m
        base = [[r[j * m + k] + x[i] for k, i in enumerate(self.cols(j + 1))] for j in range(J - 1)]
        base.append(r[(J - 1) * m:])
        items = []
        for j in range(J):
            for col in self.cols(j):
                xp = list(x)
                xp[col] += (h := self.rel * (abs(x[col]) or 1))
                items.append((xp, j, (base[j], j * m, col, h, self.knots[-1] if j == J - 1 else None)))
        for col in range(self.N - n, self.N):
            xp = list(x)
            xp[col] += (h := self.rel * abs(x[col]))
            items += [(xp, j, (base[j], j * m, col, h, self.knots[-1] if j == J - 1 else None)) for j in range(J)]
        res = self.run(items, raw=True)
        self.sweeps += m + n
        ident = [(j * m + k, i) for j in range(J - 1) for k, i in enumerate(self.cols(j + 1))]
        if _K:  # "row col value" text straight to the C LU
            minus = _hex(mpfr(-1))
            return " ".join([t for t in res if t] + [f"{i} {c} {minus}" for i, c in ident])
        rows = [dict() for _ in range(self.N)]
        for i, c, v in (e for t in res for e in t):
            rows[i][c] = v
        for i, c in ident:
            rows[i][c] = mpfr(-1)
        return rows

    def solve(self, x, tol):
        """Chord Newton: one Jacobian per stage, refreshed if the step stops shrinking 3x per iteration. Converged
        when f(0), y(0) and eps (the unknowns everything else follows from) move by less than tol, relatively."""
        n, r = self.sy.n, self.residual(x)
        glob = list(range(self.n0)) + list(range(self.N - n, self.N))
        lu, prev, hist, self.jacs = None, None, [], 0
        for self.iters in range(1, 41):
            fresh = lu is None
            # the Jacobian and its solve at jbits, residuals at full precision (not gmpy2.local_context: on exit it
            # installs a copy, which would detach _solve's ctx and freeze every later stage's precision)
            ctx, prec = gmpy2.get_context(), gmpy2.get_context().precision
            ctx.precision = self.jbits
            try:
                if fresh:
                    J = self.jacobian(x)
                    lu, self.jacs = factor((J, self.N) if _K else J, self.sy.n), self.jacs + 1
                dx = lu_solve(lu, [-v for v in r])
            except ZeroDivisionError:  # singular at the cheap precision: redo it at full precision
                if not self.escalate():
                    raise
                hist.append("S")
                lu = None
                continue
            finally:
                ctx.precision = prec
            g = max(abs(dx[i]) / (abs(x[i]) or 1) for i in glob)
            hist.append(f"{'J' if fresh else ''}{float(g):.0e}")
            if not fresh and prev is not None and g > prev / 3:
                lu = None  # chord too slow: refresh the Jacobian at the current x, at full precision and order
                self.escalate()
                continue
            xn = [u + d for u, d in zip(x, dx)]
            rn = self.residual(xn)
            if not all(map(gmpy2.is_finite, rn)):  # a wild step from the cheap LU: back to x, full precision
                if not self.escalate():
                    raise ArithmeticError("non-finite residual")
                hist.append("N")
                lu = None
                continue
            x, r, prev = xn, rn, g
            if g <= tol:
                return x
        raise RuntimeError(f"multiple shooting did not converge to {float(tol):.0e}: steps {' '.join(hist)}")

    def escalate(self):
        """Switch the Jacobian to full precision and order; False if it already is."""
        if not self.full:
            return False
        (self.jbits, self.jp, self.rel), self.full = self.full, None
        return True

    def states_at(self, x, rs):
        """The solution's state at each r in rs (0 < r <= R), integrated from the knot below it."""
        bits, n = gmpy2.get_context().precision, self.sy.n
        js = [max(j for j in range(len(self.knots) - 1) if self.knots[j] < r) for r in rs]
        args = [(self.sy.Z, self.sy.orbitals, bits, self.start(x, j), self.knots[j], r, x[-n:], self.p, self.c, self.t,
                 None, "") for j, r in zip(js, rs)]
        return [st for st, _, _ in (self.pool.map(_task, args) if self.pool else map(_task, args))]


def hydrogenic(Z, n, l, r):
    """(f, f') of the normalised hydrogenic orbital, f = R_nl / r^l."""
    x, m = 2 * mpfr(Z) / n, n - l - 1
    N = gmpy2.sqrt(x ** 3 * factorial(m) / (2 * n * factorial(n + l))) * x ** l
    co = [(-1) ** i * comb(m + 2 * l + 1, m - i) / mpfr(factorial(i)) for i in range(m + 1)]  # L^(2l+1)_m
    L = sum(cf * (x * r) ** i for i, cf in enumerate(co))
    dL = sum(cf * i * x * (x * r) ** (i - 1) for i, cf in enumerate(co) if i)
    e = gmpy2.exp(-x * r / 2)
    return N * e * L, N * e * (dL - x / 2 * L)



def solve(digits, Z=4, orbitals="1s 2s", c=0.15, guard=40, log=print, workers=None):
    """workers: processes for the segment integrations (default: all cores; 1 = serial)."""
    workers = workers or os.cpu_count()
    with (multiprocessing.get_context("spawn").Pool(workers) if workers > 1 else contextlib.nullcontext()) as pool:
        return _solve(digits, Z, orbitals, c, guard, log, pool)


def _solve(digits, Z, orbitals, c, guard, log, pool):
    t0 = time.time()
    up = lambda v: math.ceil(v * 64) / 64  # noqa: E731  knots and walls on a 1/64 grid: exact dyadics
    ctx = gmpy2.get_context()
    sy = system(Z, orbitals)
    n, key, n0 = sy.n, sy.key, sy.n + len(sy.pots)

    def knots(R, kap, g):
        """Over a segment no orbital's growing mode may outgrow the bulk (the outermost orbital) by more than 10^g.
        Orbital i, beyond its radius ~n_i / kappa_i, has already fallen by exp(-(kappa_i - kappa_top) r) against the
        bulk, down to a floor of 1e-4 for its induced tail, and may grow back by that much: inner orbitals only
        shorten the segments where they are themselves large. ponytail: crude magnitude model; it sets cost only."""
        kt, ks = min(kap), [0.0]
        while ks[-1] < R:
            r = ks[-1]
            L = min((g * LN10 + min((k - kt) * max(r - nn / k, 0), 4 * LN10)) / (k + kt) for k, (nn, _) in zip(kap, sy.nl))
            ks.append(min(max(up(r + L), r + 1 / 64), R))
        return [mpfr(v) for v in ks]

    stiff = []  # set once a wall stage needed the full-precision Jacobian: the next (larger R) will too

    def stage(x_of, R, t, d, g, kap):
        # the Jacobian is carried to PJ digits (forward differences: PJ / 2 correct), enough to contract the chord
        P, lc = d + 2 * g + guard, math.log10(1 / c)
        PJ = P if stiff or t < 1 else min(P, 4 * g + 30)  # the homotopy (heavy atoms) needs the full one
        ctx.precision = int(P / math.log10(2))
        ks, p = knots(R, kap, g), math.ceil((d + 5) / lc)
        sh = Shooting(sy, ks, p, mpfr(c), mpfr(t), mpfr(10) ** (-PJ / 2), pool, int(PJ / math.log10(2)),
                      min(p, math.ceil((PJ / 2 + 5) / lc)))
        sh.full = (ctx.precision, p, mpfr(10) ** (-P / 2)) if PJ < P else None  # stiff atoms may need it
        x = sh.solve(x_of(ks), mpfr(10) ** -d)
        if t == 1 and PJ < P and sh.full is None:
            stiff.append(R)
        if max(x[-n:]) >= 0:  # a box state of the wall, not a bound orbital
            raise ArithmeticError(f"unbound orbital at R={float(R)}: eps={[float(e) for e in x[-n:]]}")
        return x, sh

    def regrid(x, sh, kt, x0, eps):
        """Guess for new knots from a solved stage (x, sh): the old solution at each new knot, integrated from the
        old knot below it. Within ~1 / kt of its wall the old solution is distorted by it: beyond
        r* = R - 1 / kt each orbital continues with its own logarithmic derivative at r* (no slower than kt / 2),
        each potential as its multipole tail y ~ r^-(2k+1)."""
        R = sh.knots[-1]
        rs = max(R - 1 / mpfr(kt), R / 2)

        def x_of(ks):
            inner = [r for r in ks[1:-1] if r <= rs]
            sts = sh.states_at(x, inner + [rs])
            base, lam = sts[-1], [min(dv / v, -kt / 2) if v else -kt for v, dv in sts[-1]["f"]]
            gx, it = list(x0), iter(sts)
            for r in ks[1:-1]:
                if r <= rs:
                    st = next(it)
                else:
                    e = [gmpy2.exp(lm * (r - rs)) for lm in lam]
                    w = {pk: (rs / r) ** (2 * pk[1] + 1) for pk in sy.pots}
                    st = {"f": [(v * ei, lm * v * ei) for (v, _), ei, lm in zip(base["f"], e, lam)],
                          "y": {pk: (v * w[pk], -(2 * pk[1] + 1) * v * w[pk] / r) for pk, (v, _) in base["y"].items()}}
                gx += pack(st, sy)
            return gx + list(eps)
        return x_of

    kaps = lambda e: [math.sqrt(-2 * float(v)) for v in e]  # noqa: E731
    # r^(n-1) e^(-kappa r) is down to ~1e-3 of its peak at kappa r = 2n + 6, for every n (8 / kappa squeezed 5s unbound)
    wall = lambda kap: up(max((2 * nn + 6) / k for k, (nn, _) in zip(kap, sy.nl)))  # noqa: E731

    # Homotopy t: 0 -> 1 from hydrogenic orbitals (exact at t = 0; y from the first Newton step). The wall follows
    # the orbitals, R = max (2n + 6) / kappa(t), so it never squeezes them. The coupling step dt changes the
    # potentials by O(dt), which a segment amplifies by 10^g: keep g ~ 1 here (the wall continuation only changes
    # them by ~exp(-kappa R) and can afford long segments). dt grows after easy steps and halves after a failure.
    d, g = 10, 1
    eps = [-mpfr(Z) ** 2 / (2 * nn ** 2) for nn, _ in sy.nl]
    R = wall(kaps(eps))

    def hyd(ks):
        x = [hydrogenic(Z, nn, ll, 0)[0] for nn, ll in sy.nl] + [mpfr(0)] * len(sy.pots)
        for r in ks[1:-1]:
            x += [v for nn, ll in sy.nl for v in hydrogenic(Z, nn, ll, r)] + [mpfr(0)] * (2 * len(sy.pots))
        return x + eps

    x, sh = stage(hyd, R, 0, d, g, kaps(eps))
    t, dt, prev, steps = 0, 1 / 32, None, 0
    while t < 1:
        kap, cur = kaps(x[-n:]), x[:n0] + x[-n:]  # the start values and eps carry over; knot states are regridded
        tn, Rn = min(1, t + dt), max(R, wall(kap))
        pred = cur if prev is None else [u + (u - v) * (tn - t) / (t - prev[0]) for u, v in zip(cur, prev[1])]
        try:
            xn, shn = stage(regrid(x, sh, min(kap), pred[:n0], pred[n0:]), Rn, tn, d, g, kap)
        except (ArithmeticError, RuntimeError, ZeroDivisionError):
            dt /= 2
            if dt < 1e-4:
                raise
            continue
        prev, t, x, sh, R, steps = (t, cur), tn, xn, shn, Rn, steps + 1
        dt = min(1.5 * dt, 0.25) if shn.jacs == 1 else dt  # one Jacobian sufficed: Newton stayed linear
    log(f"homotopy done in {steps} steps: R={R} eps={[float(e) for e in x[-n:]]}  {time.time() - t0:.1f}s")

    # The wall grows by `ratio` per stage: up to 2 after stages that needed one Jacobian, back to 1.2 after a harder
    # one or a failure (then the stage is retried). Only the path changes: the last stage has the same wall, target,
    # order and tolerance as with a fixed 1.2, and at least the same g, hence at least the same precision.
    ratio = 1.2
    while True:
        kap = kaps(x[-n:])
        kt = min(kap)
        Rf = up((digits + 8) * LN10 / (2 * kt))
        Rn = Rf if Rf <= 1.1 * ratio * R else up(R * ratio)  # no sliver of a last stage
        d = min(digits, 2 * kt * Rn / LN10) + 10
        # moving the wall changes the solution by ~exp(-kappa R) near the old wall, which a segment amplifies by 10^g;
        # never below the g of the 1.2 path (old wall Rn / 1.2)
        g = min(30, max(1, kt * max(R, Rn / 1.2) / LN10 / 2))
        try:
            xn, shn = stage(regrid(x, sh, kt, x[:n0], x[-n:]), Rn, 1, d, g, kap)
        except (ArithmeticError, RuntimeError, ZeroDivisionError):
            if ratio <= 1.2:
                raise
            ratio = 1.2
            continue
        x, sh, R = xn, shn, Rn
        ratio = min(2.0, 1.25 * ratio) if sh.jacs == 1 else 1.2
        log(f"R={R:9.4f} digits={d:6.1f} segments={len(sh.knots) - 1:3d} order={sh.p:4d} steps={sum(map(len, sh.mesh)):4d} "
            f"sweeps={sh.sweeps:4d} eps={[round(float(e), 12) for e in x[-n:]]}  {time.time() - t0:7.1f}s")
        if R == Rf:
            break

    # energy: E = 2 sum_l (2l+1) sum_ij D_ij h_ij
    #           + sum_ll' (2l+1)(2l'+1) sum D_ij D_mo [2 R^0(ij, mo) - sum_k (l k l')^2 R^k(im, jo)],  D = S^-1 per l
    z = mpfr(0)
    I = {"T": {q: z for q in sy.same}, "V": {q: z for q in sy.same}, "M": {pk: z for pk in sy.pots}, "J": {q: z for q in sy.J}}
    elems = []
    for st, _, el in sh.run([(x, j, "elements") for j in range(len(sh.knots) - 1)]):
        elems += el
        for k_ in I:
            for q in I[k_]:
                I[k_][q] += st[k_][q]
    G = lambda q, p_, k: I["J"][(q, p_, k)]  # noqa: E731  R^k(q, p) = int P_q Y^k_p / r (y is the full potential)
    blocks = {}
    for i, li in enumerate(sy.l):
        blocks.setdefault(li, []).append(i)
    Dm = {}
    for li, idx in blocks.items():
        cols = [solve_lin([[I["M"][(key(i, j), 0)] for j in idx] for i in idx], [mpfr(int(i == k)) for i in idx]) for k in idx]
        Dm.update({(i, j): cols[y][x_] for x_, i in enumerate(idx) for y, j in enumerate(idx)})
    E = Evir = mpfr(0)
    for li, idx in blocks.items():
        for i in idx:
            for j in idx:
                E += 2 * (2 * li + 1) * Dm[(i, j)] * (I["T"][key(i, j)] + I["V"][key(i, j)])
                Evir -= 2 * (2 * li + 1) * Dm[(i, j)] * I["T"][key(i, j)]
    for li, A in blocks.items():
        for lj, B in blocks.items():
            w = (2 * li + 1) * (2 * lj + 1)
            for i in A:
                for j in A:
                    for m in B:
                        for o in B:
                            v = 2 * G(key(i, j), key(m, o), 0) - sum(
                                mpfr(threej2(li, k, lj)) * G(key(i, m), key(j, o), k) for k in sy.K[key(i, m)])
                            E += w * Dm[(i, j)] * Dm[(m, o)] * v
    nd = int(d) + 5
    exact = lambda v: [str(m) for m in mpfr(v).as_mantissa_exp()]  # noqa: E731
    fmt = lambda v: format(v, f".{nd}e")  # noqa: E731
    return {
        "Z": Z, "n": n, "orbitals": orbitals.split(), "l": sy.l, "walls": [R] * n, "digits": digits, "order": sh.p,
        "step": c, "segments": len(sh.knots) - 1, "precision_bits": ctx.precision, "seconds": round(time.time() - t0, 1),
        "E": format(E, f".{nd}f"), "E_virial": format(Evir, f".{nd}f"),
        "eps": [fmt(e) for e in x[-n:]], "f0": [fmt(v) for v in x[:n]],
        "S_offdiag": [fmt(I["M"][(q, 0)]) for q in sy.same if q[0] != q[1]],
        "elements": [[exact(r0), exact(h), [[exact(v) for v in u] for u in f]] for r0, h, f in elems],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--digits", type=int, required=True)
    ap.add_argument("--Z", type=int, default=4)
    ap.add_argument("--orbitals", default="1s 2s", help='closed subshells, e.g. "1s 2s 2p"')
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--guard", type=int, default=40)
    ap.add_argument("--workers", type=int, help="processes (default: all cores)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = solve(a.digits, a.Z, a.orbitals, a.step, a.guard, workers=a.workers)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"E      = {out['E']}\nE_vir  = {out['E_virial']}\neps    = {out['eps']}\nS_ij   = {out['S_offdiag']}\n"
          f"({out['seconds']}s, wrote {a.out})")
