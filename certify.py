"""Rigorous upper bound on the closed-shell RHF energy of an atom, in exact rational arithmetic.

Reads the piecewise Taylor polynomials written by oracle.py (one orbital) or oracle_ns.py (several, each
with its own wall) and turns them into admissible trial orbitals *exactly*: dyadic coefficients,
continuity forced at every knot, f_i(R_i) = 0 and f_i = 0 beyond R_i. For one orbital the energy is

    E[f] = 2 (T + V) / N + J / N^2
    N = int f^2 r^2,  T = 1/2 int f'^2 r^2,  V = -Z int f^2 r,
    J = 2 int f^2 r Q(r),  Q(r) = int_0^r f^2 s^2 ds        (= (phi phi | phi phi))

and in general the determinant energy with P = S^-1 (see certify()), using integers and fractions
only: every integrand is a polynomial on a dyadic interval.
By the variational principle E_HF <= E[f], with no rounding anywhere. The decimal printed
is rounded towards +infinity, so it is itself an upper bound. Shares no code with oracle.py.

    python certify.py he_100.json [--order q] [--print-digits n]
"""
import argparse
import json
import multiprocessing
from math import factorial

from gmpy2 import lcm, log10, mpfr, mpq, mpz

_L = {}


def lcm_to(n):
    """lcm(1..n), cached."""
    if n not in _L:
        v = mpz(1)
        for k in range(2, n + 1):
            v = lcm(v, k)
        _L[n] = v
    return _L[n]


def pmul_school(a, b):
    out = [mpz(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def pmul(a, b):
    """Exact product of integer polynomials by Kronecker substitution: each sign part is packed into one
    integer (k bytes per coefficient, enough that no slot of the product overflows), multiplied once by
    GMP, and unpacked; a = a+ - a-."""
    if min(len(a), len(b)) < 16:
        return pmul_school(a, b)
    n = len(a) + len(b) - 1
    k = (max(map(abs, a)).bit_length() + max(map(abs, b)).bit_length() + min(len(a), len(b)).bit_length() + 8) // 8

    def pack(c):
        return mpz(int.from_bytes(b"".join(int(v).to_bytes(k, "little") for v in c), "little"))

    def parts(c):
        return pack([max(v, 0) for v in c]), pack([max(-v, 0) for v in c])

    def unpack(x):
        d = int(x).to_bytes(n * k, "little")
        return [int.from_bytes(d[i * k:(i + 1) * k], "little") for i in range(n)]

    (ap, an), (bp, bn) = parts(a), parts(b)
    pos, neg = unpack(ap * bp + an * bn), unpack(ap * bn + an * bp)
    return [mpz(x - y) for x, y in zip(pos, neg)]


def pint(a):
    """int_0^1 a(s) ds for an integer polynomial, exactly."""
    L = lcm_to(len(a))
    return mpq(sum(c * (L // (k + 1)) for k, c in enumerate(a)), L)


def dyadic(pair):
    m, e = int(pair[0]), int(pair[1])
    return mpq(m * 2 ** e) if e >= 0 else mpq(m, 2 ** -e)


def inverse(S):
    """Exact inverse of a small positive-definite rational matrix (Gauss-Jordan; pivots are positive)."""
    n = len(S)
    M = [list(row) + [mpq(int(i == j)) for j in range(n)] for i, row in enumerate(S)]
    for c in range(n):
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c:
                M[r] = [x - M[r][c] * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def trial(data, order=None):
    """The trial orbitals of `data` exactly: oracle.py writes one per element, oracle_ns.py ("walls" present)
    a list, each zero beyond its own wall. Returns (els, F, A): knots (r0, h) as mpq, and per element the
    integer coefficients F[e][i] in s = (r - r0) / h (all scaled by one 2^B), with continuity and
    f_i(R_i) = 0 imposed exactly; 2^A makes every knot an integer."""
    els = []
    for r0, h, fs in data["elements"]:
        r0, h = dyadic(r0), dyadic(h)
        fs = fs if "walls" in data else [fs]
        fs = [[dyadic(c) for c in f][: None if order is None else order + 1] for f in fs]
        els.append((r0, h, [[c * h ** k for k, c in enumerate(f)] for f in fs]))  # coefficients in s = (r - r0) / h
    if not els or els[0][0] != 0:
        raise ValueError("first element must start at r = 0")
    if any(h <= 0 for _, h, _ in els) or any(a0 + h0 != a1 for (a0, h0, _), (a1, _, _) in zip(els, els[1:])):
        raise ValueError("elements must tile [0, R] exactly")  # a reversed interval would fake a lower 'bound'
    n = len(els[0][2])
    last = [max(e for e, (_, _, fs) in enumerate(els) if fs[i]) for i in range(n)]
    if any(not fs[i] for i in range(n) for _, _, fs in els[: last[i] + 1]):
        raise ValueError("each orbital must be supported on [0, R_i]")
    # one common power of two makes every coefficient and knot an integer (E is scale invariant)
    B = max(c.denominator.bit_length() - 1 for _, _, fs in els for f in fs for c in f)
    A = max(v.denominator.bit_length() - 1 for a, h, _ in els for v in (a, h))
    F = [[[mpz(c * 2 ** B) for c in f] for f in fs] for _, _, fs in els]
    for i in range(n):
        for e in range(1, last[i] + 1):  # exact continuity: next value at s=0 := previous value at s=1
            F[e][i][0] = sum(F[e - 1][i])
        F[last[i]][i][1] -= sum(F[last[i]][i])  # exact f_i(R_i) = 0
    return els, F, A


def threej2(a, b, c):
    """(a b c; 0 0 0)^2, exactly."""
    J = a + b + c
    if J % 2 or c > a + b or c < abs(a - b):
        return mpq(0)
    g = J // 2
    return mpq(factorial(J - 2 * a) * factorial(J - 2 * b) * factorial(J - 2 * c), factorial(J + 1)) * mpq(
        factorial(g), factorial(g - a) * factorial(g - b) * factorial(g - c)) ** 2


def element(task):
    """One element's exact contributions (run in a worker): its S, T, V and Q pieces, and for each needed
    (q, p, k) the factor c1 that multiplies Q^k_p accumulated over earlier elements, plus its own inner part c2."""
    (a, h, _), Fe, A, Z, l, lq, K, need, same = task
    o = {"S": {}, "T": {}, "V": {}, "Q": {}, "C": {}}
    al, et = mpz(a * 2 ** A), mpz(h * 2 ** A)  # r = (al + et s) / 2^A
    U = [[mpz(1)]]
    for _ in range(4 * max(l) + 2):
        U.append(pmul(U[-1], [al, et]))  # U[m] = (al + et s)^m = (2^A r)^m
    live = [q for q in lq if Fe[q[0]] and Fe[q[1]]]
    rho = {q: pmul(Fe[q[0]], Fe[q[1]]) for q in live}
    dP = [[(k + 1) * P[k + 1] for k in range(len(P) - 1)] for P in Fe]
    g, W, Lw = {}, {}, {}
    for q, p, k in need:
        if q not in live:
            continue
        mq, mp = lq[q] + 1 - k, lq[p] + k + 2
        if (q, mq) not in g:
            g[(q, mq)] = pmul(rho[q], U[mq])
        gq = g[(q, mq)]
        c1, c2 = mpq(et, 2 ** ((mq + 1) * A)) * pint(gq), mpq(0)
        if p in live:
            if (p, mp) not in W:
                w = pmul(rho[p], U[mp])
                Lw[(p, mp)] = lcm_to(len(w))
                W[(p, mp)] = [mpz(0)] + [c * (Lw[(p, mp)] // (j + 1)) for j, c in enumerate(w)]  # Lw * int_0^s w
            c2 = mpq(et * et, 2 ** ((mq + mp + 2) * A) * Lw[(p, mp)]) * pint(pmul(gq, W[(p, mp)]))
        o["C"][(q, p, k)] = c1, c2
    for q in live:
        if q in same:
            m = 2 * l[q[0]] + 2
            o["S"][q] = mpq(et, 2 ** ((m + 1) * A)) * pint(pmul(rho[q], U[m]))
            o["T"][q] = pint(pmul(pmul(dP[q[0]], dP[q[1]]), U[m])) / (2 * et * 2 ** ((m - 1) * A))
            o["V"][q] = -Z * mpq(et, 2 ** (m * A)) * pint(pmul(rho[q], U[m - 1]))
        for k in K[q]:
            m = lq[q] + k + 2
            o["Q"][(q, k)] = mpq(et, 2 ** ((m + 1) * A)) * pint(pmul(rho[q], U[m]))
    return o


def certify(data, order=None):
    """Exact E[f] (an mpq) for the trial orbitals of `data`, R_i = r^l_i f_i. E is the closed-shell determinant's,
        E = 2 sum_l (2l+1) sum_ij D_ij h_ij
            + sum_ll' (2l+1)(2l'+1) sum_{ij in l, mo in l'} D_ij D_mo [2 R^0(ij, mo) - sum_k (l k l'; 000)^2 R^k(im, jo)],
    D = S^-1 within each l, exact for the span of the orbitals whether or not they are orthonormal; one s orbital
    gives 2 (T + V) / N + J / N^2. R^k(q, p) = int int P_q(r) P_p(s) r_<^k / r_>^(k+1) with P_q = r^(l_q+2) f_i f_j,
    l_q = l_i + l_j: its integrand r^(l_q+1-k) f_i f_j(r) int_0^r s^(l_p+k+2) f_m f_o is a polynomial because the
    triangle rule gives k <= l_q, so every term is an exact rational."""
    Z = int(data["Z"])
    els, F, A = trial(data, order)
    n = len(F[0])
    l = data.get("l", [0] * n)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    lq = {q: l[q[0]] + l[q[1]] for q in pairs}
    K = {q: range(abs(l[q[0]] - l[q[1]]), lq[q] + 1, 2) for q in pairs}
    same = [q for q in pairs if l[q[0]] == l[q[1]]]
    ltype = lambda q: sorted((l[q[0]], l[q[1]]))  # noqa: E731
    need = sorted({(q, p, 0) for q in same for p in same}
                  | {(q, p, k) for q in pairs for p in pairs if ltype(q) == ltype(p) for k in K[q]})
    S, T, V = ({q: mpq(0) for q in same} for _ in range(3))
    Q = {(p, k): mpq(0) for p in pairs for k in K[p]}  # int_0^r s^(l_p+k+2) rho_p ds
    C = {x: mpq(0) for x in need}  # int rho_q r^(l_q+1-k) Q^k_p
    # elements in parallel; only Q (int_0^r) runs across elements, so C is completed by a prefix sum afterwards
    with multiprocessing.get_context("spawn").Pool() as pool:
        parts = pool.map(element, [(e, Fe, A, Z, l, lq, K, need, same) for e, Fe in zip(els, F)])
    for o in parts:
        for x, (c1, c2) in o["C"].items():
            C[x] += c1 * Q[(x[1], x[2])] + c2
        for acc, d in ((S, o["S"]), (T, o["T"]), (V, o["V"]), (Q, o["Q"])):
            for y, v in d.items():
                acc[y] += v
    key = lambda i, j: (min(i, j), max(i, j))  # noqa: E731
    G = lambda q, p, k: C[(q, p, k)] + C[(p, q, k)]  # noqa: E731  R^k(q, p)
    blocks = {}
    for i, li in enumerate(l):
        blocks.setdefault(li, []).append(i)
    D = {}
    for li, idx in blocks.items():
        inv = inverse([[S[key(i, j)] for j in idx] for i in idx])
        D.update({(i, j): inv[x][y] for x, i in enumerate(idx) for y, j in enumerate(idx)})
    E = sum(2 * (2 * li + 1) * D[(i, j)] * (T[key(i, j)] + V[key(i, j)]) for li, I in blocks.items() for i in I for j in I)
    for li, I in blocks.items():
        for lj, M in blocks.items():
            for i in I:
                for j in I:
                    for m in M:
                        for o in M:
                            x = 2 * G(key(i, j), key(m, o), 0) - sum(
                                threej2(li, k, lj) * G(key(i, m), key(j, o), k) for k in K[key(i, m)])
                            E += (2 * li + 1) * (2 * lj + 1) * D[(i, j)] * D[(m, o)] * x
    return E


def decimal_up(q, n):
    """q rounded towards +infinity to n decimals, as a string."""
    s = 10 ** n
    v = -((-q.numerator * s) // q.denominator)
    sign, v = ("-", -v) if v < 0 else ("", v)
    return f"{sign}{v // s}.{str(v % s).zfill(n)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orbital")
    ap.add_argument("--order", type=int, help="truncate each step's polynomial to this order")
    ap.add_argument("--print-digits", type=int)
    a = ap.parse_args()
    data = json.load(open(a.orbital))
    E = certify(data, a.order)
    n = a.print_digits or int(data["digits"]) + 10
    gap = mpfr(E - mpq(data["E"]), 64)
    print(f"E_cert  <= {decimal_up(E, n)}   (exact rational upper bound, rounded up)")
    print(f"E_oracle = {data['E'][: n + 3]}")
    sf = -log10(abs(gap) / abs(mpfr(E, 64))) if gap else float("inf")
    print(f"E_cert - E_oracle = {gap:.3e}  ->  agree to {float(sf):.1f} significant figures")
