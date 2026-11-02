"""End-to-end check (~25 s on a multi-core laptop): oracle runs reproduce reference HF energies and orbital properties, and their exact-rational
certificates are valid (above the reference) and tight upper bounds."""
import gmpy2
from gmpy2 import mpfr, mpq

import oracle_ns
from certify import certify
from oracle import solve
from properties import properties

# From he_like_series.txt: two independent oracle runs per Z agree to >= 103 s.f.
# He also matches all 66 digits of Hatano & Yamamoto, J. Comput. Chem. Jpn. Int. Ed. 11 (2025).
E_REF = {
    1: mpq("-0.48792973437085174869283837275680759069853910785709626903828113406656"),
    2: mpq("-2.861679995612238878775543740022563429412879776752447053906864697434188362627955"),
    10: mpq("-93.861113519231338154978163881035306429111285436383821944783379879614"),
}


def test_isoelectronic_25_digits():
    """H- (largest wall, no Coulomb tail), He, Ne8+ (smallest wall: continuation must scale with Z)."""
    for Z, ref in E_REF.items():
        data = solve(25, Z=Z, log=lambda *_: None)
        tol = abs(ref) / 10**24
        assert abs(mpq(data["E"]) - ref) < tol, (Z, data["E"])
        E = certify(data)
        assert ref < E < ref + tol, (Z, float(E - ref))

# Hatano & Yamamoto, J. Comput. Chem. Jpn. Int. Ed. 11 (2025): 43 s.f., truncated
E_BE = mpq("-14.5730231683163995824114473598700144538777")


def test_beryllium_20_digits():
    """Two s orbitals: multiple shooting with one wall, certificate with D = S^-1."""
    data = oracle_ns.solve(20, Z=4, orbitals="1s 2s", log=lambda *_: None)
    tol = abs(E_BE) / 10**19
    assert abs(mpq(data["E"]) - E_BE) < tol, data["E"]
    E = certify(data)
    assert E_BE < E < E_BE + tol, float(E - E_BE)


# Cinal, J. Math. Chem. 58, 1571 (2020), as quoted in arXiv:2309.10704 (14 s.f.)
E_NE = mpq("-128.54709810938")


def test_neon_15_digits():
    """p shell: multipole potentials (k = 0, 1, 2), 3j-weighted exchange, the 1s/2s tails induced by the 2p
    (one common wall), and the certificate's F^k/G^k energy."""
    data = oracle_ns.solve(15, Z=10, orbitals="1s 2s 2p", log=lambda *_: None)
    assert abs(mpq(data["E"]) - E_NE) < abs(E_NE) / 10**13, data["E"]
    E = certify(data)
    assert abs(E - mpq(data["E"])) < abs(E) / 10**22, float(E - mpq(data["E"]))  # bound = solver energy


def test_he_properties_30_digits():
    """Exact cusp; rho(0), <r^-2>, <r> against Hatano & Yamamoto (2025) (32, 35, 63 s.f.); <1/r> = V_H(0)."""
    data = solve(30, log=lambda *_: None)
    p = properties(data, (-2, -1, 1))
    assert p["cusp"] == [-2]
    for got, ref in ((p["rho0"], "3.5959182637622489207069875630017"), (p["r^k"][-2][0], "5.9955036673125245972735587422203366"),
                     (p["r^k"][1][0], "0.927273404731493197418927707229715980644270061322365802985452499"),
                     (p["r^k"][-1][0], data["VH0"])):
        assert abs(mpq(got) / mpq(ref) - 1) < mpq(1, 10**29), (got, ref)


def test_c_kernel_matches_python():
    """taylor.c against oracle_ns.sweep (d shell, Frobenius start and off-origin, energy and elements): the same
    recurrences, but C forms each Cauchy product exactly and rounds once, so they agree to the Python sweep's own
    rounding error (k roundings per product), far below 2^-(bits - 40)."""
    assert oracle_ns._K, "taylor.c did not build"
    bits = 300
    gmpy2.get_context().precision = bits
    sy = oracle_ns.system(30, "1s 2s 2p 3s 3p 3d 4s")
    eps = [-mpfr(30) ** 2 / (3 * nn ** 2) for nn, _ in sy.nl]
    flat = lambda st: ([v for fa in st["f"] for v in fa] + [v for p in st["y"].values() for v in p]  # noqa: E731
                       + [v for k in "TVMJ" for v in st[k].values()])
    for r0, r1 in [(mpfr(0), mpfr("0.375")), (mpfr("0.5"), mpfr("0.8125"))]:
        st = {"f": [oracle_ns.hydrogenic(30, nn, ll, r0) for nn, ll in sy.nl],
              "y": {pk: (mpfr(1) / (3 + i), -mpfr(1) / (7 + i)) for i, pk in enumerate(sy.pots)}}
        a, hs, el = oracle_ns.sweep(st, r0, r1, eps, sy, 24, mpfr("0.15"), mpfr("0.7"), None, True)
        b, hc, ec = oracle_ns.csweep(st, r0, r1, eps, sy, 24, mpfr("0.15"), mpfr("0.7"), hs, "elements")
        tol = mpfr(2) ** (40 - bits)
        assert max(abs(u - v) / abs(v) for u, v in zip(flat(b), flat(a)) if v) < tol, float(r0)
        for (_, h, fc), (_, _, fp, _) in zip(ec, el):  # series coefficients by what they contribute, |du_k| h^k
            for x, y in zip(fc, fp):
                assert max(abs(u - v) * h ** k for k, (u, v) in enumerate(zip(x, y))) <= tol * max(
                    abs(v) * h ** k for k, v in enumerate(y)), float(r0)


if __name__ == "__main__":
    test_c_kernel_matches_python()
    test_isoelectronic_25_digits()
    test_beryllium_20_digits()
    test_neon_15_digits()
    test_he_properties_30_digits()
    print("ok")
