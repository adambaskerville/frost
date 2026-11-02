# FROST: Hartree–Fock limits to hundreds of digits

Fock Radial Orbital Shooting with Taylor series. Requires Python 3 and `gmpy2` (`pip install -r requirements.txt`
installs the tested versions; the paper scripts also need `matplotlib`). `oracle_ns.py` compiles its C kernel
`taylor.c` on first use against the MPFR and GMP headers and libraries under the Python prefix (a conda environment
with `mpfr` and `gmp` provides them); without them it falls back to the much slower pure-Python integration.

Paper I (He and Be, release v1.0) is in `paper/` and paper II (the closed-shell atoms from Ne to Xe, release v2.0)
in `paper2/`. The orbital files of every run in both papers are archived on Zenodo (DOIs to be added).

Closed-shell 1s² RHF is a 1-D nonlinear eigenproblem, so no basis set is needed:

    r f'' + 2 f' = 2 (Y − Z − ε r) f,     Y'' = −r f²        (φ = f/√4π,  Y = r V_H)

| file | what |
|---|---|
| `oracle.py` | Taylor-series shooting. Frobenius series at r = 0, then steps sized from each series' own radius of convergence; nested secant solves for ε − V_H(0) (f(R) = 0) and f(0) (normalisation), with continuation in the wall radius R. Order, wall and precision are independent knobs. |
| `oracle_ns.py` | The same method for any closed-shell atom (`--orbitals "1s 2s 2p"`). With f = P/r^(l+1) and y^k = Y^k/r^(k+1) every equation reads r u'' + 2(L+1) u' = g. One common wall, multiple shooting over segments of bounded growth (chord Newton, sparse LU), homotopy in the electron–electron coupling from hydrogenic orbitals, then continuation in the wall. Paper I's Be runs used the earlier version in release v1.0 (one wall per orbital, nested scalar shooting). |
| `taylor.c` | C kernel of `oracle_ns.py` (MPFR): the Taylor integration of a segment, with exact dot products, and the sparse LU. Built automatically; the test checks it against the Python integration. |
| `certify.py` | Independent exact-rational evaluation of the RHF energy functional for the oracle's orbitals (dyadic coefficients; continuity and f_i(R_i) = 0 imposed exactly; for several orbitals the determinant energy with P = S⁻¹, exact for their span whether or not they are orthonormal). For p and d shells the exchange multipoles k ≥ 1 enter through Slater integrals with exact 3j symbols; writing P = r^(l+1) f keeps every integrand polynomial. The variational principle gives E_HF ≤ E[f] with no rounding anywhere. |
| `properties.py` | Exact cusp ratio f'(0)/f(0), ρ(0) and ⟨rᵏ⟩ of the certified trial orbitals (exact rational integration). |
| `test_hf_taylor.py` | End-to-end check, about 11 s on 12 cores (H⁻, He, Ne⁸⁺ with `oracle.py`; Be and Ne with `oracle_ns.py`; He properties vs Hatano & Yamamoto; the C kernel against the Python integration). |
| `he_like_series.txt` | Certified energies, Z = 1–10, to 100 significant figures. |
| `paper/` | Paper I (`hf_taylor.tex`); `make_paper.py` generates its numbers, tables and figures. |
| `paper2/` | Paper II (`frost_pd.tex`); `run_all.sh` makes every run, `make_paper2.py` its numbers, tables and figure. `data/runs/` holds each run's certificate output, logs and orbital-file checksums. |

## Results: He (Apple M4 Pro, one core, Python + gmpy2)

| | significant figures | wall time |
|---|---|---|
| Hatano & Yamamoto, J. Comput. Chem. Jpn. Int. Ed. 11 (2025) — previous record | 66 | |
| certified upper bound, R = 109 | 124 | 11 s + 26 s |
| oracle, two independent runs (R 177/185, order 358/261, step 0.25/0.15) | 204 (agreement) | ~1 min each |
| certified upper bound, R = 185 | 204 | 1 min + 1.5 min |
| oracle, two independent runs (R 432/445, order 626/856, step 0.15/0.25) | 503 (agreement) | 7.4 / 8.2 min |
| certified upper bound, R = 432 | 503 (lies 1.7e−503 above the R = 445 run) | 7.4 min + 34 min |

    E_HF(He) = −2.861679995612238878775543740022563429412879776752447053906864697434188362627955696043994965
               3409977869098417516573190127023919647238612566769378991533811865805702964675400062623498834061443950
               9601437492413535847678518248340821214097097161854532588946771779765923302453303780736622271074836399
               7217257536670973963250401353173639891864905807424508584898210511928945721199380240447623179624760205
               8266331292126485637864080845362955571466447416432193936214125615947217433179322306477985718919570787
               001078165747   (503 s.f.)

All 66 digits of Hatano & Yamamoto agree; the digits they print beyond 66 do not.
Certified digits ≈ 1.15 R: the Dirichlet wall is the only floor.

## Beryllium (1s²2s²), paper I

| | significant figures | wall time |
|---|---|---|
| Hatano & Yamamoto (2025) — previous record | 43 | |
| certified upper bound, 100-digit run (walls 40 / 158 bohr), vs independent 110-digit run | 103 | 170 s + 18 s |
| certified upper bound, 200-digit run (walls 78 / 305 bohr), vs independent 210-digit run | 202 | 23 min + 1.5 min |
| certified upper bound, 500-digit run (walls 190 / 744 bohr), vs independent 510-digit run | 501 (bound lies 7.1e−501 above it; wall law fitted on the 100-digit run predicts 7.0e−501) | 4.0 h + 13 min |

    E_HF(Be) = -14.5730231683163995824114473598700144538777213577740054138052227220687300019382263514550803169376936498
                   9813411646059041422081007462751047907294527780507389752281849318054044550627632938656700960809617754
                   8999876512967220287914508898666849101213450870750794270003470399581021887552900121239020270945596918
                   4619912969154498953491963320425651173890872736629691718557047823504902492715841598820767039342895408
                   179338392625576822389455941180416448874596914581406525412122256259765032056688952590379295553928939   (501 s.f.)

All 43 figures of Hatano & Yamamoto agree. Each bound lies above its independent run (larger walls,
c = 0.25, guard 30); the difference is the wall error of the certified run. Paper I's solver (release v1.0) handled the two orbitals
as below; the current `oracle_ns.py` replaces the first three points with one common wall and multiple shooting
(paper II):

- **One wall per orbital.** Integrated out to the 2s wall, the 1s growing mode exp(3.1 r) would need ~4x
  the working precision. The 1s gets its own wall at the same exp(-2κR); its space H¹₀(0, R₁) is still
  admissible, so the bound is unaffected.
- **Canonical orbitals, not exact orthogonality.** With unequal walls, exact orthogonality needs a
  multiplier whose Newton step forces a 1s tail far above its natural size, which the exchange term feeds
  into the 2s growing mode. The solver keeps the canonical equations (λ = 0); the orbitals are orthogonal to
  ~exp(-κ₁R₁), costing O(S₁₂²), the size of the wall error. The certificate's P = S⁻¹ energy is exact for
  the span regardless.
- **Nested scalar solves, innermost orbital first.** Every trial 2s energy re-solves the 1s energy, so no
  growing 1s mode reaches the 2s equation through exchange (a joint solve finds spurious roots).
- **Start.** Homotopy in the electron-electron coupling from exact hydrogenic orbitals.

## Paper II: closed-shell atoms from neon to xenon

Occupied p and d shells bring exchange multipoles with k ≥ 1, whose induced tails give every inner canonical
orbital the decay of the outermost one (Handy, Marron & Silverstone 1969), so all orbitals share one wall.
Multiple shooting over segments of bounded growth keeps the stiff inner orbitals under control. Run A is
certified; run B is independent (10 more digits, c = 0.25, guard 30). Times are on the 12 cores of an Apple M4 Pro.

| atom | Z | digits | A–B agreement (s.f.) | certified (s.f.) | previous record (s.f.) | run A / run B / certificate |
|---|---|---|---|---|---|---|
| Ne | 10 | 500 | 504.6 | 504.6 | 30 (Yamamoto & Hatano 2026) | 26 min / 41 min / 17 min |
| Mg | 12 | 100 | 103.7 | 103.7 | 30 (Yamamoto & Hatano 2026) | 86 s / 4 min / 3 min |
| Ar | 18 | 100 | 105.4 | 105.4 | 30 (Yamamoto & Hatano 2026) | 13 min / 9 min / 6 min |
| Ca | 20 | 100 | 103.5 | 103.5 | 25 (Cinal 2020) | 7 min / 15 min / 11 min |
| Zn | 30 | 100 | 105.0 | 105.0 | 25 (Cinal 2020) | 24 min / 51 min / 16 min |
| Kr | 36 | 100 | 105.8 | 105.8 | 25 (Cinal 2020) | 37 min / 43 min / 25 min |
| Sr | 38 | 100 | 103.9 | 103.9 | 25 (Cinal 2020) | 1.8 h / 1.5 h / 71 min |
| Pd | 46 | 100 | 106.5 | 106.6 | 25 (Cinal 2020) | 1.9 h / 1.6 h / 54 min |
| Cd | 48 | 100 | 105.2 | 105.2 | 25 (Cinal 2020) | 1.6 h / 2.1 h / 1.6 h |
| Xe | 54 | 100 | 105.9 | 105.9 | 25 (Cinal 2020) | 3.9 h / 3.2 h / 2.1 h |

Every certified figure, the orbital energies and the exchange-induced nodes of the inner orbitals are in
`paper2/frost_pd.pdf` and `paper2/data/runs.json`.

## Orbital quality (`properties.py`)

The energy is stationary, so it only sees orbital errors squared; properties see them to first order. In
the Λ-basis record (66-figure energy) the He cusp ratio has 30 correct figures, ρ(0) 32, ⟨r⁻²⟩ 35. Here:

- **Cusp exact.** The Frobenius recurrence gives f₁ = −Z f₀ with no rounding for Z = 2, 4, so the certified
  orbitals' f'(0)/f(0) is the rational −Z (both Be orbitals too).
- **He, runs B/C (500 digits):** ρ(0), ⟨r⁻²⟩, ⟨r⁻¹⟩ agree to 500 s.f., ⟨r⟩ 498, ⟨r¹²⟩ 475. Every figure
  Hatano & Yamamoto print (ρ(0) and ⟨r⁻²⟩…⟨r⁹⟩, 32–65 s.f.) agrees.
- **Pointwise error in closed form.** In the Coulomb region the wall adds the growing Whittaker solution:
  f_R/f − 1 = −W(2κR)M(2κr)/[M(2κR)W(2κr)] ∝ (R/r)^{2σ} e^{−2κ(R−r)}. This matches the measured run B/C
  difference to ~1e-6 relative for r ≥ 10 bohr: 500 s.f. at the nucleus, ~250 at R/2.
- **⟨rᵏ⟩ loses ~2 s.f. per power of r**, because the wall changes f²r² uniformly over the Coulomb region
  (W·M → const); δ⟨rᵏ⟩ ∝ R^{k+1}/(k+1) − R⟨rᵏ⟩ fits k = 1…12 to 0.1 s.f.
- **Be:** density properties (ρ(0), Σ⟨rᵏ⟩, virial ratio) agree between the 500-digit runs to 478–498 s.f.
  (Hatano & Yamamoto: 20–55); the individual 1s/2s moments only to ~|S₁₂| (315–321 s.f., S₁₂ = 4.9e-315;
  127–133 at 200 digits, S₁₂ = 2.6e-127), consistent with a rotation within the occupied space that leaves
  the density unchanged to first order. Use per-orbital normalisation: the P = S⁻¹ kinetic energy is also
  only good to ~S₁₂.

## Isoelectronic series, Z = 1–10

Certified upper bounds (run at `--digits 100`, 10–16 s each), checked against an independent oracle run
(`--digits 110 --step 0.25 --guard 30`, larger wall). Every bound lies above its reference. Full 100-digit
values are in `he_like_series.txt`.

| Z | ion | E_HF (hartree) | agreement (s.f.) |
|---|---|---|---|
| 1 | H⁻ | −0.4879297343708517486928383… | 107.8 |
| 2 | He | −2.861679995612238878775543… | 104.2 |
| 3 | Li⁺ | −7.236415201452256247682907… | 103.7 |
| 4 | Be²⁺ | −13.61129943061910903495583… | 103.5 |
| 5 | B³⁺ | −21.98623446682434301967704… | 103.3 |
| 6 | C⁴⁺ | −32.36119287571801233131520… | 103.3 |
| 7 | N⁵⁺ | −44.73616396494247312978203… | 103.3 |
| 8 | O⁶⁺ | −59.11114270191365385859847… | 103.2 |
| 9 | F⁷⁺ | −75.48612640626050983759562… | 103.2 |
| 10 | Ne⁸⁺ | −93.86111351923133815497816… | 103.2 |

## H⁻ ("unbound in HF")

That phrase means E_HF(H⁻) = −0.48793 lies above the H(1s) + e⁻ threshold (−0.5): HF predicts a negative
electron affinity. The closed-shell RHF problem itself still has a normalisable solution, with orbital energy
ε = −0.046222445627687 < 0 and f ~ e^{−0.304 r}/r, because the escape channel (one electron in 1s, one at
infinity) is not a closed-shell determinant. Only unrestricted HF collapses (infimum −0.5, not attained).
The run shows it directly: ε is fixed by R ≈ 70 and does not move as the wall goes out to 409 bohr, which a
wall-supported unbound state could not do. The certificate is an upper bound on the infimum of the
closed-shell functional either way. H⁻ needs a larger wall (small κ, no Coulomb tail); the code picks R from κ.

## Why not a Laguerre basis

The HF orbital's nearest singularities are a complex pair at r* ≈ −1.099 ± 0.577i (fitted from the
626th-order Frobenius series; the radius of convergence is |r₀ − r*| along the inner region). A single-exponent
expansion (Laguerre, Λ functions, rᵏe^{−ζr}) must resolve both these and the e^{−κr} tail with one ζ, so the best
ζ grows as N^{1/3} and the error falls only as exp(−cN^{2/3}). That estimate gives ζ_opt = 4.98 at N = 150,
against Hatano & Yamamoto's optimised 5.1448, and N ≈ 2800 for 500 digits. Local Taylor steps converge
geometrically instead.

## Reproduce

    python oracle.py --digits 120 --out he120.json      # ~13 s
    python certify.py he120.json                        # a few seconds
    for Z in $(seq 1 10); do
        python oracle.py --Z $Z --digits 100 --out z$Z.json && python certify.py z$Z.json
        python oracle.py --Z $Z --digits 110 --step 0.25 --guard 30 --out z${Z}ref.json
    done
    python oracle_ns.py --Z 4 --digits 100 --out be100.json && python certify.py be100.json   # ~10 s
    python oracle_ns.py --Z 4 --digits 110 --step 0.25 --guard 30 --out be100ref.json
    python oracle_ns.py --Z 10 --orbitals "1s 2s 2p" --digits 100 --out ne100.json && python certify.py ne100.json   # ~35 s
    paper2/run_all.sh OUT && python paper2/make_paper2.py --extract OUT   # every run of paper II, about 30 h
    python test_hf_taylor.py

## License

MIT, see `LICENSE`.
