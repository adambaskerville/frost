# FROST fast: Hartree–Fock limits in double precision

`hfatom` computes the restricted Hartree–Fock limit of a closed-shell atom in double precision, in milliseconds.
For all fourteen atoms and ions with a certified limit from papers I and II (H⁻, He, Li⁺, Be, Ne, Mg, Ar, Ca, Zn,
Kr, Sr, Pd, Cd, Xe), the energy it returns is the certified limit correctly rounded to double precision: the
nearest double, within half a unit in the last place.

    cc -O3 -mcpu=native -o hfatom hfatom.c -framework Accelerate
    ./hfatom Xe                       # neutral closed-shell ground state, default target 1e-12
    ./hfatom 54 "1s 2s 2p 3s 3p 3d 4s 4p 4d 5s 5p" --tol 1e-9
    ./hfatom 1 1s -v                  # H-, with grid details
    python3 test_hfatom.py            # against the certified limits

## Python

`frostfast.py` binds the same code as a shared library through `ctypes` (standard library only). It compiles
`libhfatom.dylib` on first use, and again whenever `hfatom.c` changes.

```python
import frostfast
r = frostfast.atom("Xe")                    # or atom(54), atom(54, "1s 2s ... 5p", tol=1e-9)
r["E"], r["eps"]["5p"], r["virial"]         # energy, orbital energies, -V/T - 2
r["N"], r["a"], r["R"], r["seconds"]        # the grid that was used, and the time it took
```

`atom(Z, orbitals=None, tol=1e-12, N=0, a=0.0, R=0.0)`: `orbitals` defaults to the neutral closed-shell ground
state, and zero for N, a or R means automatic. It raises `ValueError` when there is no closed-shell configuration or
the input is invalid. A call costs the same as the CLI without the process start, so 100 neon atoms take 0.2 s.

## Method

The orbitals P = rR and the multipole potentials Y^k are represented by their values at N right Gauss–Radau nodes
t ∈ (0, 2], mapped to r = a(e^{St/2} − 1) with the wall R at t = 2. The basis is the regularised Lagrange basis
(t/t_j) ℓ_j(t), which vanishes at r = 0. Every integral is Radau quadrature at the nodes, so the energy is a
polynomial functional of the nodal values with a diagonal mass matrix. The potentials are the discrete Galerkin
solutions of the radial Poisson equation, with the exact exterior condition R Y′ + kY = 0 at the wall.

The orbitals are the stationary point of this functional under the norm constraints. As in FROST, all orbitals share
one wall, so the canonical orbitals of each l are eigenvectors of one Fock matrix and are orthogonal without
constraints. A Thomas–Fermi guess and a few SCF+DIIS steps bring the orbitals into Newton's basin. Two Newton steps
on the Lagrangian then finish the job, with all orbitals and orbital energies solved at once and the exact Hessian,
and the orbitals of each l are Löwdin-orthonormalised before the energy is evaluated.

Four measured facts decide the precision:

- **A variational discretisation, not collocation.** With collocation the computed energy is first-order sensitive
  to rounding noise in the orbitals (1e-10 of noise moved Ne by 1e-9 Eh). At the stationary point of a discrete
  functional the orbitals enter only at second order.
- **P, not FROST's f = P/r^{l+1}.** On a global basis the weights r^{2l+2} span 1e38 for f shells, which destroys
  the lowest eigenvalues in double precision.
- **Exact operators, but only in the energy.** Everything that defines the functional is a first-order
  perturbation of the energy once it is rounded: the differentiation matrix (rounding it put Ne 5e-11 Eh off), the
  nodes (1e-14 node errors gave 1e-12 weight errors), the weights and the mapped mesh (a scatter of several ulp in
  Xe). Newton therefore runs on plain double-precision operators, and only the final energy is evaluated exactly:
  mesh, weights and differentiation matrix in double-double, every product formed exactly, and the variational
  form 2ρᵀY − YᵀKY/(2k+1) for each Coulomb and exchange integral. Operator errors then enter the energy at second
  order. The Radau nodes are solved by Newton with one double-double step.
- **Orthonormal orbitals.** The energy expression assumes orthonormal orbitals. A Newton step leaves norm and
  overlap defects of the order of its square, which Löwdin orthonormalisation removes; the rounding-level norm
  defect that remains is still first order (ε_1s of Xe is −1224 Eh), so it is corrected with norms in double-double,
  ΔE = −Σ_a occ_a ε_a (⟨a|a⟩ − 1).

## Grid

The grid is a = 3/Z and N = ⌈10 ln Z + 2d + 3⌉ for a target of 10^-d Eh, plus (Z − 54)/2.5 beyond Xe. This rule was
calibrated on He–Xe against the certified limits. The wall is R = 40, enlarged when the wall law
E_R − E_HF ≈ 3·10⁴ exp(−2κR) requires it (κ is taken from the highest orbital; H⁻ moves to R = 63). An
under-resolved grid fails abruptly rather than gradually, so the virial theorem serves as the check: if
|−V/T − 2| > 1e-9 (resolved grids give about 1e-14), N is enlarged by 15% and the solve repeated. Hg needed this
before the rule was extended past Xe.

## Accuracy and speed

Energies against the certified limits, with the default target (Apple M4 Pro; the time is that of the library
call, `-v` prints it; a process start adds about 3 ms):

| atom | E (hfatom) | error (Eh) | ulp of E | time |
|---|---|---|---|---|
| H⁻ | −0.48792973437085174 | +6.7e−18 | +0.12 | 1.4 ms |
| He | −2.861679995612239 | −7.9e−17 | −0.18 | 0.6 ms |
| Li⁺ | −7.236415201452256 | +4.4e−16 | +0.50 | 0.6 ms |
| Be | −14.5730231683164 | −4.1e−16 | −0.23 | 1.0 ms |
| Ne | −128.54709810938203 | +8.2e−15 | +0.29 | 2.1 ms |
| Mg | −199.6146364245067 | +1.1e−14 | +0.40 | 2.5 ms |
| Ar | −526.8175128027234 | −4.5e−14 | −0.39 | 3.2 ms |
| Ca | −676.7581859245134 | +1.4e−14 | +0.12 | 4.5 ms |
| Zn | −1777.848116191361 | +8.3e−14 | +0.37 | 6.2 ms |
| Kr | −2752.0549773455764 | +6.5e−14 | +0.14 | 6.7 ms |
| Sr | −3131.5456864385574 | −1.8e−13 | −0.40 | 7.5 ms |
| Pd | −4937.921024070101 | +2.3e−13 | +0.26 | 10.1 ms |
| Cd | −5465.133142530087 | −4.1e−13 | −0.45 | 10.4 ms |
| Xe | −7232.138363871841 | +1.5e−13 | +0.16 | 12.6 ms |

Li⁺'s certified limit lies almost exactly halfway between two doubles; hfatom returns the nearer one. Beyond Xe there
are no certified limits; Ba, Hg and Rn agree with published HF limits to within 2 ulp, and Yb to all of its
published figures (15–37 ms).

The comparison with HelFEM, x2dhf and Gaussian basis sets, all on the same machine, is in
`frost-data/fast/bench_atoms.md` and `frost-data/fast/bench_other.py`.
