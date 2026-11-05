"""Python bindings for hfatom: closed-shell atomic Hartree-Fock limits in double precision.

    import frostfast
    r = frostfast.atom("Xe")                    # or frostfast.atom(54, "1s 2s 2p ...", tol=1e-9)
    r["E"], r["eps"]["5p"], r["virial"], r["seconds"]

The shared library is compiled from hfatom.c on first use (and whenever the source is newer); ctypes only.
"""
import ctypes
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC, _LIB = os.path.join(_HERE, "hfatom.c"), os.path.join(_HERE, "libhfatom.dylib")


def _load():
    if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
        subprocess.run(["cc", "-O3", "-mcpu=native", "-DHFATOM_LIB", "-dynamiclib", "-o", _LIB, _SRC,
                        "-framework", "Accelerate"], check=True)
    lib = ctypes.CDLL(_LIB)
    lib.hfatom_run.restype = ctypes.c_int
    lib.hfatom_run.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_double, ctypes.c_int, ctypes.c_double,
                               ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    lib.hfatom_config.restype = lib.hfatom_symbol.restype = ctypes.c_char_p
    lib.hfatom_config.argtypes = lib.hfatom_symbol.argtypes = [ctypes.c_int]
    return lib


_lib = _load()
_Z = {_lib.hfatom_symbol(z).decode(): z for z in range(1, 87)}


def atom(Z, orbitals=None, tol=1e-12, N=0, a=0.0, R=0.0):
    """Restricted HF limit of a closed-shell atom. Z: number or symbol; orbitals: occupied subshells such as
    "1s 2s 2p" (default: the neutral closed-shell ground state); tol: target error in hartree; N, a, R: grid
    (0 = automatic). Returns E, the orbital energies, -V/T - 2, the grid used, the step counts and the wall time."""
    Z = _Z[Z] if isinstance(Z, str) else int(Z)
    if orbitals is None:
        cfg = _lib.hfatom_config(Z)
        if cfg is None:
            raise ValueError(f"Z={Z} has no closed-shell ground state; give the orbitals")
        orbitals = cfg.decode()
    out = (ctypes.c_double * 40)()
    n = _lib.hfatom_run(Z, orbitals.encode(), tol, N, a, R, out)
    if n < 0:
        raise ValueError(f"bad input: Z={Z}, orbitals={orbitals!r}, tol={tol}")
    return {"E": out[0], "virial": out[1], "eps": dict(zip(orbitals.split(), out[8:8 + n])),
            "N": int(out[2]), "a": out[3], "R": out[4], "scf": int(out[5]), "newton": int(out[6]),
            "seconds": out[7]}


if __name__ == "__main__":
    for z in ("He", "Ne", "Ar", "Kr", "Xe"):
        r = atom(z)
        print(f"{z:2s}  E = {r['E']:.15f}  -V/T-2 = {r['virial']:+.1e}  {1e3 * r['seconds']:.2f} ms")
