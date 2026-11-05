"""hfatom against the certified HF limits of papers I and II: every energy must be the certified limit correctly
rounded to double precision (the nearest double), so the error printed in ulp of E is at most 0.5.
    python3 test_hfatom.py      (compiles hfatom.c when the binary is missing or older than its source)"""
import math
import os
import re
import subprocess
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, BIN = os.path.join(HERE, "hfatom.c"), os.path.join(HERE, "hfatom")
REF = {  # certified upper bounds, every printed figure confirmed by an independent run
    ("1", "1s"): "-0.48792973437085174869283838",                 # H-
    ("He", None): "-2.8616799956122388787755437400",
    ("3", "1s"): "-7.2364152014522562476829074240",               # Li+
    ("Be", None): "-14.573023168316399582411447359",
    ("Ne", None): "-128.54709810938204237647578134",
    ("Mg", None): "-199.61463642450671047474163610",
    ("Ar", None): "-526.81751280272335501876609353",
    ("Ca", None): "-676.75818592451345494805575176",
    ("Zn", None): "-1777.8481161913610697784425032",
    ("Kr", None): "-2752.0549773455764543944491382",
    ("Sr", None): "-3131.5456864385571844341241539",
    ("Pd", None): "-4937.9210240701015729322898038",
    ("Cd", None): "-5465.1331425300864336728186312",
    ("Xe", None): "-7232.1383638718408882092892165",
}


def main():
    if not os.path.exists(BIN) or os.path.getmtime(BIN) < os.path.getmtime(SRC):
        subprocess.run(["cc", "-O3", "-mcpu=native", "-o", BIN, SRC, "-framework", "Accelerate"], check=True)
    bad = 0
    for (atom, orb), ref in REF.items():
        out = subprocess.run([BIN, atom] + ([orb] if orb else []), capture_output=True, text=True, check=True).stdout
        E = float(re.search(r"E\s+= (\S+)", out).group(1))       # printed with %.17g: the exact double
        ms = float(re.search(r"time\s+= (\S+)", out).group(1))
        err = float(Decimal(E) - Decimal(ref))
        ok = E == float(Decimal(ref))                            # float(Decimal) rounds correctly
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {atom:3s} E = {E!r:22s} error {err:+.1e} = {err / math.ulp(E):+.2f} ulp  {ms:.1f} ms")
    assert not bad, f"{bad} energies are not the correctly rounded limit"


if __name__ == "__main__":
    main()
