"""Number macros, tables, appendix and figure for paper 2 (frost_pd.tex).

Each atom has a pair of runs: A (target D digits, step c = 0.15, guard 40) with its certificate, and an independent
B (D + 10 digits, so a larger wall; c = 0.25; guard 30), made by
    python ../oracle_ns.py --Z Z --orbitals "..." --digits D --out XA.json
    python ../certify.py XA.json --print-digits D+15 > XA.cert
    python ../oracle_ns.py --Z Z --orbitals "..." --digits D+10 --step 0.25 --guard 30 --out XB.json
run_all.sh makes all of them.
The run files hold every Taylor coefficient (up to ~300 MB); data/ keeps what the paper uses:
    python make_paper2.py --extract DIR   refresh data/runs.json and data/tails.json from DIR/{name}{a,b}.json, .cert
    python make_paper2.py                 -> numbers2.tex, tab_atoms.tex, tab_energies.tex, app_digits2.tex,
                                             fig_tails.pdf
"""
import json
import math
import re
import sys
from pathlib import Path

import gmpy2
import matplotlib
from gmpy2 import mpfr, mpz

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
gmpy2.get_context().precision = 4000

# atom: (production pairs, best first, configuration)
ATOMS = {
    "Ne": (("ne500",), r"[He]$2s^22p^6$"),
    "Mg": (("mg100",), r"[Ne]$3s^2$"),
    "Ar": (("ar100",), r"[Ne]$3s^23p^6$"),
    "Ca": (("ca100",), r"[Ar]$4s^2$"),
    "Zn": (("zn100",), r"[Ar]$3d^{10}4s^2$"),
    "Kr": (("kr100",), r"[Ar]$3d^{10}4s^24p^6$"),
    "Sr": (("sr100",), r"[Kr]$5s^2$"),
    "Pd": (("pd100",), r"[Kr]$4d^{10}$"),
    "Cd": (("cd100",), r"[Kr]$4d^{10}5s^2$"),
    "Xe": (("xe100",), r"[Kr]$4d^{10}5s^25p^6$"),
}
# Cinal, J. Math. Chem. 58, 1571 (2020), Table 3: -E_tot (25 s.f.) and -eps_HOMO (22-25 s.f.), quadruple precision
CINAL = {
    "Ne": ("128.5470981093820423764758", "0.850409650347267227432802"),
    "Mg": ("199.6146364245067104747416", "0.253052582006473108897914"),
    "Ar": ("526.8175128027233550187661", "0.591017409350350745054567"),
    "Ca": ("676.7581859245134549480558", "0.19552969246777618100484"),
    "Zn": ("1777.848116191361069778443", "0.29250714640279143840312"),
    "Kr": ("2752.054977345576454394449", "0.52418668102659565978801"),
    "Sr": ("3131.545686438557184434124", "0.17845644685143475066359"),
    "Pd": ("4937.921024070101572932290", "0.33600201366577555157073"),
    "Cd": ("5465.133142530086433672819", "0.26485591255388706830333"),
    "Xe": ("7232.138363871840888209289", "0.45729005785183302646846"),
}
PREV = {el: (30, r"\cite{Yamamoto2026}") if el in ("Ne", "Mg", "Ar") else (25, r"\cite{Cinal2020}") for el in ATOMS}
KEYS = ("Z", "orbitals", "l", "digits", "order", "step", "segments", "precision_bits", "seconds", "E", "E_virial",
        "eps", "S_offdiag")


def seconds(text):
    """Wall time from zsh's `time` line ('... 2:08:27.57 total', '... 22:19.66 total' or '... 41.2 total')."""
    m = re.search(r"(?:(?:(\d+):)?(\d+):)?(\d+(?:\.\d+)?) total", text)
    return None if not m else 3600 * int(m.group(1) or 0) + 60 * int(m.group(2) or 0) + float(m.group(3))


def extract(src):
    runs = {}
    for el, (names, _) in ATOMS.items():
        rec = {}
        # the highest-precision pair whose run A has been certified (else the first with a run A at all)
        name = next((nm for nm in names if (src / f"{nm}a.cert").exists() and "E_cert" in (src / f"{nm}a.cert").read_text()),
                    next((nm for nm in names if (src / f"{nm}a.json").exists()), names[0]))
        for ab in "ab":
            p = src / f"{name}{ab}.json"
            if p.exists():
                d = json.load(open(p))
                rec[ab] = {k: d[k] for k in KEYS} | {"R": d["walls"][0]}
        c = src / f"{name}a.cert"
        if c.exists() and "E_cert" in (text := c.read_text()):
            rec["cert"] = {"E_upper": re.search(r"E_cert\s+<=\s+(\S+)", text).group(1), "seconds": seconds(text)}
        for ab in "ab":
            if (src / f"{name}{ab}.json").exists():
                rec[f"nodes_{ab}"] = nodes(json.load(open(src / f"{name}{ab}.json")))
        runs[el] = rec
    json.dump(runs, open(DATA / "runs.json", "w"), indent=1)
    # |P_a(r)| = r^(l+1) |f_a(r)| of krypton run A, 6 points per Taylor element, for the figure
    d = json.load(open(src / "kr100a.json"))  # the production run; the figure shows r <= 40 bohr
    dy = lambda pr: mpfr(mpz(pr[0])) * mpfr(2) ** int(pr[1])  # noqa: E731
    rows = []
    for r0, h, fs in d["elements"]:
        r0, h = dy(r0), dy(h)
        for s in (0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6):
            r = r0 + h * s
            if r == 0:
                continue
            row = [float(r)]
            for f, l in zip(fs, d["l"]):
                v = mpfr(0)
                for c in reversed(f):
                    v = v * (r - r0) + dy(c)
                row.append(float(gmpy2.log10(abs(v * r ** (l + 1)))) if v else None)
            rows.append(row)
    json.dump({"orbitals": d["orbitals"], "eps": d["eps"], "R": d["walls"][0], "rows": rows},
              open(DATA / "tails.json", "w"))


def nodes(d, bits=400):
    """Sign changes of each radial function f_a (so of P_a) inside the wall: found on 8 samples per Taylor element
    and refined by bisection on the element's polynomial (exact dyadic coefficients) to ~bits."""
    with gmpy2.local_context(gmpy2.get_context(), precision=bits):
        dy = lambda pr: mpfr(mpz(pr[0])) * mpfr(2) ** int(pr[1])  # noqa: E731
        els = [(dy(r0), dy(h), [[dy(c) for c in f] for f in fs]) for r0, h, fs in d["elements"]]
        out = []
        for a in range(len(d["orbitals"])):
            found = []
            for r0, h, fs in els:
                f = fs[a]
                ev = lambda x: sum(c * x ** k for k, c in enumerate(f))  # noqa: E731
                xs = [h * s / 8 for s in range(9)]
                vs = [ev(x) for x in xs]
                for (x0, v0), (x1, v1) in zip(zip(xs, vs), zip(xs[1:], vs[1:])):
                    if v0 and v1 and (v0 > 0) != (v1 > 0) and r0 + x1 < d["walls"][0] - 0.5:
                        lo, hi = x0, x1
                        for _ in range(bits):
                            mid = (lo + hi) / 2
                            if (ev(mid) > 0) == (v0 > 0):
                                lo = mid
                            else:
                                hi = mid
                        found.append(format(r0 + (lo + hi) / 2, ".100f"))
            out.append(found)
        return out


def sf(a, b):
    """Significant figures to which a and b agree."""
    a, b = mpfr(a), mpfr(b)
    return float("inf") if a == b else float(-gmpy2.log10(abs(a - b) / abs(b)))


def trunc(s, n):
    """Decimal string s truncated (towards zero) to n significant figures. For a negative energy this raises the
    value, so a truncated upper bound is still an upper bound."""
    sign, s = ("-", s[1:]) if s.startswith("-") else ("", s)
    ip, fp = s.split(".")
    lead = len(ip.lstrip("0")) if ip.lstrip("0") else -(len(fp) - len(fp.lstrip("0")))
    return sign + ip + "." + fp[: n - lead]


def grouped(s, every=3, brk=False):
    """1234.5678901 -> 1234.567\\,890\\,1 (thin spaces in the fraction; optional line-break points)."""
    ip, fp = s.split(".")
    sep = r"\,\allowbreak " if brk else r"\,"
    return ip + "." + sep.join(fp[i:i + every] for i in range(0, len(fp), every))


def sci(x, d=0):
    """LaTeX a x 10^b."""
    x = mpfr(x)
    if not x:
        return "0"
    m, e = format(x, f".{d}e").split("e")  # gmpy2 formats far below float range
    return rf"{m}\times10^{{{int(e)}}}"


def split_lines(s, lead, per=21):
    """A long negative decimal as aligned lines of `per` fraction digits (for a split environment)."""
    ip, fp = s.lstrip("-").split(".")
    lines = [grouped("0." + fp[i:i + per]).split(".")[1] for i in range(0, len(fp), per)]
    return rf"\mathord{{-}}{ip}." + rf"\\ &\hphantom{{{lead}}}".join(lines)


def main():
    if "--extract" in sys.argv:
        extract(Path(sys.argv[sys.argv.index("--extract") + 1]))
    runs = json.load(open(DATA / "runs.json"))
    macros, atoms, energies, digits = [], [], [], []
    ok_cinal, done = [], []
    for el, (name, conf) in ATOMS.items():
        rec = runs.get(el, {})
        a, b, cert = rec.get("a"), rec.get("b"), rec.get("cert")
        prev, ref = PREV[el]
        if not (a and cert):
            atoms.append(rf"{el} & {conf} & \multicolumn{{8}}{{c}}{{in progress}} & {prev}~{ref} \\")
            continue
        Eu = cert["E_upper"]
        ab = sf(a["E"], b["E"]) if b else None
        cb = sf(Eu, b["E"]) if b else None
        above = b is not None and mpfr(Eu) > mpfr(b["E"])
        nsf = math.floor(min(ab, cb)) if b else None
        eps_homo = format(max(map(mpfr, a["eps"])), ".60f")  # least negative, as a plain decimal
        c_e, c_h = CINAL[el]
        e25 = sf(Eu, "-" + c_e)
        h25 = sf(eps_homo, "-" + c_h)
        if e25 > 24 and h25 > 21:
            ok_cinal.append(el)
        if b:
            done.append(el)
        smax = max((abs(mpfr(s)) for s in a["S_offdiag"]), default=mpfr(0))
        atoms.append(
            rf"{a['Z']} & {el} & {conf} & {a['digits']} & {a['R']:.2f} & {a['segments']} & {a['order']} & "
            rf"{a['precision_bits']} & {f'{ab:.1f}' if b else '--'} & {f'{cb:.1f}' if b else '--'} & "
            rf"{prev}~{ref} \\")
        shown = trunc(Eu, min(nsf, 45)) if nsf else trunc(Eu, 30)
        energies.append(rf"{el} & ${grouped(shown)}\ldots$ & ${grouped(trunc(eps_homo, 25))}\ldots$ \\")
        if nsf:
            digits.append(rf"\noindent {el} ({nsf} s.f.): $\EHF\le{grouped(trunc(Eu, nsf), brk=True)}$\par\medskip")
        tag = el
        macros += [rf"\newcommand{{\{tag}Nsf}}{{{nsf if nsf else '--'}}}",
                   rf"\newcommand{{\{tag}AB}}{{{f'{ab:.1f}' if b else '--'}}}",
                   rf"\newcommand{{\{tag}R}}{{{a['R']:.2f}}}",
                   rf"\newcommand{{\{tag}Smax}}{{{sci(smax)}}}",
                   rf"\newcommand{{\{tag}Above}}{{{'yes' if above else 'no'}}}"]
    rows = []
    for el, rec in runs.items():
        if "nodes_a" not in rec:
            continue
        a, nb = rec["a"], rec.get("nodes_b")
        for i, (o, l, rs) in enumerate(zip(a["orbitals"], a["l"], rec["nodes_a"])):
            extra = rs[int(o[:-1]) - l - 1:]  # beyond the hydrogenic count: the outermost ones, in the tail
            for j, r in enumerate(extra):
                rb = nb[i][int(o[:-1]) - l - 1 + j] if nb and len(nb[i]) == len(rs) else None
                agree = min(sf(r, rb), 99) if rb else 0
                shown = min(30, math.floor(agree) - 1) if rb else 10
                # one row per node: name the atom and the orbital on their first row only
                lead = (el if not rows or rows[-1][0] != el else "", f"${o}$" if j == 0 else "")
                rows.append((el, rf"{lead[0]} & {lead[1]} & ${grouped(trunc(r, shown))}\ldots$ & "
                                 rf"{'--' if not rb else math.floor(agree) if agree < 99 else '$>$99'}"))
    # two panels side by side (a full-width table, half as tall), split between atoms nearest the middle
    half = min((i for i in range(1, len(rows)) if rows[i][0] != rows[i - 1][0]), key=lambda i: abs(2 * i - len(rows)))
    left, right = [t for _, t in rows[:half]], [t for _, t in rows[half:]]
    pad = lambda side, i: side[i] if i < len(side) else " & & & "  # noqa: E731
    (HERE / "tab_nodes.tex").write_text("\n".join(f"{pad(left, i)} & & {pad(right, i)} \\\\"
                                                  for i in range(max(len(left), len(right)))) + "\n")
    kr = runs["Kr"]["a"]
    k1 = math.sqrt(-2 * min(float(e) for e in kr["eps"]))
    nl = len(kr["l"])
    npots = sum(min(kr["l"][i], kr["l"][j]) + 1 for i in range(nl) for j in range(i, nl))
    n0 = nl + npots
    macros += [rf"\newcommand{{\KrKappaOne}}{{{k1:.2f}}}",
               rf"\newcommand{{\KrGrowth}}{{{k1 * kr['R'] / math.log(10):.0f}}}",
               rf"\newcommand{{\KrPots}}{{{npots}}}",
               rf"\newcommand{{\KrM}}{{{2 * n0}}}",
               rf"\newcommand{{\KrSeg}}{{{kr['segments']}}}",
               rf"\newcommand{{\KrUnknowns}}{{{n0 + (kr['segments'] - 1) * 2 * n0 + nl}}}",
               rf"\newcommand{{\KrNodes}}{{{', '.join(f'{float(r):.2f}' for r in runs['Kr']['nodes_a'][0])}}}"]
    tl = json.load(open(DATA / "tails.json"))  # krypton 1s at r = 10 against its own decay from its maximum
    i1 = tl["orbitals"].index("1s")
    rp, yp = max(((r[0], r[i1 + 1]) for r in tl["rows"] if r[i1 + 1] is not None), key=lambda p: p[1])
    r10 = min((r for r in tl["rows"] if r[i1 + 1] is not None), key=lambda r: abs(r[0] - 10))
    own = yp - math.sqrt(-2 * float(tl["eps"][i1])) * (r10[0] - rp) / math.log(10)
    macros += [rf"\newcommand{{\KrOneSGap}}{{{r10[i1 + 1] - own:.0f}}}",
               rf"\newcommand{{\KrOneSAtTen}}{{{r10[i1 + 1] - yp:.0f}}}",
               rf"\newcommand{{\KrOneSOwnAtTen}}{{{own - yp:.0f}}}"]
    heavy = [m for m in (math.floor(min(sf(r["a"]["E"], r["b"]["E"]), sf(r["cert"]["E_upper"], r["b"]["E"])))
                         for el, r in runs.items() if el not in ("Ne", "Mg", "Ar") and "b" in r and "cert" in r)]
    macros += [rf"\newcommand{{\HeavyMin}}{{{min(heavy)}}}", rf"\newcommand{{\HeavyMax}}{{{max(heavy)}}}"]
    certh = [r["cert"]["seconds"] / 3600 for el, r in runs.items() if el not in ("Ne", "Mg", "Ar") and "cert" in r]
    macros += [rf"\newcommand{{\CaTimeA}}{{{runs['Ca']['a']['seconds'] / 60:.0f}}}",
               rf"\newcommand{{\KrTimeA}}{{{kr['seconds'] / 60:.0f}}}",
               rf"\newcommand{{\XeTimeA}}{{{runs['Xe']['a']['seconds'] / 3600:.1f}}}",
               rf"\newcommand{{\CertMinH}}{{{min(certh):.1f}}}", rf"\newcommand{{\CertMaxH}}{{{max(certh):.1f}}}"]
    ne = runs["Ne"]
    neu = ne["cert"]["E_upper"]
    nensf = math.floor(min(sf(ne["a"]["E"], ne["b"]["E"]), sf(neu, ne["b"]["E"])))
    macros += [
        rf"\newcommand{{\NeEsplit}}{{{split_lines(trunc(neu, 60), '-128.')}}}",
        rf"\newcommand{{\NeNsfAll}}{{{nensf}}}",
        rf"\newcommand{{\NeCertTime}}{{{ne['cert']['seconds'] / 60:.0f}}}",
        rf"\newcommand{{\NeTimeA}}{{{ne['a']['seconds'] / 60:.0f}}}",
        rf"\newcommand{{\NeTimeB}}{{{ne['b']['seconds'] / 60:.0f}}}",
        rf"\newcommand{{\NAtomsDone}}{{{len(done)}}}",
        rf"\newcommand{{\CinalAgree}}{{{', '.join(ok_cinal)}}}",
        rf"\newcommand{{\NCinalAgree}}{{{len(ok_cinal)}}}",
        rf"\newcommand{{\NeMaxOrder}}{{{ne['a']['order']}}}",
    ]
    (HERE / "numbers2.tex").write_text("% generated by make_paper2.py\n" + "\n".join(macros) + "\n")
    (HERE / "tab_atoms.tex").write_text("\n".join(atoms) + "\n")
    (HERE / "tab_energies.tex").write_text("\n".join(energies) + "\n")
    (HERE / "app_digits2.tex").write_text("\n".join(digits) + "\n")

    # figure: krypton's inner orbitals ride on the 4p tail. Solid: computed |P_a|; dotted: the orbital's own decay
    # exp(-kappa_a r), kappa_a = sqrt(-2 eps_a), drawn from its maximum. Categorical slots 1-3 (validated, CVD-safe);
    # identity is also carried by black direct labels, so the figure reads in greyscale.
    t = json.load(open(DATA / "tails.json"))
    rows, X = t["rows"], 40
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name, c in (("1s", "#2a78d6"), ("3d", "#eb6834"), ("4p", "#1baf7a")):
        i = t["orbitals"].index(name)
        pts = [(r[0], r[i + 1]) for r in rows if r[i + 1] is not None and r[0] <= X]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=1.4, solid_capstyle="round")
        ax.annotate(rf"${name}$", (pts[-1][0], pts[-1][1]), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8)
        if name != "4p":
            k = math.sqrt(-2 * float(t["eps"][i]))
            rp, yp = max(pts, key=lambda p: p[1])
            r_end = min(X, rp + (yp + 45) * math.log(10) / k)
            ax.plot([rp, r_end], [yp, yp - k * (r_end - rp) / math.log(10)], color=c, lw=1.0, ls=(0, (1, 1.5)))
            ax.annotate(rf"$e^{{-\kappa_{{{name}}}r}}$", (r_end, -45), xytext=(2, 6), textcoords="offset points",
                        fontsize=7, color="0.25")
    i1 = t["orbitals"].index("1s")
    node = min((r for r in rows if 1.0 < r[0] < 1.6 and r[i1 + 1] is not None), key=lambda r: r[i1 + 1])
    ax.annotate("node", (node[0], node[i1 + 1]), xytext=(9, -14), textcoords="offset points", fontsize=7,
                color="0.25", arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))
    ax.plot([], [], color="0.2", lw=1.4, label="computed")
    ax.plot([], [], color="0.2", lw=1.0, ls=(0, (1, 1.5)), label="own decay")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_xlabel(r"$r$ (bohr)")
    ax.set_ylabel(r"$\log_{10}|P_a(r)|$")
    ax.set_xlim(0, X + 4)
    ax.set_ylim(-45, 2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_tails.pdf")


if __name__ == "__main__":
    main()
