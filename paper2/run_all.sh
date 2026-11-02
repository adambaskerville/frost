#!/bin/zsh
# Every run of paper II, one after another on all cores (about 30 h on an Apple M4 Pro): run A (certified), the
# independent run B (10 more digits, c = 0.25, guard 30) and the certificate of A. Finished files are skipped, so an
# interrupted batch can be restarted. Usage: paper2/run_all.sh OUTDIR, then python paper2/make_paper2.py --extract OUTDIR
O=${1:?usage: run_all.sh OUTDIR}
mkdir -p $O && O=${O:A}
cd ${0:A:h}/..
run() {  # name Z orbitals digits
  [[ -f $O/$1a.json ]] || python3 -u oracle_ns.py --Z $2 --orbitals "$3" --digits $4 --out $O/$1a.json > $O/$1a.log 2>&1
  [[ -f $O/$1b.json ]] || python3 -u oracle_ns.py --Z $2 --orbitals "$3" --digits $(( $4 + 10 )) --step 0.25 --guard 30 \
    --out $O/$1b.json > $O/$1b.log 2>&1
  grep -q E_cert $O/$1a.cert 2>/dev/null || ( time python3 certify.py $O/$1a.json --print-digits $(( $4 + 15 )) ) > $O/$1a.cert 2>&1
}
run ne500 10 "1s 2s 2p" 500
run mg100 12 "1s 2s 2p 3s" 100
run ar100 18 "1s 2s 2p 3s 3p" 100
run ca100 20 "1s 2s 2p 3s 3p 4s" 100
run zn100 30 "1s 2s 2p 3s 3p 3d 4s" 100
run kr100 36 "1s 2s 2p 3s 3p 3d 4s 4p" 100
run sr100 38 "1s 2s 2p 3s 3p 3d 4s 4p 5s" 100
run pd100 46 "1s 2s 2p 3s 3p 3d 4s 4p 4d" 100
run cd100 48 "1s 2s 2p 3s 3p 3d 4s 4p 4d 5s" 100
run xe100 54 "1s 2s 2p 3s 3p 3d 4s 4p 4d 5s 5p" 100
