#!/bin/bash
# Regenerate every reported kernel table and figure.
set -e
cd "$(dirname "$0")"

# merge the two bandwidth sweeps into one file
python3 - <<'PY'
import csv
rows = list(csv.DictReader(open("eps_d1.csv")))
try:
    rows += list(csv.DictReader(open("eps_d1_hi.csv")))
except FileNotFoundError:
    pass
rows.sort(key=lambda r: float(r["eps"]))
with open("eps_d1_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"eps_d1_all.csv: {len(rows)} rows")
PY

python3 postprocess_refine.py refine refine_d1.csv _d1 --slope 3 --gamma_slope 1.5
python3 postprocess_refine.py refine refine_d2.csv _d2 --gamma_slope 1.5
python3 postprocess_refine.py eps    eps_d1_all.csv _d1
python3 postprocess_refine.py domain domain_d1.csv  _d1
python3 postprocess_refine.py theory theory_d1.csv  _d1
python3 table_fd_compare.py
echo "done"
