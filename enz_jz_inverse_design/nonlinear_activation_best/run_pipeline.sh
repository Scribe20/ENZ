#!/bin/bash
# Post-lookup pipeline for the nonlinear activation lane (run after build_lookup.py --tag order7 is done).
set -e
cd "$(dirname "$0")"
O=outputs/nonlinear_best
echo "== TTM scans (best design) =="
python3 ttm_activation.py --tag base
python3 ttm_activation.py --tag G05 --G-scale 0.5
python3 ttm_activation.py --tag G2  --G-scale 2.0
python3 ttm_activation.py --tag dt1   --dt 1e-15   --nI 13
python3 ttm_activation.py --tag dt025 --dt 0.25e-15 --nI 13
echo "== figures / quantification (best design) =="
python3 make_figures.py --tags base G05 G2 --lam-ops 1292
echo "== TCMT fit (best design) =="
python3 tcmt_fit.py --tag order7 --heat base
echo "== certification of selected states at [9,9] and [11,11] =="
LAMS=$(python3 -c "import json;print(' '.join(f'{l:.0f}' for l in json.load(open('$O/activation_summary.json'))['lam_ops']))")
python3 build_lookup.py --order 9 9   --threads 4 --tag order9  --te 300 2000 4000 6000 8000 --lam $LAMS
python3 build_lookup.py --order 11 11 --threads 4 --tag order11 --te 300 2000 4000 6000 8000 --lam $LAMS
python3 certify_compare.py
echo "== cylinder comparison lane =="
python3 build_lookup.py --geom cyl --order 7 7 --threads 4 --tag cyl_order7 --lam-range 1240 1360 2
python3 ttm_activation.py --lookup $O/lookup_cyl_order7.npz --tag cyl
python3 make_figures.py --tags cyl --lookup $O/lookup_cyl_order7.npz --suffix _cyl --lam-ops 1292 1294 1300
python3 tcmt_fit.py --lookup $O/lookup_cyl_order7.npz --tag cyl_order7 --heat cyl --ttm-meta $O/ttm_meta_cyl.json --lam-ops 1292 1294 1300 1302
echo "[pipeline] done"
