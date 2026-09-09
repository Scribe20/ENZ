#!/bin/bash
set -e
cd "$(dirname "$0")"
O=outputs/nonlinear_best
echo "== cylinder comparison lane =="
python3 build_lookup.py --geom cyl --order 7 7 --threads 4 --tag cyl_order7 --lam-range 1240 1360 2
python3 ttm_activation.py --lookup $O/lookup_cyl_order7.npz --tag cyl
python3 make_figures.py --tags cyl --lookup $O/lookup_cyl_order7.npz --suffix _cyl --lam-ops 1292 1294 1300
python3 activation_R.py --tags cyl --suffix _cyl --lam-ops 1292 1294 1300
python3 tcmt_fit.py --lookup $O/lookup_cyl_order7.npz --tag cyl_order7 --heat cyl --ttm-meta $O/ttm_meta_cyl.json --lam-ops 1292 1294 1300 1302
echo "[cyl pipeline] done"
