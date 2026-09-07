"""Stage 0: provenance manifest (SHA256, geometry, conventions) + config."""
import json
import sys

import common as cm

rows = cm.manifest()
cm.OUT.mkdir(parents=True, exist_ok=True)
cm.jdump(rows, cm.OUT / "manifest.json")
cm.jdump(dict(LAMBDA_E=cm.LAMBDA_E, D_ITO=cm.D_ITO, ORDER=cm.ORDER,
              S_LEVELS=list(cm.S_LEVELS), harness_grid="96x96x7 in ITO",
              energy_grid="48x48; a-Si 8 z, ITO 5 z, claddings 12x40 nm with (0,0) removed",
              pole_scan_dense=[1280, 1620, 4], pole_scan_track=[1300, 1600, 6],
              pole_acceptance=dict(RT_TOL=0.02, DAMP_MIN=5e-4, PEAK_FRAC=0.03,
                                   STAB_TOL=0.02, OVL_MIN=0.6),
              symmetry="no fliplr projection anywhere in this audit; historical "
                       "geometries are read-only",
              nonlinear_model="NONE in repository (grep TTM/hot-electron/T_e/"
                              "fluence/nonlinear over *.py,*.md,*.txt,*.csv,*.json): "
                              "linear proxies only"),
         cm.OUT / "config.json")
for r in rows:
    print(f"{r['name']:18s} {r['file'][:70]:70s} sha={str(r['sha256'])[:12]} P={r['P_nm']:.0f} h={r['h_nm']:.0f} pad={r['padding_nm']} fill={r['fill']:.3f}")
