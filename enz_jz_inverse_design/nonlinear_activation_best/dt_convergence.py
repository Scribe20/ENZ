"""Time-step convergence of the TTM: compares the base run (dt = 0.5 fs, 61 intensities) with the
dt = 1 fs and dt = 0.25 fs runs (13 intensities) at the common intensities, all wavelengths."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs" / "nonlinear_best"


def load(tag, key):
    z = np.load(OUT / f"heatmap_{key}_{tag}.npz"); return z["I_peak_Wcm2"], z["value"], z["valid_model"]


def main():
    Ib, Tb, vb = load("base", "T"); _, Rb, _ = load("base", "R"); _, Ab, _ = load("base", "A"); _, Pb, _ = load("base", "Te_peak")
    res = {}
    for tag in ("dt1", "dt025"):
        if not (OUT / f"heatmap_T_{tag}.npz").exists():
            continue
        I, T, v = load(tag, "T"); _, R, _ = load(tag, "R"); _, A, _ = load(tag, "A"); _, P, _ = load(tag, "Te_peak")
        idx = [int(np.argmin(abs(Ib - i))) for i in I]
        m = vb[:, idx] & v
        d = dict(max_abs_dT=float(np.max(abs(T - Tb[:, idx])[m])), max_abs_dR=float(np.max(abs(R - Rb[:, idx])[m])), max_abs_dA=float(np.max(abs(A - Ab[:, idx])[m])),
                 max_abs_dTe_peak_K=float(np.max(abs(P - Pb[:, idx])[m])), max_rel_dTe_peak=float(np.max((abs(P - Pb[:, idx]) / Pb[:, idx])[m])),
                 dt_fs=float(json.load(open(OUT / f"ttm_meta_{tag}.json"))["dt_fs"]), n_states=int(m.sum()))
        res[tag] = d
        print(tag, json.dumps(d))
    json.dump(res, open(OUT / "dt_convergence.json", "w"), indent=1)


if __name__ == "__main__":
    main()
