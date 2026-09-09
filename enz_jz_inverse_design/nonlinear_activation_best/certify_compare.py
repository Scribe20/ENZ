"""Order-convergence certification of selected (lambda, Te) states: compares lookup tables built
with build_lookup.py at [7,7] (maps) against [9,9] and [11,11] re-evaluations of the selected states.

    python certify_compare.py --base outputs/nonlinear_best/lookup_order7.npz --others outputs/nonlinear_best/lookup_order9.npz outputs/nonlinear_best/lookup_order11.npz
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs" / "nonlinear_best"
KEYS = ("T", "R", "A", "Fz", "Ftot", "Fx", "Fy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(OUT / "lookup_order7.npz"))
    ap.add_argument("--others", nargs="+", default=[str(OUT / "lookup_order9.npz"), str(OUT / "lookup_order11.npz")])
    ap.add_argument("--tag", default="best")
    a = ap.parse_args()
    b = np.load(a.base)
    rows, summary = [], {}
    for path in a.others:
        z = np.load(path)
        order = f"[{int(z['order'][0])},{int(z['order'][1])}]"
        diffs = {k: [] for k in KEYS}
        for i, Te in enumerate(z["Te"]):
            ib = int(np.argmin(abs(b["Te"] - Te)))
            if abs(b["Te"][ib] - Te) > 1e-6:
                continue
            for j, lam in enumerate(z["lam"]):
                jb = int(np.argmin(abs(b["lam"] - lam)))
                if abs(b["lam"][jb] - lam) > 1e-6:
                    continue
                row = dict(order=order, Te=float(Te), lam=float(lam))
                for k in KEYS:
                    row[f"{k}_7"] = float(b[k][ib, jb]); row[k] = float(z[k][i, j]); row[f"d{k}"] = float(z[k][i, j] - b[k][ib, jb])
                    diffs[k].append(row[f"d{k}"])
                rows.append(row)
        summary[order] = {k: dict(max_abs=float(np.max(np.abs(diffs[k]))), rms=float(np.sqrt(np.mean(np.square(diffs[k]))))) for k in KEYS if diffs[k]}
        print(order, {k: f"max|d|={v['max_abs']:.2e}" for k, v in summary[order].items()})
    with open(OUT / f"certification_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(dict(base=a.base, others=a.others, summary=summary, n_states=len(rows)), open(OUT / f"certification_{a.tag}.json", "w"), indent=1)
    for r in rows:
        print(f"{r['order']} Te={r['Te']:.0f} lam={r['lam']:.0f}: T {r['T_7']:.5f}->{r['T']:.5f} ({r['dT']:+.1e})  A {r['A_7']:.4f}->{r['A']:.4f} ({r['dA']:+.1e})  Fz {r['Fz_7']:.4f}->{r['Fz']:.4f} ({r['dFz']:+.1e})")


if __name__ == "__main__":
    main()
