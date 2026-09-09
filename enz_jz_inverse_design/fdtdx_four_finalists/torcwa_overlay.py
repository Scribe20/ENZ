"""Overlay the existing TORCWA spectra (Phase-1 stage5 outputs, read-only) with the new FDTDX spectra for the designs
where TORCWA spectra exist (final3, final0); numerical comparison only."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROV = json.load(open(HERE / "config" / "geometry_provenance.json"))
LAM_ZE = json.load(open(HERE / "materials" / "material_models.json"))["lambda_ZE_data_nm"]
TAGS = {"final3": "final3_div_F1_P825_h500_pad0.12_s333_h525", "final0": "final0_F2_P850_h600_pad0.12_s333_P825"}


def main():
    out = {}
    for d, tag in TAGS.items():
        tp = HERE.parent / "outputs" / "stage5" / tag / "spectra.npz"
        if not tp.exists():
            continue
        t = np.load(tp)
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
        res = {}
        for j, (ft, tk, lab) in enumerate((("prod", "with", "with ITO"), ("noito", "noito", "without ITO"))):
            fp = HERE / d / ft / f"spectra_{ft}.npz"
            ax = axs[j]
            for k, c in (("R", "b"), ("T", "r"), ("A", "k")):
                ax.plot(t["lam"], t[f"{tk}_{k}"], c + "-", lw=1.2, label=f"{k} TORCWA")
            if fp.exists():
                f = np.load(fp)
                for k, c in (("R", "b"), ("T", "r"), ("A", "k")):
                    ax.plot(f["lam_nm"], f[k], c + "--", lw=1.4, label=f"{k} FDTDX")
                common = (f["lam_nm"] >= max(t["lam"].min(), 1200)) & (f["lam_nm"] <= min(t["lam"].max(), 1400))
                res[lab] = {k: dict(max_abs_diff=float(np.max(np.abs(np.interp(f["lam_nm"][common], t["lam"], t[f"{tk}_{k}"]) - f[k][common]))),
                                   rms_diff=float(np.sqrt(np.mean((np.interp(f["lam_nm"][common], t["lam"], t[f"{tk}_{k}"]) - f[k][common]) ** 2))),
                                   at_ZE=[float(np.interp(LAM_ZE, t["lam"], t[f"{tk}_{k}"])), float(np.interp(LAM_ZE, f["lam_nm"], f[k]))]) for k in ("R", "T", "A")}
                res[lab]["lam_range_nm"] = [float(f["lam_nm"][common].min()), float(f["lam_nm"][common].max())]
            ax.axvline(LAM_ZE, color="0.5", ls=":", lw=0.8); ax.set_xlim(1200, 1400); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("λ [nm]"); ax.set_ylabel("R, T, A"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
            ax.set_title(f"{d} {lab}: TORCWA (solid, [7,7]) vs FDTDX (dashed)", fontsize=9)
        fig.tight_layout(); fig.savefig(HERE / "comparison" / f"torcwa_vs_fdtdx_{d}.png", dpi=150); plt.close(fig)
        out[d] = res
    json.dump(out, open(HERE / "comparison" / "torcwa_vs_fdtdx_spectra.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
