"""RCWA lookup table for the FROZEN best design: T, R, A, Fx, Fy, Fz, Ftot on a
(lambda, Te) grid with the Lane-B eps_ITO(lambda, Te) of ito_nonlinear.py.
Every other material at its supplied cold dispersion (a-Si:H, glass do not heat
in this model).  Order [7,7]; selected states re-certified at [9,9]/[11,11].

    python build_lookup.py [--order 7 7] [--threads 4]
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat          # noqa: E402
import ito_nonlinear as nl                               # noqa: E402

OUT = HERE / "outputs" / "nonlinear_best"
RHO = HERE.parent / "outputs" / "best" / "rho_hard_binary.npy"
P, H, PAD = 825.0, 525.0, 0.12
TE_GRID = np.array([300, 400, 500, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000], float)
LAM_GRID = np.arange(1200.0, 1380.01, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, nargs=2, default=[7, 7])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="order7")
    ap.add_argument("--te", type=float, nargs="*", default=None)
    ap.add_argument("--lam", type=float, nargs="*", default=None)
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    rho = torch.as_tensor(np.load(RHO), dtype=config.GEO_DTYPE)
    model = nl.ITOHot()
    lam_ze, _ = mat.ito_zero_crossing()
    Te_grid = np.array(a.te, float) if a.te else TE_GRID
    lams = np.array(a.lam, float) if a.lam else LAM_GRID
    lams = lams[np.abs(lams - lam_ze) > 0.05]            # never exactly on the (lossy, fine) crossing; harmless
    keys = ("T", "R", "A", "Fx", "Fy", "Fz", "Ftot", "mean_Ez2")
    tab = {k: np.zeros((len(Te_grid), len(lams))) for k in keys}
    eps_tab = np.zeros((len(Te_grid), len(lams)), complex)
    log = open(OUT / f"lookup_{a.tag}.log", "a")
    t0 = time.time()
    for i, Te in enumerate(Te_grid):
        for j, lam in enumerate(lams):
            ep = model.eps(float(lam), float(Te))
            eps_tab[i, j] = ep
            d = fwd.to_floats(fwd.evaluate(rho, P, H, float(lam), a.order, n_z=7, eps_ito=ep))
            for k in keys:
                tab[k][i, j] = d[k]
        jz = int(np.argmin(np.abs(lams - lam_ze)))
        msg = (f"Te={Te:6.0f} K: eps(1302)={eps_tab[i, jz].real:+.4f}+{eps_tab[i, jz].imag:.4f}i  T(1302)={tab['T'][i, jz]:.4f} "
               f"A={tab['A'][i, jz]:.4f} Fz={tab['Fz'][i, jz]:.4f} | T_min={tab['T'][i].min():.4f}@{lams[tab['T'][i].argmin()]:.0f} "
               f"A_max={tab['A'][i].max():.4f}@{lams[tab['A'][i].argmax()]:.0f} | max|Ftot-A|={np.abs(tab['Ftot'][i]-tab['A'][i]).max():.1e} | {time.time()-t0:.0f}s")
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        np.savez_compressed(OUT / f"lookup_{a.tag}.npz", Te=Te_grid[:i + 1], lam=lams, eps=eps_tab[:i + 1],
                            **{k: v[:i + 1] for k, v in tab.items()}, order=np.array(a.order), P=P, h=H, pad=PAD)
    print(f"[lookup {a.tag}] done in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
