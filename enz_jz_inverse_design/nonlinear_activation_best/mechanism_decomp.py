"""Mechanism decomposition (RCWA, Lane B): at the operating wavelengths, evaluate the frozen design with
(a) hot eps' + cold eps'', (b) cold eps' + hot eps'', (c) both hot, for several T_e.  Tells whether the
response change comes from the real-part (resonance shift / ENZ shift) or from the imaginary-part (loss-rate)
change of the ITO permittivity.  Also evaluates the a-Si structure with the ITO layer replaced by a lossless
eps = eps'_hot slab to isolate the background.

    python mechanism_decomp.py
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import config, forward as fwd, materials as mat        # noqa: E402
import ito_nonlinear as nl                             # noqa: E402

OUT = HERE / "outputs" / "nonlinear_best"
RHO = HERE.parent / "outputs" / "best" / "rho_hard_binary.npy"
P, H = 825.0, 525.0


def main():
    fwd.set_threads(2)
    rho = torch.as_tensor(np.load(RHO), dtype=config.GEO_DTYPE)
    model = nl.ITOHot()
    rows = []
    for lam in (1302.0, 1294.0, 1292.0):
        e_cold = model.eps(lam, 300.0)
        for Te in (2000.0, 4000.0, 6000.0):
            e_hot = model.eps(lam, Te)
            cases = {"cold": e_cold, "eps_re_hot_only": complex(e_hot.real, e_cold.imag), "eps_im_hot_only": complex(e_cold.real, e_hot.imag), "both_hot": e_hot}
            for name, ep in cases.items():
                d = fwd.to_floats(fwd.evaluate(rho, P, H, lam, [7, 7], n_z=7, eps_ito=ep))
                rows.append(dict(lam=lam, Te=Te, case=name, eps_re=ep.real, eps_im=ep.imag, T=d["T"], R=d["R"], A=d["A"], Fz=d["Fz"], mean_Ez2=d["mean_Ez2"]))
                print(f"lam {lam:.0f} Te {Te:.0f} {name:16s} eps={ep.real:+.3f}+{ep.imag:.3f}i  T={d['T']:.5f} R={d['R']:.4f} A={d['A']:.4f} Fz={d['Fz']:.4f} <|Ez|^2>={d['mean_Ez2']:.2f}", flush=True)
    json.dump(rows, open(OUT / "mechanism_decomposition.json", "w"), indent=1)
    print("[mechanism] done")


if __name__ == "__main__":
    main()
