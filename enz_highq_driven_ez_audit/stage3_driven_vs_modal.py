"""Stage 3: for every tracked branch, MODE QUALITY (Q_rad, modal
participation at the lossless-auxiliary pole and at the loaded pole) vs
DRIVEN ACCESSIBILITY (F_Ez, A_ITO, stored energy at lambda_E and at the
loaded pole).  Modal and driven normalizations are kept separate."""
import pandas as pd
import torch

import common as cm

torch.set_num_threads(4)
br = cm.jload(cm.OUT / "stage2_branches.json")
d1 = cm.jload(cm.OUT / "stage1_driven_lambdaE.json")
rows = []
for name, brs in br.items():
    rho, P, h = cm.load_candidate(name)
    for b in brs:
        fit, r0 = b["fit"], b["rows"][0]
        lam_p = r0["lambda_pole"]
        drv_p = cm.driven_metrics(rho, P, h, lam=lam_p)
        en_loaded = cm.energy_participation(rho, P, h, lam_p, 1.0)
        lossless = fit.get("lossless_pole")
        en_lossless = (cm.energy_participation(rho, P, h, lossless["lambda_nm"], 0.0)
                       if lossless else None)
        en_E = d1[name]["energy_lambda_E"]
        rows.append(dict(
            candidate=name, branch=b["branch"], P=P, h=h,
            lambda_pole=lam_p, Q_loaded=r0["Q"],
            Q_rad=fit.get("Q_rad"), Q_nr=fit.get("Q_nr"), gamma_ratio=fit.get("gamma_ratio"),
            Q_lossless=fit.get("Q_rad_from_lossless"), linearity_resid=fit.get("linearity_resid"),
            jumps=sum(1 for r in b["rows"] if r.get("jump_flag")),
            eta_ENZ_z_modal=(en_lossless["eta_ENZ_z"] if en_lossless else None),
            eta_z_modal=(en_lossless["eta_z"] if en_lossless else None),
            ito_E_frac_modal=(en_lossless["ito_E_energy_fraction"] if en_lossless else None),
            eta_ENZ_z_loaded=en_loaded["eta_ENZ_z"], eta_z_loaded=en_loaded["eta_z"],
            ito_E_frac_loaded=en_loaded["ito_E_energy_fraction"],
            F_Ez_lambdaE=d1[name]["F_Ez"], F_Ez_pole=drv_p["F_Ez"],
            eta_z_driven_lambdaE=d1[name]["eta_z"], eta_z_driven_pole=drv_p["eta_z"],
            A_lambdaE=d1[name]["A_rt"], A_pole=drv_p["A_rt"],
            U_lambdaE=en_E["U"], U_pole=en_loaded["U"],
            peak_Ez2_lambdaE=d1[name]["peak_Ez2"], Ez2_p95_lambdaE=d1[name]["Ez2_p95"],
            Ez2_p99_lambdaE=d1[name]["Ez2_p99"]))
        print(f"{b['branch']:30s} Qrad={fit.get('Q_rad', float('nan')):9.1f} Qnr={fit.get('Q_nr', float('nan')):6.2f} "
              f"ratio={fit.get('gamma_ratio', float('nan')):.4f} etaENZ(modal)="
              f"{(en_lossless or {}).get('eta_ENZ_z', float('nan')):.3e} F_Ez(E)={d1[name]['F_Ez']:.2f} "
              f"F_Ez(pole)={drv_p['F_Ez']:.2f} A(E)={d1[name]['A_rt']:.3f} A(pole)={drv_p['A_rt']:.3f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(cm.OUT / "stage3_branches_table.csv", index=False)
print("[stage3] done")
