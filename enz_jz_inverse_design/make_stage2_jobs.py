"""Build the Stage-2 (adaptive refinement) job list from the Stage-1 landscape.

Selection rule (predeclared, no hand-picking):
  * cells = (P, h, pad) with best-of-seeds Fz_hard from Stage 1
  * families = the top-K (P, h) cells overall (K = 4) + the best cell of every
    other period whose Fz_hard is within 75% of the global best (diversity)
  * for every family: warm-start refinement (beta_start = 8, from the best
    seed's rho_raw_final) at the centre and at the neighbours
        P +- dP, h +- dh, pad +- dpad  (clipped to the allowed ranges;
        P kept below the substrate diffraction threshold P_max)
  * warm-start CONTROLS from the frozen historical designs (direct-Ez
    winner, robust P800/P925 finalists) at the best family cell
  * two extra from-scratch seeds at the best family cell (longer schedule)
"""

import argparse
import json
from pathlib import Path

import numpy as np

import config

OUT1 = config.OUT / "stage1"
OUT2 = config.OUT / "stage2"


def best_cells(rows):
    cells = {}
    for r in rows:
        k = (r["P"], r["h"], r["pad_frac"])
        if k not in cells or r["Fz_hard"] > cells[k]["Fz_hard"]:
            cells[k] = r
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dP", type=float, default=25.0)
    ap.add_argument("--dh", type=float, default=25.0)
    ap.add_argument("--dpad", type=float, default=0.04)
    ap.add_argument("--P-max", type=float, default=None, help="default: lambda_ZE/n_glass - 5 nm")
    ap.add_argument("--diversity", type=float, default=0.75)
    a = ap.parse_args()
    import materials as mat
    lam, _ = mat.ito_zero_crossing()
    P_max = a.P_max or (lam / mat.n_glass(lam) - 5.0)
    rows = json.load(open(OUT1 / "results.json"))
    cells = best_cells(rows)
    by_ph = {}
    for (P, h, pad), r in cells.items():
        if (P, h) not in by_ph or r["Fz_hard"] > by_ph[(P, h)]["Fz_hard"]:
            by_ph[(P, h)] = r
    ranked = sorted(by_ph.values(), key=lambda r: -r["Fz_hard"])
    best = ranked[0]["Fz_hard"]
    fam = ranked[:a.K]
    seenP = {r["P"] for r in fam}
    for r in ranked[a.K:]:
        if r["P"] not in seenP and r["Fz_hard"] >= a.diversity * best:
            fam.append(r); seenP.add(r["P"])
    jobs, seen = [], set()

    def add(tag, P, h, pad, seed, warm, beta_start, n_iter=a.iters, order=None):
        P = float(np.clip(P, 400.0, P_max)); h = float(np.clip(h, 60.0, 600.0)); pad = float(np.clip(pad, config.PAD_MIN, 0.20))
        key = (round(P, 1), round(h, 1), round(pad, 3), seed, warm)
        if key in seen:
            return
        seen.add(key)
        jobs.append(dict(tag=tag, P=P, h=h, pad=pad, seed=int(seed), n_iter=int(n_iter),
                         order=list(order or config.ORDER_SCREEN), warm=warm, beta_start=beta_start,
                         eval_order=list(config.ORDER_FULL), n_threads=1))

    for i, r in enumerate(fam):
        P, h, pad, seed = r["P"], r["h"], r["pad_frac"], r["seed"]
        warm = str(OUT1 / "runs" / r["tag"] / "rho_raw_final.npy")
        fam_tag = f"F{i}_{r['tag']}"
        add(f"{fam_tag}_centre", P, h, pad, seed, warm, 8.0)
        for dP in (-a.dP, a.dP):
            add(f"{fam_tag}_P{P+dP:.0f}", P + dP, h, pad, seed, warm, 8.0)
        for dh in (-a.dh, a.dh):
            add(f"{fam_tag}_h{h+dh:.0f}", P, h + dh, pad, seed, warm, 8.0)
        for dpad in (-a.dpad, a.dpad):
            if config.PAD_MIN <= pad + dpad <= 0.20:
                add(f"{fam_tag}_pad{pad+dpad:.2f}", P, h, pad + dpad, seed, warm, 8.0)
    # controls: frozen historical designs as warm starts at the best cell (normalized cell, rescaled)
    b = fam[0]
    for name, rel in (("directEz", "enz_direct_enz_excitation/outputs/geometries/rho_hard_binary.npy"),
                      ("robustP800", "enz_robust_aito_campaign/outputs/stage4/runs/finalist0_P800_h200_pad0.040_warm/rho_raw_final.npy"),
                      ("robustP925", "enz_robust_aito_campaign/outputs/stage4/runs/finalist1_P925_h240_pad0.040_warm/rho_raw_final.npy")):
        add(f"CTRL_{name}_at_P{b['P']:.0f}_h{b['h']:.0f}_pad{b['pad_frac']:.2f}", b["P"], b["h"], b["pad_frac"], 0, str(config.ROOT / rel), 8.0)
    for s in (2024, 4096):
        add(f"SCRATCH_s{s}_P{b['P']:.0f}_h{b['h']:.0f}_pad{b['pad_frac']:.2f}", b["P"], b["h"], b["pad_frac"], s, None, 1.0, n_iter=int(1.5 * a.iters))
    OUT2.mkdir(parents=True, exist_ok=True)
    with open(OUT2 / "jobs.json", "w") as f:
        json.dump(jobs, f, indent=1)
    with open(OUT2 / "families.json", "w") as f:
        json.dump(dict(P_max=P_max, K=a.K, diversity=a.diversity, families=fam), f, indent=1)
    print(f"P_max = {P_max:.1f} nm; families:")
    for r in fam:
        print(f"  {r['tag']:28s} Fz_hard={r['Fz_hard']:.4f} A={r['A_hard']:.3f} eta_z={r['eta_z_hard']:.3f}")
    print(f"{len(jobs)} jobs -> {OUT2/'jobs.json'}")


if __name__ == "__main__":
    main()
