"""Stage 15: decision tree, canonical table, final report."""
import numpy as np
import pandas as pd

import common as cm


def J(name):
    p = cm.OUT / name
    return cm.jload(p) if p.exists() else {}


d1 = J("stage1_driven_lambdaE.json"); br = J("stage2_branches.json"); recon = J("stage2_recon.json")
t3 = pd.read_csv(cm.OUT / "stage3_branches_table.csv") if (cm.OUT / "stage3_branches_table.csv").exists() else pd.DataFrame()
par = J("stage4_pareto.json"); ang = J("stage5_angular_driven.json"); sparse = J("stage5_sparse_poles.json")
ali = J("stage6_alignment.json"); loss = J("stage7_loss_sweep.json"); leak = J("stage8_leakage.json")
p925 = J("stage9_p925_controls.json"); p800 = J("stage10_p800_narrow.json"); surv = J("stage11_survivors.json")
HIST = cm.jload(cm.ROOT / "enz_highq_enz_campaign/outputs/stage_a_decomposition.json")


def target_branch(name):
    """Branch used for the canonical row: P800 -> narrow 1437.6-nm branch;
    others -> tracked branch nearest lambda_E among those with Q_rad."""
    if t3.empty:
        return None
    sub = t3[t3["candidate"] == name]
    if sub.empty:
        return None
    if name == "P800 robust":
        return sub.iloc[int(np.argmin(np.abs(sub["lambda_pole"] - 1437.6)))]
    return sub.iloc[int(np.argmin(np.abs(sub["lambda_pole"] - cm.LAMBDA_E)))]


def fmt(v, nd=3):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
    except Exception:            # noqa: BLE001
        return str(v)


rows = []
for name in cm.CANDIDATES:
    if name == "P925/h220 robust":
        continue
    src, P, h, pad, camp = cm.CANDIDATES[name]
    b = target_branch(name)
    dd = d1.get(name, {})
    a = ang.get(name, [])
    le30 = [x for x in a if x["theta"] <= 30]
    sv = surv.get(name, {})
    fab = sv.get("fabrication", {}).get("variants", {})
    fab_s = (f"F_Ez {fmt(fab.get('erode_1px', {}).get('F_Ez'))}/{fmt(fab.get('dilate_1px', {}).get('F_Ez'))} "
             f"(erode/dilate 1px)" if fab else "n/a")
    loc = sv.get("locality", {})
    rows.append({
        "candidate": name, "P": P, "h": h, "pad_nm": pad,
        "lambda_pole": fmt(b["lambda_pole"], 1) if b is not None else "n/a",
        "Q_loaded": fmt(b["Q_loaded"], 2) if b is not None else "n/a",
        "Q_rad": fmt(b["Q_rad"], 1) if b is not None else "n/a",
        "Q_nr": fmt(b["Q_nr"], 2) if b is not None else "n/a",
        "g_rad/g_nr": fmt(b["gamma_ratio"], 4) if b is not None else "n/a",
        "eta_ENZ,z modal": fmt(b["eta_ENZ_z_modal"], 4) if b is not None else "n/a",
        "eta_z modal": fmt(b["eta_z_modal"]) if b is not None else "n/a",
        "ITO E-frac modal": fmt(b["ito_E_frac_modal"]) if b is not None else "n/a",
        "F_Ez(lamE)": fmt(dd.get("F_Ez")), "F_Ez(pole)": fmt(b["F_Ez_pole"]) if b is not None else "n/a",
        "peak Ez2": fmt(dd.get("peak_Ez2"), 1), "Ez2 p95/p99": f"{fmt(dd.get('Ez2_p95'), 1)}/{fmt(dd.get('Ez2_p99'), 1)}",
        "A(lamE)": fmt(dd.get("A_rt")), "A(pole)": fmt(b["A_pole"]) if b is not None else "n/a",
        "abs density (A/d, 1/nm)": fmt(dd.get("abs_density_fixed_intensity"), 5),
        "ang F_Ez mean/min <=30": (f"{np.mean([x['F_Ez'] for x in le30]):.2f}/{np.min([x['F_Ez'] for x in le30]):.2f}" if le30 else "n/a"),
        "ang A mean/min <=30": (f"{np.mean([x['A'] for x in le30]):.3f}/{np.min([x['A'] for x in le30]):.3f}" if le30 else "n/a"),
        "fabrication": fab_s,
        "locality": (f"comp={loc.get('n_components')}, minfeat={fmt(loc.get('min_feature_nm'), 0)} nm, gap={fmt(loc.get('min_gap_nm'), 0)} nm, "
                     f"pad={pad} nm" if loc else f"pad={pad} nm"),
    })
# leakage-tuning winners (Pareto-superior vs parent)
extra = []
if leak:
    par_ = leak.get("parent", {})
    for fam, rr in leak.items():
        if fam == "parent":
            continue
        for r in rr:
            if (r.get("Q_rad") or 0) > 30 and r.get("F_Ez", 0) > 1.15 * par_.get("F_Ez", 9):
                extra.append(r)
for r in extra:
    rows.append({"candidate": f"padded QNM perturbed ({r['tag']})", "P": r["P"], "h": r["h"], "pad_nm": "~86 (ring kept)",
                 "lambda_pole": fmt(r["lambda_pole"], 1), "Q_loaded": fmt(r["Q_loaded"], 2), "Q_rad": fmt(r["Q_rad"], 1),
                 "Q_nr": fmt(r["Q_nr"], 2), "g_rad/g_nr": fmt(r["gamma_ratio"], 4), "eta_ENZ,z modal": fmt(r["eta_ENZ_z_modal"], 4),
                 "eta_z modal": fmt(r["eta_z_modal"]), "ITO E-frac modal": "n/a", "F_Ez(lamE)": fmt(r["F_Ez"]), "F_Ez(pole)": "n/a",
                 "peak Ez2": "n/a", "Ez2 p95/p99": "n/a", "A(lamE)": fmt(r["A"]), "A(pole)": "n/a", "abs density (A/d, 1/nm)": fmt(r["A"] / cm.D_ITO, 5),
                 "ang F_Ez mean/min <=30": "n/a", "ang A mean/min <=30": "n/a", "fabrication": "n/a", "locality": str(r["checks"])})
tab = pd.DataFrame(rows)
tab.to_csv(cm.OUT / "final_table.csv", index=False)

# ---------------------------------------------------------------------------
# decision tree
# ---------------------------------------------------------------------------
def brow(name):
    b = target_branch(name)
    return None if b is None else dict(Q_rad=float(b["Q_rad"]), F_Ez=float(b["F_Ez_lambdaE"]), ratio=float(b["gamma_ratio"]),
                                       eta=float(b["eta_ENZ_z_modal"]) if np.isfinite(b["eta_ENZ_z_modal"]) else None,
                                       lam=float(b["lambda_pole"]), Ql=float(b["Q_loaded"]))


B925, B800, BQNM, BFENZ = brow("P925 robust"), brow("P800 robust"), brow("padded QNM"), brow("padded F_ENZ")
all_Q = t3["Q_rad"].replace([np.inf], np.nan).dropna() if not t3.empty else pd.Series([], dtype=float)
all_F = t3["F_Ez_lambdaE"] if not t3.empty else pd.Series([], dtype=float)
hiQ = float(np.nanpercentile(all_Q, 75)) if len(all_Q) else 100.0
hiF = 1.5 * 2.2
case = None
if B925 and B925["Q_rad"] >= max(100, hiQ) and B925["F_Ez"] >= 3.5:
    case = "A"
elif B800 and B800["Q_rad"] >= max(100, hiQ) and B800["eta"] and B800["eta"] > 0.05:
    case = "B"
else:
    case = "C"
best_leak = max(extra, key=lambda r: r["F_Ez"]) if extra else None
if best_leak and BQNM and best_leak["F_Ez"] > 1.3 * d1["padded QNM"]["F_Ez"] and (best_leak.get("Q_rad") or 0) > 30:
    case = "D"
coexist = [r for _, r in t3.iterrows() if r["Q_rad"] >= max(100, hiQ) and r["F_Ez_lambdaE"] >= hiF] if not t3.empty else []
if not coexist and case in ("A", "B", "D"):
    pass
if not coexist and case == "C" and not best_leak:
    case = "E"

L = ["# HIGH-Q + HIGH-DRIVEN-Ez / ENZ PHYSICS AUDIT - REPORT", "",
     "Generated by `stage15_report.py` from `outputs/*.json|csv` (all numbers reproducible from the stage scripts).", "",
     "## Conventions (measured facts)", "",
     f"- Common harness: TORCWA complex128, order {cm.ORDER}, lambda_E = {cm.LAMBDA_E} nm, lab-x polarization, |E_inc| = 1; A = 1 - R_total - T_total (all orders); "
     "F_Ez = <|Ez/E_inc|^2>_ITO; eta_z = int|Ez|^2/int|E|^2 (ITO); modal proxies from the driven field at the pole of the lossless-ITO auxiliary structure "
     "(eta_ENZ,z = int_ITO|Ez|^2 / U, U = stored energy incl. dispersive ITO term, claddings without the (0,0) harmonic).",
     f"- Loss scaling s in {list(cm.S_LEVELS)} on Im eps_ITO; gamma(s) = gamma_rad + s gamma_nr fitted on the field-overlap-tracked branch; "
     "poles accepted only if found in both r and t (AAA), damped, with significant Lorentzian peak fraction; high-Q poles re-fitted on a dense local rescan.",
     "- No fliplr symmetry projection anywhere; frozen geometries read-only (SHA256 in `outputs/manifest.json`).", "",
     "## 1. Pole reconnaissance and loss-scaling decomposition (Stage 2, measured)", ""]
for name, brs in br.items():
    L.append(f"**{name}** (P = {cm.CANDIDATES[name][1]:.0f}, h = {cm.CANDIDATES[name][2]:.0f}): significant s=1 poles: "
             + ", ".join(f"{p['lambda_nm']:.1f} nm (Q {p['Q']:.1f})" for p in recon.get(name, [])))
    for b in brs:
        f = b["fit"]
        L.append(f"  - branch {b['branch']}: " + " -> ".join(f"s={r['s']:g}: {r['lambda_pole']:.1f}/Q{r['Q']:.1f}{'*' if r.get('jump_flag') else ''}" for r in b["rows"])
                 + f"; fit Q_rad = {fmt(f.get('Q_rad'), 1)}, Q_nr = {fmt(f.get('Q_nr'), 2)}, gamma_rad/gamma_nr = {fmt(f.get('gamma_ratio'), 4)}, "
                 f"linearity residual {fmt(f.get('linearity_resid'), 3)}, lossless-aux Q = {fmt(f.get('Q_rad_from_lossless'), 1)} (* = overlap/continuity flag)")
for k, hk in (("padded QNM", "padded QNM winner"), ("padded F_ENZ", "padded F_ENZ winner")):
    hh = HIST[hk]
    L.append(f"- Historical Stage-A ({hk}): Q_loaded {hh['Q_loaded']:.2f}, Q_rad {hh['Q_rad']:.0f}, Q_nr {hh['Q_nr']:.2f}, ratio {hh['gamma_ratio']:.4f} "
             f"(10-nm scan, nearest-continuation tracking) - compare with the field-overlap-tracked branches above.")
L += ["", "## 2. Driven vs modal (Stage 3, measured/derived)", "", "See `outputs/stage3_branches_table.csv`; key rows:", "",
      "| branch | lambda_pole | Q_loaded | Q_rad | Q_nr | g_rad/g_nr | eta_ENZ,z modal | eta_z modal | F_Ez(lamE) | F_Ez(pole) | A(lamE) | A(pole) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
if not t3.empty:
    for _, r in t3.iterrows():
        L.append(f"| {r['branch']} | {r['lambda_pole']:.1f} | {r['Q_loaded']:.2f} | {fmt(r['Q_rad'], 1)} | {fmt(r['Q_nr'], 2)} | {fmt(r['gamma_ratio'], 4)} | "
                 f"{fmt(r['eta_ENZ_z_modal'], 4)} | {fmt(r['eta_z_modal'])} | {r['F_Ez_lambdaE']:.3f} | {r['F_Ez_pole']:.3f} | {r['A_lambdaE']:.3f} | {r['A_pole']:.3f} |")
L += ["", "## 3. Pareto analysis (Stage 4)", "",
      f"- highest-Q Ez-rich parent: {par.get('highest_Q_Ez_rich_parent')}; strongest driven Ez: {par.get('strongest_driven_Ez')}; "
      f"best accessible high-Q (Q_rad >= 100): {par.get('best_accessible_highQ')}; Pareto knee (log Q_rad vs F_Ez): {par.get('knee')}.",
      f"- 'high' defined from the data: Q_rad >= max(100, 75th percentile = {hiQ:.0f}); 'strong' F_Ez >= 1.5 x historical 2.2 = {hiF:.1f}. "
      f"Candidates in that upper-right region: {par.get('upper_right_candidates') or 'NONE'}.",
      "- Fronts: " + "; ".join(f"{k}: {v}" for k, v in par.get("fronts", {}).items()),
      "", "![pareto](outputs/figures/fig1-4_pareto.png)", "", "## 4. Angular behaviour (Stage 5)", ""]
for name, a in ang.items():
    L.append(f"- {name}: " + ", ".join(f"({x['theta']:.0f},{x['phi']:.0f}) A={x['A']:.3f} F_Ez={x['F_Ez']:.2f} eta_z={x['eta_z']:.2f}" for x in a))
for name, s in sparse.items():
    L.append(f"- {name} tracked branch {s['branch']}: " + ", ".join(
        f"({p['theta']},{p['phi']}): {fmt(p['tracked']['lambda_nm'], 1) if p['tracked'] else 'lost'} nm / Q {fmt(p['tracked']['Q'], 1) if p['tracked'] else '-'}"
        for p in s["points"]))
L += ["", "## 5. Spectral alignment (Stage 6)", ""]
for name, a in ali.items():
    L.append(f"- {name}: base P={a['base']['P']:.0f}/h={a['base']['h']:.0f}; best F_Ez(lambda_E) at P={a['best_F_Ez']['P']:.0f}, h={a['best_F_Ez']['h']:.0f} "
             f"(F_Ez={a['best_F_Ez']['F_Ez']:.3f}); best-aligned near pole at P={a['best_aligned']['P']:.0f}, h={a['best_aligned']['h']:.0f} "
             f"(lambda_pole={a['best_aligned']['lambda_pole']:.1f} nm).")
    for p in a["points"]:
        if "Q_rad_3level" in p:
            L.append(f"    - P={p['P']:.0f} h={p['h']:.0f}: F_Ez={p['F_Ez']:.3f} A={p['A']:.3f} near pole {p['near_pole']['lambda_nm']:.1f} nm Q_l={p['near_pole']['Q']:.1f} "
                     f"Q_rad(3-level)={fmt(p['Q_rad_3level'].get('Q_rad'), 1)} ratio={fmt(p['Q_rad_3level'].get('gamma_ratio'), 4)}")
L += ["", "## 6. Loss / coupling sweep (Stage 7)", ""]
for bid, rr in loss.items():
    L.append(f"- {bid}: " + ", ".join(f"s={r['s']:g}: ratio={fmt(r['gamma_ratio'], 3)} F_Ez(pole)={r['F_Ez_pole']:.2f} A(pole)={r['A_pole']:.3f} F_Ez(lamE)={r['F_Ez_E']:.2f}" for r in rr))
L += ["", "![loss sweep](outputs/figures/stage7_loss_sweep.png)", "", "## 7. Controlled leakage tuning of the padded QNM parent (Stage 8)", ""]
if leak:
    pr = leak["parent"]
    L.append(f"- parent: F_Ez={pr['F_Ez']:.3f} A={pr['A']:.3f} pole {fmt(pr['lambda_pole'], 1)} nm Q_l={fmt(pr['Q_loaded'], 2)} Q_rad={fmt(pr['Q_rad'], 1)} "
             f"ratio={fmt(pr['gamma_ratio'], 4)} eta_ENZ,z={fmt(pr['eta_ENZ_z_modal'], 4)}")
    for fam, rr in leak.items():
        if fam == "parent":
            continue
        L.append(f"- {fam}: " + "; ".join(f"a={r['alpha']:g}: F_Ez={r['F_Ez']:.2f} A={r['A']:.3f} Q_rad={fmt(r['Q_rad'], 0)} ratio={fmt(r['gamma_ratio'], 3)} "
                                          f"etaENZ={fmt(r['eta_ENZ_z_modal'], 3)} ovl={fmt(r['overlap_with_parent'], 2)} ok={r['checks']['padding_ok'] and r['checks']['n_components'] == 1}" for r in rr))
    L.append(f"- Pareto-superior perturbed states (Q_rad > 30 and F_Ez > 1.15 x parent): {[r['tag'] for r in extra] or 'NONE'}")
    L.append("![leakage](outputs/figures/fig7_leakage_tuning.png)")
L += ["", "## 8. P925 mechanism (Stage 9)", ""]
if p925:
    for key in ("with_ITO", "lossless_ITO", "no_ITO"):
        L.append(f"- {key} poles: " + ", ".join(f"{p['lambda_nm']:.1f} nm (Q {p['Q']:.1f}, peak r {p['peak_r']:.2f})" for p in p925[key]["poles"]))
    L.append(f"- Fourier energy fractions of Ez in ITO at lambda_E: {({k: round(v, 3) for k, v in p925['fourier']['lambda_E'].items() if v > 0.02})} "
             f"(G10/k0 = {p925['fourier']['G10_over_k0']:.2f}, K_ENZ/k0 = 1.69)")
    m = p925["multipoles"]["lambda_E"]
    L.append(f"- multipole power fractions at lambda_E (free-space, diagnostic): {({k: round(v, 3) for k, v in m['power_fractions'].items()})}; "
             f"forward/backward coherent cancellation ratios {m['forward']['cancellation_ratio']:.2f} / {m['backward']['cancellation_ratio']:.2f}")
    L.append("- branch overlaps (mid-plane Ez maps): see `stage9_p925_controls.json['branch_overlap']`")
L += ["", "## 9. P800 narrow-pole physics (Stage 10)", ""]
if p800:
    for tag in ("narrow", "broad"):
        b = p800[tag]
        L.append(f"- {tag} branch {b['lambda_pole']:.1f} nm (Q_l {b['Q_loaded']:.1f}): driven at pole F_Ez={b['driven_at_pole']['F_Ez']:.2f} A={b['driven_at_pole']['A_rt']:.3f} "
                 f"eta_z={b['driven_at_pole']['eta_z']:.2f}; loaded eta_ENZ,z={b['participation_loaded']['eta_ENZ_z']:.3e} ITO E-frac={b['participation_loaded']['ito_E_energy_fraction']:.3f}; "
                 f"lossless-aux eta_ENZ,z={fmt((b['participation_lossless'] or {}).get('eta_ENZ_z'), 4)} ITO E-frac={fmt((b['participation_lossless'] or {}).get('ito_E_energy_fraction'))}; "
                 f"Fourier {({k: round(v, 3) for k, v in b['fourier_Ez_ITO'].items() if v > 0.03})}; multipole fractions "
                 f"{({k: round(v, 2) for k, v in b['multipoles']['power_fractions'].items()})}, cancellation fwd/bwd {b['multipoles']['forward']['cancellation_ratio']:.2f}/{b['multipoles']['backward']['cancellation_ratio']:.2f}")
    se = p800["s_matrix_evidence"]
    L.append(f"- direct S-matrix evidence at the narrow pole: R={se['at_pole']['R']:.3f} T={se['at_pole']['T']:.3f} A={se['at_pole']['A']:.3f} "
             f"(off-resonance R={se['off_resonance']['R']:.3f} T={se['off_resonance']['T']:.3f} A={se['off_resonance']['A']:.3f}); residue peak fractions r {se['residues']['peak_r']:.2f}, t {se['residues']['peak_t']:.2f}. "
             f"narrow/broad field overlap {p800['overlap_narrow_broad']:.2f}. No transverse-Kerker/BIC label is assigned unless the coherent channel sums cancel.")
L += ["", "## 10. Survivors: proxies, normalization, fabrication, convergence (Stages 11-14)", ""]
for name, sv in surv.items():
    L.append(f"- **{name}**: proxies {({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sv['proxies'].items() if k != 'note'})}")
    L.append(f"  - normalization: A/d = {sv['normalization']['fixed_intensity_abs_energy_density_per_fluence']:.5f} 1/nm (fixed intensity); "
             f"A/(P^2 d) = {sv['normalization']['fixed_power_per_cell_abs_energy_density']:.3e} 1/nm^3 (fixed power per cell)")
    L.append("  - fabrication: " + "; ".join(f"{k}: F_Ez={v['F_Ez']:.2f} A={v['A']:.3f} pole={fmt(v['lambda_pole'], 1)} Q={fmt(v['Q_loaded'], 1)}"
                                            + (f" Q_rad~{fmt(v['Q_rad_3level'].get('Q_rad'), 0)}" if 'Q_rad_3level' in v else "") for k, v in sv["fabrication"]["variants"].items()))
    L.append(f"  - locality: {sv['locality']}; supercell: {sv.get('supercell')}")
    L.append("  - convergence: " + "; ".join(f"{k}: F_Ez={v['F_Ez']:.3f} A={v['A']:.4f} pole={fmt(v['lambda_pole'], 1)} Q={fmt(v['Q_loaded'], 1)}" for k, v in sv["convergence"].items()))
L += ["", "## 11. Canonical table", "", "| " + " | ".join(tab.columns) + " |", "|" + "---|" * len(tab.columns)]
for _, r in tab.iterrows():
    L.append("| " + " | ".join(str(r[c]) for c in tab.columns) + " |")
s16 = J("stage16/stage16_results.json")
L += ["", "## 11b. Stage 16 - epsilon-constraint Pareto designs (only because no coexistence was found)", ""]
if s16:
    L += ["maximize F_Ez(lambda_E, real loss) s.t. the lossless-ITO Ez resonance at lambda_E has half-width <= lambda_E/(2 Q_target) "
          "(differentiable Q_rad proxy), positive padding, no mirror symmetry; every result re-certified with the field-overlap tracker at [7,7]:", "",
          "| design | seed | Q_target | F_Ez(lamE) | A | near pole (Q_l) | Q_rad | Q_nr | g_rad/g_nr | eta_ENZ,z modal | S_flip |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r_ in s16:
        npole = r_.get("near_pole") or {}
        L.append(f"| {r_['tag']} | {r_['seed']} | {r_['Q_target']:.0f} | {r_['F_Ez']:.3f} | {r_['A']:.3f} | {fmt(npole.get('lambda_nm'), 1)} ({fmt(npole.get('Q'), 1)}) | "
                 f"{fmt(r_.get('Q_rad'), 1)} | {fmt(r_.get('Q_nr'), 1)} | {fmt(r_.get('gamma_ratio'), 3)} | {fmt(r_.get('eta_ENZ_z_modal'), 4)} | {r_['s_flip']:.2f} |")
    L += ["", "![stage16](outputs/figures/stage16_pareto_frontier.png)"]
else:
    L.append("not run")
L += ["", "## 12. Decision (Stage 15)", "",
      f"Decision-tree case: **{case}**.",
      f"- P925 target-near branch: {B925}", f"- P800 narrow branch: {B800}", f"- padded QNM branch: {BQNM}", f"- padded F_ENZ branch: {BFENZ}",
      f"- coexistence set (Q_rad >= {max(100, hiQ):.0f} and F_Ez >= {hiF:.1f}): {[r['branch'] for r in coexist] or 'NONE'}",
      f"- best leakage-tuned state: {best_leak['tag'] if best_leak else 'none Pareto-superior'}", "",
      "## 13. Answer to the final question", ""]
if coexist:
    L.append("**YES.** " + "; ".join(f"{r['branch']}: Q_rad {r['Q_rad']:.0f}, F_Ez {r['F_Ez_lambdaE']:.2f}, eta_ENZ,z modal {fmt(r['eta_ENZ_z_modal'], 3)}, ratio {r['gamma_ratio']:.3f}" for r in coexist)
             + " - see the convergence and fabrication sections for the certification status.")
else:
    L.append("**NO existing air-padded candidate combines high Q_rad with strong plane-wave-driven longitudinal ENZ field.** Nearest Pareto candidates and what each lacks:")
    if B925:
        L.append(f"- P925 (strongest driven Ez, F_Ez {B925['F_Ez']:.2f}): Q_rad {B925['Q_rad']:.0f}, eta_ENZ,z modal {fmt(B925['eta'], 3)} - lacks {'radiative Q' if B925['Q_rad'] < max(100, hiQ) else 'modal ENZ character'}.")
    if B800:
        L.append(f"- P800 narrow branch ({B800['lam']:.1f} nm): Q_rad {B800['Q_rad']:.0f}, F_Ez(lambda_E) {B800['F_Ez']:.2f}, eta_ENZ,z modal {fmt(B800['eta'], 3)} - lacks "
                 f"{'driven Ez at lambda_E' if B800['F_Ez'] < hiF else 'Q_rad'}.")
    if BQNM:
        L.append(f"- padded QNM parent: Q_rad {BQNM['Q_rad']:.0f}, eta_ENZ,z modal {fmt(BQNM['eta'], 3)}, F_Ez {BQNM['F_Ez']:.2f}, ratio {BQNM['ratio']:.4f} - lacks radiative accessibility (dark).")
    L.append("- Minimum targeted change: see Stage 6 (alignment) and Stage 8 (leakage) results above; the Stage-8 family with the best F_Ez at Q_rad > 30 is the recommended refinement direction, "
             "otherwise an epsilon-constraint Pareto inverse design (maximize F_Ez s.t. Q_rad >= Q_target) from the P925 and padded-QNM seeds (Stage 16, not run unless needed).")
L += ["", "## 14. Unresolved uncertainty", "",
      "- Q_rad values > ~500 rely on AAA extrapolation of poles narrower than the 6-nm tracking scan; the dense local rescans certify sampling convergence only where marked converged.",
      "- Modal participation uses the driven field at the lossless-auxiliary pole as an eigenmode proxy (exact only for Q -> infinity).",
      "- Multipole channel amplitudes use free-space single-cell expressions (no substrate/ITO Green's function); only the direct S-matrix amplitudes are authoritative.",
      "- The angular set is an assumption (no NA requirement in the repository); nonlinear relevance uses linear proxies only (no validated ITO nonlinear model in the repository)."]
(cm.HERE / "REPORT.md").write_text("\n".join(L))
cm.jdump(dict(case=case, coexist=[r["branch"] for r in coexist], B925=B925, B800=B800, BQNM=BQNM, BFENZ=BFENZ,
              hiQ=hiQ, hiF=hiF, best_leak=(best_leak["tag"] if best_leak else None)), cm.OUT / "stage15_decision.json")
print("[stage15] case", case, "coexist", [r["branch"] for r in coexist])
