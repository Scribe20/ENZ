"""Generate comparison/NOITO_RERUN.md from the numbers the runs produced.

Everything in the report is read from
  comparison/noito_fresh_convergence.json      (per-run closure / window-drift test)
  comparison/spectra_table_noito_accepted.json (accepted with/without-ITO pairs)
so the text cannot drift from the data.  Numerical-convergence reporting only; no physics
interpretation, which the campaign brief leaves to the reader.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMP = HERE / "comparison"
TOL = 0.01
ORDER = ["final3", "final1", "final0", "final2"]


def fmt(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def main():
    conv = json.load(open(COMP / "noito_fresh_convergence.json"))
    acc_p = COMP / "spectra_table_noito_accepted.json"
    acc = json.load(open(acc_p)) if acc_p.exists() else []
    L = []
    L.append("# Fresh no-ITO FDTDX runs\n")
    L.append("Every number below comes from a new FDTDX simulation. The geometry, materials, source, "
             "spatial mesh and monitors are those of the with-ITO production run of the same design; "
             "the ITO layer is replaced by the substrate glass and nothing else is changed, except "
             "where the table records a longer simulation time or a deeper glass region. R and T are "
             "the flux-plane phasors divided by the incident flux of the empty-domain reference run on "
             "the same mesh, and A = 1 − R − T. No spectrum is smoothed, clipped, renormalised or "
             "otherwise modified after the simulation.\n")
    L.append("## Acceptance test\n")
    L.append("Without the ITO every medium in the stack is lossless (the supplied a-Si:H and SiO2 data "
             "have k = 0 in band, and the fitted Lorentz models are undamped), so the exact absorption "
             f"is zero at every wavelength. A run is accepted only if all of the following hold:\n")
    L.append(f"* `max |A| = max |1 − R − T| <= {TOL}` over the whole 1200–1400 nm grid;\n"
             "* no wavelength with `R > 1`;\n"
             "* no wavelength with `T < 0`;\n"
             f"* the spectrum has stopped changing with simulation time: the difference between the "
             f"longest early DFT window and the full window is `<= {TOL}` in both R and T.\n")
    L.append("\nThe first three are the physical-validity test and the last is the time-stability test. "
             "They are reported separately because they fail differently: a drift in which R and T "
             "exchange with their sum conserved (`|dR + dT|` small) leaves the energy budget intact, "
             "whereas one in which the sum moves is error in the spectrum itself.\n")
    L.append("The early windows are recorded by the same run — extra flux-plane phasor detectors whose "
             "DFT window is closed early — so the convergence certificate needs no second simulation "
             "and no post-processing.\n")
    L.append("## Runs and their acceptance test\n")
    L.append("| design | run | time | max\\|A\\| | at | pts >tol | R_max | T_min | window drift R / T | physical | stable |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in ORDER + [k for k in conv if k not in ORDER]:
        r = conv.get(d)
        if not r:
            continue
        phys = "yes" if r.get("passes_physical", r["passes"]) else "NO"
        if r.get("passes_stability", r["passes"]):
            stab = "yes"
        else:
            stab = "NO (%d pts, max \\|dR+dT\\| %.4f)" % (r.get("n_points_drift_gt_tol", 0),
                                                      r.get("max_abs_dR_plus_dT", 0.0))
        L.append(f"| {d} | `{r.get('run_tag', '?')}` | {r['time_fs']:.0f} fs ({r['n_steps']} steps) | "
                 f"{fmt(r['max_abs_A'])} | {r['lam_max_abs_A']:.0f} nm | "
                 f"{r['n_points_absA_gt_tol']}/{r['n_points']} | {fmt(r['R_max'])} | {r['T_min']:+.4f} | "
                 f"{fmt(r['max_window_diff_R'])} / {fmt(r['max_window_diff_T'])} | "
                 f"{phys} | {stab} |")
    L.append("")
    wc = {d: r for d, r in conv.items() if r.get("window_fs")}
    if wc:
        L.append("### Closure residual vs simulated time (windows recorded inside each run)\n")
        L.append("| design | " + " | ".join(f"{t:.0f} fs" for t in next(iter(wc.values()))["window_fs"]) + " | full |")
        L.append("|---" * (len(next(iter(wc.values()))["window_fs"]) + 2) + "|")
        for d, r in wc.items():
            L.append(f"| {d} | " + " | ".join(fmt(v) for v in r["window_max_abs_A"]) +
                     f" | {fmt(r['max_abs_A'])} |")
        L.append("")
    if acc:
        L.append("## Accepted with / without ITO pairs\n")
        L.append("| design | case | run | reference | time | mesh | R(λ_ZE) | T(λ_ZE) | A(λ_ZE) | max\\|A\\| |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in acc:
            ge = r.get("glass_extra_nm") or 0.0
            mesh = f"{r['nxy']}²×{r['n_z']}" + (f" (+{ge:.0f} nm glass)" if ge else "")
            L.append(f"| {r['design']} | {'without ITO' if r['kind'] == 'noito' else 'with ITO'} | "
                     f"`{r['tag']}` | `{r['reference']}` | {r['time_fs']:.0f} fs | {mesh} | "
                     f"{fmt(r['R_ZE'])} | {fmt(r['T_ZE'])} | {r['A_ZE']:+.4f} | {fmt(r['max_abs_A'])} |")
        L.append("")
    L.append("## Files\n")
    L.append("* `<design>/<tag>/spectra_noito_fresh.csv` — λ, R, T, A and every early-window R/T of that run\n"
             "* `<design>/<tag>/spectra_noito_fresh.npz` — the same arrays\n"
             "* `<design>/<tag>/spectra_noito_fresh.png` — spectrum and the closure/convergence check\n"
             "* `<design>/spectra_noito_accepted.csv|npz` — the accepted run for that design\n"
             "* `<design>/compare_with_without_ITO.png` — accepted pair\n"
             "* `comparison/noito_fresh_convergence.json` — the acceptance test above\n"
             "* `comparison/overlay_RT_noITO_accepted.png` — all accepted no-ITO spectra\n")
    (COMP / "NOITO_RERUN.md").write_text("\n".join(L))
    print(f"wrote {COMP / 'NOITO_RERUN.md'} ({len(L)} lines)")


if __name__ == "__main__":
    main()
