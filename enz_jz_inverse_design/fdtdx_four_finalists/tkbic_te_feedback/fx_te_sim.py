"""FDTDX runs of the frozen final3 geometry with the electron-temperature-dependent ITO models
(outputs/ito_te_models.json).  Everything except the ITO dispersion poles and the recorded wavelengths is
taken unchanged from the validated campaign driver ../fx_sim.py (same grid, source, PML / periodic
boundaries, detectors, subpixel pattern, glass / a-Si:H models); fx_sim.py itself is not modified.

    <python> fx_te_sim.py --reference [--time-fs 250]        # empty-domain reference on the Te wavelength grid
    <python> fx_te_sim.py --Te 3000 [--time-fs 400]           # final3 with the 3000 K ITO model

Outputs: runs/<tag>/phasors.npz + run_meta.json (same layout as the campaign runs).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
sys.path.insert(0, str(CAMP))
import fx_sim                                          # noqa: E402  (campaign driver; imports fdtdx / jax)
import jax                                             # noqa: E402

RUNS = HERE / "runs"
TE_MODELS = json.load(open(HERE / "outputs" / "ito_te_models.json"))
# spectral grid: the campaign 1200-1400 nm / 2 nm grid (300 K cross-check) plus 1230-1280 nm every 1 nm (study band)
LAM_SPEC_TE = np.unique(np.concatenate([np.arange(1200.0, 1400.01, 2.0), np.arange(1230.0, 1280.01, 1.0)])) * 1e-9
LAM_FIELD_CHECK = [1255e-9]                            # single volume-field wavelength for the ITO-loss (A = A_ITO) check


def ito_model_for(Te):
    m = TE_MODELS["models"][f"{Te:.0f}"]
    return dict(eps_inf=m["eps_inf"], wp_rad_s=m["wp_rad_s"], gamma_d_rad_s=m["gamma_d_rad_s"],
                delta_eps_L=m["delta_eps_L"], w0_L_rad_s=m["w0_L_rad_s"], gamma_L_rad_s=m["gamma_L_rad_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Te", type=float, default=None, help="electron temperature [K] of the ITO model")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--time-fs", type=float, default=400.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bench-steps", type=int, default=None)
    a = ap.parse_args()
    assert a.reference or a.Te is not None, "--Te or --reference"
    tag = a.tag or ("ref" if a.reference else f"Te{a.Te:.0f}")
    out = RUNS / tag
    out.mkdir(parents=True, exist_ok=True)

    # --- the only physics change: swap the ITO poles for the Te model (fx_sim.material_from_model reads fx_sim.MODELS)
    fx_sim.LAM_SPEC = LAM_SPEC_TE
    ito_used = None
    if not a.reference:
        ito_used = ito_model_for(a.Te)
        fx_sim.MODELS["ITO"] = dict(fx_sim.MODELS["ITO"], **ito_used)
    objects, constraints, config, meta = fx_sim.build_scene("final3", False, 5, a.time_fs * 1e-15, LAM_FIELD_CHECK,
                                                            reference=a.reference, nxy=64, subpixel=True)
    meta.update(Te_K=a.Te, ito_model=ito_used, ito_model_source=str(HERE / "outputs" / "ito_te_models.json"),
                fdtdx_source=dict(module=str(Path(fx_sim.fdtdx.__file__).resolve().parent)),
                lam_spec_m=LAM_SPEC_TE.tolist(), lam_field_m=[float(l) for l in LAM_FIELD_CHECK], campaign_driver=str(CAMP / "fx_sim.py"))
    print(f"[fx-te] tag={tag} Te={a.Te} ref={a.reference} cells={meta['n_cells']} dt={meta['dt_s']*1e18:.2f} as "
          f"steps={meta['n_steps']} stride={meta['dft_stride']} n_lam={len(LAM_SPEC_TE)} ito={ito_used}", flush=True)
    arrays, objs, config, timing = fx_sim.run(objects, constraints, config, bench_steps=a.bench_steps)
    meta.update(timing=timing, bench_steps=a.bench_steps, n_steps_run=int(config.time_steps_total))
    ds = arrays.detector_states
    save = {}
    for name, st in ds.items():
        for k, v in st.items():
            arr = np.asarray(v)
            save[f"{name}/{k}"] = arr[0] if (k == "phasor" and arr.shape[0] == 1) else arr
    np.savez_compressed(out / "phasors.npz", **save, z_edges_m=np.array(meta["z_edges_m"]), lam_spec_m=LAM_SPEC_TE,
                        lam_field_m=np.array(meta["lam_field_m"]))
    E = np.asarray(arrays.fields.E)
    meta.update(max_abs_E_final=float(np.abs(E).max()), finite=bool(np.isfinite(E).all()))
    json.dump(meta, open(out / "run_meta.json", "w"), indent=1)
    print(f"[fx-te] done {tag}: setup {timing['setup_s']:.0f} s, run {timing['run_s']:.0f} s for {meta['n_steps_run']} steps "
          f"({meta['n_steps_run'] * np.prod(meta['n_cells']) / max(timing['run_s'], 1e-9):.2e} cell-steps/s); "
          f"final max|E| = {meta['max_abs_E_final']:.3e} finite={meta['finite']}", flush=True)


if __name__ == "__main__":
    main()
