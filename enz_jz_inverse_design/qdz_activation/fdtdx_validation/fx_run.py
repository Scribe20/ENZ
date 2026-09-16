"""FDTDX validation runs for the FROZEN finalists of the Q-Dz activation campaign.

Thin wrapper around the validated four-finalists driver (../../fdtdx_four_finalists/fx_sim.py): the SAME
scene builder, mesh, materials (materials/material_models.json of that package), source, PML, flux planes,
detectors, DFT settings, segmented/resumable run loop and output writer are used.  The only differences,
all passed as explicit arguments of the (backward-compatible) hooks added to build_scene:
    * geometry registry  -> this directory's manifest (register_finalists.py) instead of the four-finalists one
    * design directory   -> this directory (<candidate>/geometry/rho_hard_binary.npy is symlinked as
                            <candidate>/rho_hard_binary.npy so the loader path stays identical)
    * volume detector    -> deeper glass (default 24 coarse cells ~ 620 nm) and H components stored too
    * field wavelengths  -> the campaign's lambda_op(+), lambda_E and the loaded-resonance wavelengths
Outputs: <candidate>/raw_fields/<tag>/phasors.npz + run_meta.json (raw complex phasors of every detector),
reference/<tag>/ for the empty-cell reference run of the identical mesh.

    /opt/venv-fdtdx/bin/python fx_run.py --design <tag> [--reference] [--time-fs 600] [--field-lams ...]
                                          [--conv-check-fs 300 450] [--segment-steps 5000] [--ito-cells 5]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FX = HERE.parent.parent / "fdtdx_four_finalists"
sys.path.insert(0, str(FX))
os.environ.setdefault("FX_JAX_CACHE", str(FX / ".jaxcache"))
import fx_sim                                                     # noqa: E402  (validated driver, unchanged physics)

MANIFEST = HERE / "manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--no-ito", action="store_true")
    ap.add_argument("--ito-cells", type=int, default=5)
    ap.add_argument("--time-fs", type=float, default=600.0)
    ap.add_argument("--field-lams", type=float, nargs="*", default=None)
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--nxy", type=int, default=64)
    ap.add_argument("--courant", type=float, default=0.95)
    ap.add_argument("--glass-extra-nm", type=float, default=0.0)
    ap.add_argument("--vol-glass-cells", type=int, default=24)
    ap.add_argument("--vol-air-cells", type=int, default=8)
    ap.add_argument("--vol-H", action="store_true", help="also record Hx, Hy, Hz in the volume detector")
    ap.add_argument("--conv-check-fs", type=float, nargs="*", default=None)
    ap.add_argument("--segment-steps", type=int, default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=None)
    a = ap.parse_args()
    man = json.load(open(MANIFEST))
    if a.design not in man["candidates"]:
        raise SystemExit(f"{a.design} not registered; run register_finalists.py first")
    c = man["candidates"][a.design]
    prov = {a.design: dict(P_nm=c["P_nm"], h_nm=c["h_nm"])}
    ddir = HERE / a.design
    link = ddir / "rho_hard_binary.npy"
    if not link.exists():
        link.symlink_to(Path("geometry") / "rho_hard_binary.npy")
    field_lams = a.field_lams if a.field_lams else c["field_lams_nm"]
    out = (HERE / "reference" / f"{a.design}_{a.tag}") if a.reference else (ddir / "raw_fields" / a.tag)
    out.mkdir(parents=True, exist_ok=True)
    comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") if a.vol_H else ("Ex", "Ey", "Ez")
    objects, constraints, config, meta = fx_sim.build_scene(
        a.design, a.no_ito, a.ito_cells, a.time_fs * 1e-15, [l * 1e-9 for l in field_lams], reference=a.reference,
        courant=a.courant, nxy=a.nxy, subpixel=True, conv_check_fs=a.conv_check_fs, glass_extra=a.glass_extra_nm * 1e-9,
        vol_glass_cells=a.vol_glass_cells, vol_air_cells=a.vol_air_cells, prov=prov, design_dir=HERE, vol_components=comps)
    meta.update(campaign="qdz_activation", geometry_sha256=c["sha256_uint8_array"], geometry_source=c["source_rho"], manifest=str(MANIFEST),
                fx_sim_file=str(FX / "fx_sim.py"), material_models=str(FX / "materials" / "material_models.json"))
    print(f"[fx] {a.design} ref={a.reference} no_ito={a.no_ito} cells={meta['n_cells']} n_z={meta['idx']['n_z']} dt={meta['dt_s']*1e18:.2f} as "
          f"steps={meta['n_steps']} stride={meta['dft_stride']} vol z-cells={meta['idx'].get('z_vol1', 0) - meta['idx'].get('z_vol0', 0)} devices={meta['devices']}", flush=True)
    state_path = out / "state.npz" if a.segment_steps else None
    if a.restart and state_path is not None and state_path.exists():
        state_path.unlink()
    total = int(config.time_steps_total)
    arrays, objs, config, timing = fx_sim.run(objects, constraints, config, bench_steps=a.bench_steps, segment_steps=a.segment_steps, state_path=state_path,
                                              on_segment=(lambda ar, st, tot, tm: fx_sim.write_outputs(out, ar, meta, st, tot, tm)) if a.segment_steps else None)
    meta.update(timing=timing, bench_steps=a.bench_steps)
    m = fx_sim.write_outputs(out, arrays, meta, total if not a.bench_steps else int(config.time_steps_total), total if not a.bench_steps else int(config.time_steps_total), timing)
    print(f"[fx] done: setup {timing['setup_s']:.0f} s, run {timing['run_s']:.0f} s, {m['n_steps_run']} steps, final max|E| = {m['max_abs_E_final']:.3e} finite={m['finite']} -> {out}", flush=True)
    if state_path is not None and state_path.exists():
        state_path.unlink()


if __name__ == "__main__":
    main()
