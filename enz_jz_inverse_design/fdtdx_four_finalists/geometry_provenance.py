"""Freeze the four finalist geometries: copy the certified hard-binary arrays, record provenance
(path, SHA256, shape, P, h, pad, fill) and draw them at the same physical scale."""
import hashlib, json, platform, subprocess, sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DESIGNS = {
    "final3": dict(tag="final3_div_F1_P825_h500_pad0.12_s333_h525", P=825.0, h=525.0, pad=0.12),
    "final1": dict(tag="final1_F0_P825_h600_pad0.12_s1001_h575", P=825.0, h=575.0, pad=0.12),
    "final0": dict(tag="final0_F2_P850_h600_pad0.12_s333_P825", P=825.0, h=600.0, pad=0.12),
    "final2": dict(tag="final2_F0_P825_h600_pad0.12_s1001_pad0.08", P=825.0, h=600.0, pad=0.08),
}


def main():
    prov = {}
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, (name, d) in zip(axs, DESIGNS.items()):
        src = PKG / "outputs" / "stage3" / "runs" / d["tag"] / "rho_hard_binary.npy"
        cert = PKG / "outputs" / "stage4" / d["tag"] / "certify.json"
        raw = open(src, "rb").read()
        rho = np.load(src)
        c = json.load(open(cert))
        assert (c["P"], c["h"]) == (d["P"], d["h"]) and abs(c["pad_frac"] - d["pad"]) < 1e-6
        (HERE / name).mkdir(exist_ok=True)
        np.save(HERE / name / "rho_hard_binary.npy", rho)
        prov[name] = dict(source_file=str(src.relative_to(PKG.parent)), sha256=hashlib.sha256(raw).hexdigest(), sha256_copy=hashlib.sha256(open(HERE / name / "rho_hard_binary.npy", "rb").read()).hexdigest(),
                          shape=list(rho.shape), dtype=str(rho.dtype), P_nm=d["P"], h_nm=d["h"], pad_frac=d["pad"], pixel_nm=d["P"] / rho.shape[0],
                          fill_fraction=float(rho.mean()), n_material_pixels=int(rho.sum()), certification_file=str(cert.relative_to(PKG.parent)),
                          torcwa_reference={k: c[k] for k in ("Fz_certified", "Fx_certified", "Fy_certified", "Ftot_certified", "A_certified", "R_certified", "T_certified", "eta_z_abs", "mean_Ez2", "max_Ez2", "lam")},
                          array_convention="rho[i, j]: i -> x (axis 0), j -> y (axis 1), cell-centred pixels of size P/128; 1 = a-Si:H, 0 = air (pad already included)")
        ext = [0, d["P"], 0, d["P"]]
        ax.imshow(rho.T, origin="lower", extent=ext, cmap="gray_r", interpolation="nearest")
        ax.set_title(f"{name}: P = {d['P']:.0f} nm, h = {d['h']:.0f} nm, pad {d['pad']*100:.0f} %, fill {rho.mean():.3f}", fontsize=9)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]"); ax.set_aspect("equal")
    fig.suptitle("Frozen certified hard-binary geometries (128 × 128, 6.445-nm pixels; black = a-Si:H)", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "config" / "geometries_four_panel.png", dpi=170)
    json.dump(prov, open(HERE / "config" / "geometry_provenance.json", "w"), indent=1)
    # environment
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:
            return f"n/a ({e})"
    env = dict(platform=platform.platform(), python_system=sys.version.split()[0], cpu=sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"), n_cores=os.cpu_count(),
               mem_total_GB=float(sh("grep MemTotal /proc/meminfo | awk '{print $2/1048576}'") or 0), nvidia_smi=sh("nvidia-smi -L 2>&1 | head -2") or "not found", dev_nvidia=sh("ls /dev/nvidia* 2>&1 | head -1"),
               jax=sh("/opt/venv-fdtdx/bin/python -c \"import jax; print(jax.__version__, [str(d) for d in jax.devices()], jax.default_backend(), 'device_count', jax.device_count())\" 2>/dev/null"),
               fdtdx=sh("/opt/venv-fdtdx/bin/python -c \"import importlib.metadata as m; print(m.version('fdtdx'))\" 2>/dev/null"),
               fdtdx_source="supplied fdtdx-main.zip (cf0f5522-fdtdxmain.zip), installed editable in /opt/venv-fdtdx",
               fdtdx_source_sha256=hashlib.sha256(open("/root/.claude/uploads/f43368cf-8272-5146-85ca-a780e58f63a3/cf0f5522-fdtdxmain.zip", "rb").read()).hexdigest())
    json.dump(env, open(HERE / "config" / "environment.json", "w"), indent=1)
    print(json.dumps(env, indent=1)); print({k: (v["sha256"][:16], v["fill_fraction"]) for k, v in prov.items()})


if __name__ == "__main__":
    main()
