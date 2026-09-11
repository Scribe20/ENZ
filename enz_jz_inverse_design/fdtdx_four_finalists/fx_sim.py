"""FDTDX (JAX FDTD) driver for the four frozen finalists: periodic 825-nm cell, normal-incidence x-polarized
Gaussian-pulse TFSF plane wave from air, PML in z, rectilinear z-mesh (fine ITO cells), dispersive ITO (Drude +
Lorentz ADE), Lorentz a-Si:H and glass (fitted to the supplied data, materials/material_models.json).

    /opt/venv-fdtdx/bin/python fx_sim.py --design final3 [--no-ito] [--ito-cells 10] [--time-fs 300]
                                          [--field-lams 1302.282 1294 1298 1306 1310] [--tag prod] [--reference]
                                          [--bench-steps 200]

Saves outputs/<design>/<tag>/phasors.npz (complex64 phasors of every detector, grid edges, dt, wavelengths, indices)
and run_meta.json.  Post-processing is done separately (fx_post.py).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import jax

# Persistent compilation cache.  The container restarts roughly hourly and kills the runs; each restart
# otherwise re-traces and re-compiles the whole FDTD step function before any stepping happens, which
# cost ~16 min when one process restarted alone and ~44 min when both restarted together and competed
# for cores - i.e. most of the hour between restarts.  Caching the compiled kernels on disk makes the
# restart nearly free.  The cache is keyed on the computation, so it is only reused for an identical
# mesh and step function.
_CACHE = Path(os.environ.get("FX_JAX_CACHE", Path(__file__).resolve().parent / ".jaxcache"))
_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_CACHE))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)   # cache every entry
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)  # regardless of compile time
import jax.numpy as jnp

import fdtdx
from fdtdx.constants import c as C0
from fdtdx.objects.static_material.static import StaticMultiMaterialObject, UniformMaterialObject
from fdtdx.materials import compute_ordered_names
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.fdtd.fdtd import custom_fdtd_forward
from fdtdx.objects.object import RealCoordinateConstraint

HERE = Path(__file__).resolve().parent
MODELS = json.load(open(HERE / "materials" / "material_models.json"))
PROV = json.load(open(HERE / "config" / "geometry_provenance.json"))
NXY = 128           # default; overridden by --nxy (cells per period in x and y)
D_ITO = 23e-9
PML_CELLS = 16
LAM_SPEC = np.arange(1200.0, 1400.01, 2.0) * 1e-9            # R/T spectra
LAM_ZE = MODELS["lambda_ZE_data_nm"] * 1e-9


# --------------------------------------------------------------------------- materials
def material_from_model(name):
    m = MODELS[name]
    if name == "ITO":
        poles = (fdtdx.DrudePole(plasma_frequency=m["wp_rad_s"], damping=m["gamma_d_rad_s"]),
                 fdtdx.LorentzPole(resonance_frequency=m["w0_L_rad_s"], damping=m["gamma_L_rad_s"], delta_epsilon=m["delta_eps_L"]))
    else:
        poles = (fdtdx.LorentzPole(resonance_frequency=m["w0_rad_s"], damping=max(m["gamma_rad_s"], 1e9), delta_epsilon=m["delta_eps"]),)
    return fdtdx.Material(permittivity=m["eps_inf"], dispersion=fdtdx.DispersionModel(poles=poles))


def eps_model(name, lam_m):
    """Complex permittivity of the fitted model at wavelength(s) lam_m [m] (exp(-i w t))."""
    m = MODELS[name]; w = 2 * np.pi * C0 / np.asarray(lam_m)
    if name == "ITO":
        return m["eps_inf"] - m["wp_rad_s"] ** 2 / (w ** 2 + 1j * m["gamma_d_rad_s"] * w) + m["delta_eps_L"] * m["w0_L_rad_s"] ** 2 / (m["w0_L_rad_s"] ** 2 - w ** 2 - 1j * m["gamma_L_rad_s"] * w)
    return m["eps_inf"] + m["delta_eps"] * m["w0_rad_s"] ** 2 / (m["w0_rad_s"] ** 2 - w ** 2 - 1j * max(m["gamma_rad_s"], 1e9) * w)


# --------------------------------------------------------------------------- voxel pattern object
def area_fraction(rho, n):
    """Exact area fraction of the 128x128 binary design (pixel = P/128) inside each of n x n equal cells of the same
    period (cell = P/n): integral of rho over the cell divided by the cell area (computed with the overlap matrix of
    the two 1D partitions).  n = 128 reproduces rho exactly."""
    m = rho.shape[0]
    e_pix = np.arange(m + 1) / m; e_cell = np.arange(n + 1) / n
    W = np.zeros((n, m))
    for c in range(n):
        lo = np.maximum(e_cell[c], e_pix[:-1]); hi = np.minimum(e_cell[c + 1], e_pix[1:])
        W[c] = np.clip(hi - lo, 0.0, None) * n
    return W @ rho.astype(float) @ W.T


@autoinit
class VoxelPattern(StaticMultiMaterialObject):
    """Binary in-plane pattern (rho[i, j], i -> x, j -> y) extruded along z.  ``fill`` is the exact area fraction of
    a-Si:H in every simulation cell (1:1 with the design pixels when NXY = 128).  With subpixel_smoothing=True the
    fill fraction drives FDTDX's Farjadpour/Meep interface averaging; otherwise cells with fill >= 0.5 are a-Si:H."""
    material_name: str = frozen_field()
    fill: np.ndarray = frozen_field()

    def get_voxel_mask_for_shape(self) -> jax.Array:
        assert tuple(self.grid_shape[:2]) == tuple(self.fill.shape), (self.grid_shape, self.fill.shape)
        m2 = jnp.asarray(self.fill >= 0.5)
        return jnp.repeat(m2[:, :, None], self.grid_shape[2], axis=2)

    def get_fill_fraction_for_shape(self) -> jax.Array:
        f2 = jnp.asarray(self.fill.astype(float))
        return jnp.repeat(f2[:, :, None], self.grid_shape[2], axis=2)

    def get_material_mapping(self) -> jax.Array:
        idx = compute_ordered_names(self.materials).index(self.material_name)
        return jnp.ones(self.grid_shape, dtype=jnp.int32) * idx


# --------------------------------------------------------------------------- z mesh
def graded(d0, d1, ratio=1.25):
    """widths growing geometrically from d0 (excluded) towards d1 (last width == d1 not guaranteed)."""
    out, d = [], d0
    while True:
        d = d * ratio
        if d >= d1:
            break
        out.append(d)
    return out


def build_z_mesh(h, n_ito, dz_coarse, dz_fine, ratio=1.3, glass_bulk=1250e-9, glass_extra=0.0, air_gap_to_source=300e-9, source_to_R=120e-9, R_to_pml=100e-9, T_below_ito=300e-9):
    """Returns widths (bottom -> top) and index bookkeeping. Bottom = glass PML, top = air PML.
    The glass region is thick (default 1.25 um of bulk + PML) because the (+-1, 0) diffraction orders are evanescent
    in the glass only just below cut-off (n_glass P = 1251 nm): at 1302 nm their decay length is ~475 nm, so a PML
    closer than ~1 um produces spurious flux through the transmission plane (observed with a 350-nm glass region)."""
    dz_ito = D_ITO / n_ito
    dz_glass = dz_coarse                                                 # bulk-glass cells (25.8 nm at NXY = 64: 33 cells per wavelength in glass)
    # glass: [PML coarse][bulk coarse][graded coarse -> dz_ito]
    g_grade = graded(dz_ito, dz_glass, ratio)[::-1]                    # from coarse (below) down to fine (at ITO)
    n_bulk = int(round((glass_bulk + glass_extra) / dz_glass))
    glass = [dz_glass] * (PML_CELLS + n_bulk) + g_grade
    ito = [dz_ito] * n_ito
    # a-Si: graded dz_ito -> dz_fine, then uniform to exactly h
    a_grade = graded(dz_ito, dz_fine, ratio)
    rem = h - sum(a_grade)
    n_uni = int(round(rem / dz_fine))
    asi = a_grade + [rem / n_uni] * n_uni
    assert abs(sum(asi) - h) < 1e-15
    # air: graded dz_fine -> dz_coarse, uniform to source, R plane, PML
    air_grade = graded(dz_fine, dz_coarse, ratio)
    n_gap = int(round((air_gap_to_source - sum(air_grade)) / dz_coarse))
    n_sr = int(round(source_to_R / dz_coarse)); n_rp = int(round(R_to_pml / dz_coarse))
    air = air_grade + [dz_coarse] * (n_gap + n_sr + n_rp + PML_CELLS)
    widths = np.array(glass + ito + asi + air)
    zc = np.cumsum(glass)
    idx = dict(z_glass0=0, z_ito0=len(glass), z_asi0=len(glass) + n_ito, z_asi1=len(glass) + n_ito + len(asi),
               z_src=len(glass) + n_ito + len(asi) + len(air_grade) + n_gap,
               z_T=int(np.argmin(np.abs(zc - (zc[-1] - T_below_ito)))))                     # T plane ~T_below_ito under the ITO (in the bulk glass)
    idx["z_R"] = idx["z_src"] + n_sr
    idx["z_pml_top0"] = len(widths) - PML_CELLS
    idx["n_z"] = len(widths)
    return widths, idx


# --------------------------------------------------------------------------- scene
def build_scene(design, no_ito, n_ito, time_s, field_lams_m, reference=False, stride=None, courant=0.95, nxy=128, subpixel=True, dz_asi=None, conv_check_fs=None, glass_extra=0.0):
    global NXY
    NXY = nxy
    P = PROV[design]["P_nm"] * 1e-9; h = PROV[design]["h_nm"] * 1e-9
    rho = np.load(HERE / design / "rho_hard_binary.npy")
    fill = area_fraction(rho, NXY)
    dxy = P / NXY
    dz_fine = dxy if dz_asi is None else dz_asi
    widths, idx = build_z_mesh(h, n_ito, dz_coarse=max(2 * dxy, 12.0e-9), dz_fine=dz_fine, glass_extra=glass_extra)
    z_edges = np.concatenate([[0.0], np.cumsum(widths)])
    xy_edges = dxy * np.arange(NXY + 1)
    grid = fdtdx.RectilinearGrid(x_edges=jnp.asarray(xy_edges), y_edges=jnp.asarray(xy_edges), z_edges=jnp.asarray(z_edges))

    def at(obj, axes, idxs):
        """Real-coordinate placement of the '-' side of obj on the given edge indices (exact: edges coincide)."""
        coords = tuple(float((xy_edges if ax in (0, 1) else z_edges)[i]) for ax, i in zip(axes, idxs))
        return RealCoordinateConstraint(object=obj.name, axes=tuple(axes), sides=tuple("-" for _ in axes), coordinates=coords)
    config = fdtdx.SimulationConfig(time=time_s, grid=grid, backend="cpu", dtype=jnp.float32, courant_factor=courant)
    dt = config.time_step_duration
    if stride is None:
        stride = max(1, int(0.25e-15 / dt))                     # ~0.25-fs DFT sampling (f_s ~ 4e15 Hz >> 2 f_max ~ 5e14 Hz)
    air = fdtdx.Material(permittivity=1.0)
    glass = material_from_model("glass"); asi = material_from_model("aSiH"); ito = material_from_model("ITO")
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(NXY, NXY, idx["n_z"]), material=air)
    objects.append(volume)
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=PML_CELLS, override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"})
    bdict, blist = fdtdx.boundary_objects_from_config(bcfg, volume)
    constraints.extend(blist); objects.extend(bdict.values())

    def slab(name, material, z0, z1, color=None):
        o = UniformMaterialObject(name=name, material=material, partial_grid_shape=(NXY, NXY, z1 - z0))
        constraints.extend([o.same_size(volume, axes=(0, 1)), o.place_at_center(volume, axes=(0, 1)), at(o, (2,), (z0,))])
        objects.append(o)
    if not reference:
        slab("glass", glass, 0, idx["z_ito0"])
        slab("ito_layer", glass if no_ito else ito, idx["z_ito0"], idx["z_asi0"])
        pat = VoxelPattern(name="asi_pattern", materials={"air": air, "asi": asi}, material_name="asi", fill=fill, subpixel_smoothing=subpixel,
                           partial_grid_shape=(NXY, NXY, idx["z_asi1"] - idx["z_asi0"]))
        constraints.extend([pat.same_size(volume, axes=(0, 1)), pat.place_at_center(volume, axes=(0, 1)), at(pat, (2,), (idx["z_asi0"],))])
        objects.append(pat)
    # source (pulse centred at 1300 nm, sigma_f = 13 THz -> covers 1200-1400 nm with amplitude >= 0.3)
    center = fdtdx.WaveCharacter(wavelength=1300e-9)
    src = fdtdx.UniformPlaneSource(name="source", partial_grid_shape=(None, None, 1), wave_character=center, direction="-", fixed_E_polarization_vector=(1.0, 0.0, 0.0),
                                   temporal_profile=fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=13e12), center_wave=center))
    constraints.extend([src.same_size(volume, axes=(0, 1)), src.place_at_center(volume, axes=(0, 1)), at(src, (2,), (idx["z_src"],))])
    objects.append(src)
    wc_spec = [fdtdx.WaveCharacter(wavelength=float(l)) for l in LAM_SPEC]
    wc_field = [fdtdx.WaveCharacter(wavelength=float(l)) for l in field_lams_m]

    def plane_det(name, z, wcs, comps):
        d = fdtdx.PhasorDetector(name=name, partial_grid_shape=(NXY, NXY, 1), wave_characters=wcs, components=comps, scaling_mode="pulse", dft_subsample=stride, plot=False)
        constraints.extend([d.same_size(volume, axes=(0, 1)), d.place_at_center(volume, axes=(0, 1)), at(d, (2,), (z,))])
        objects.append(d)
    plane_det("R_plane", idx["z_R"], wc_spec, ("Ex", "Ey", "Hx", "Hy"))
    plane_det("T_plane", idx["z_T"], wc_spec, ("Ex", "Ey", "Hx", "Hy"))
    conv_list = [] if not conv_check_fs else [float(t) for t in np.atleast_1d(conv_check_fs)]
    for k, t_end in enumerate(conv_list):
        # Identical R/T flux planes whose DFT window is CLOSED EARLY.  Comparing the early-window and
        # full-window spectra from the SAME simulation is a direct, zero-extra-cost demonstration that
        # the recorded spectrum has stopped changing with simulation time (no post-processing involved).
        # With several windows the run also records its own convergence curve vs simulated time.
        sw = fdtdx.OnOffSwitch(end_time=t_end * 1e-15)
        sfx = "_early" if k == 0 else f"_early{k + 1}"
        for base, zz in (("R_plane", idx["z_R"]), ("T_plane", idx["z_T"])):
            d = fdtdx.PhasorDetector(name=base + sfx, partial_grid_shape=(NXY, NXY, 1), wave_characters=wc_spec,
                                     components=("Ex", "Ey", "Hx", "Hy"), scaling_mode="pulse",
                                     dft_subsample=stride, plot=False, switch=sw)
            constraints.extend([d.same_size(volume, axes=(0, 1)), d.place_at_center(volume, axes=(0, 1)), at(d, (2,), (zz,))])
            objects.append(d)
    if reference:
        plane_det("inc_ito_plane", idx["z_ito0"] + n_ito // 2, wc_spec, ("Ex", "Ey", "Hx", "Hy"))   # incident field amplitude at the ITO mid-plane position (air)
        plane_det("inc_field_plane", idx["z_ito0"] + n_ito // 2, wc_field, ("Ex", "Ey", "Ez"))
    else:
        # volume phasors: 4 coarse glass cells below the ITO .. 4 cells of air above the a-Si
        z0 = idx["z_ito0"] - 4; z1 = idx["z_asi1"] + 4
        v = fdtdx.PhasorDetector(name="vol_fields", partial_grid_shape=(NXY, NXY, z1 - z0), wave_characters=wc_field, components=("Ex", "Ey", "Ez"), scaling_mode="pulse", dft_subsample=stride, plot=False)
        constraints.extend([v.same_size(volume, axes=(0, 1)), v.place_at_center(volume, axes=(0, 1)), at(v, (2,), (z0,))])
        objects.append(v); idx["z_vol0"] = z0; idx["z_vol1"] = z1
        zp0, zp1 = PML_CELLS, idx["z_pml_top0"]
        xz = fdtdx.PhasorDetector(name="xz_plane", partial_grid_shape=(NXY, 1, zp1 - zp0), wave_characters=wc_field, components=("Ex", "Ey", "Ez"), scaling_mode="pulse", dft_subsample=stride, plot=False)
        constraints.extend([xz.same_size(volume, axes=(0,)), at(xz, (1, 2), (NXY // 2, zp0))])
        objects.append(xz)
        yz = fdtdx.PhasorDetector(name="yz_plane", partial_grid_shape=(1, NXY, zp1 - zp0), wave_characters=wc_field, components=("Ex", "Ey", "Ez"), scaling_mode="pulse", dft_subsample=stride, plot=False)
        constraints.extend([yz.same_size(volume, axes=(1,)), at(yz, (0, 2), (NXY // 2, zp0))])
        objects.append(yz); idx["z_plane0"] = zp0; idx["z_plane1"] = zp1
        # decay probes (time domain): one a-Si pixel near the centre, one ITO cell below it
        cand = np.argwhere(fill >= 0.999)
        ii, jj = cand[np.argmin(np.sum((cand - NXY // 2) ** 2, axis=1))]
        for name, z in (("probe_asi", idx["z_asi0"] + (idx["z_asi1"] - idx["z_asi0"]) // 2), ("probe_ito", idx["z_ito0"] + n_ito // 2)):
            pr = fdtdx.FieldDetector(name=name, partial_grid_shape=(1, 1, 1), components=("Ex", "Ez"), reduce_volume=True, plot=False)
            constraints.append(at(pr, (0, 1, 2), (int(ii), int(jj), int(z))))
            objects.append(pr)
        idx["probe_ij"] = [int(ii), int(jj)]
    meta = dict(design=design, no_ito=no_ito, reference=reference, P_m=P, h_m=h, dxy_m=dxy, nxy=NXY, subpixel_smoothing=subpixel, fill_fraction_mean=float(fill.mean()), n_ito=n_ito, dz_ito_m=D_ITO / n_ito, widths_m=widths.tolist(), z_edges_m=z_edges.tolist(),
                idx=idx, dt_s=dt, time_s=time_s, n_steps=int(config.time_steps_total), courant_factor=courant, dft_stride=stride, lam_spec_m=LAM_SPEC.tolist(), lam_field_m=list(map(float, field_lams_m)),
                source=dict(type="UniformPlaneSource TFSF, direction -z, E along x, GaussianPulseProfile center 1300 nm, sigma_f 13 THz (sigma_t 12.2 fs, peak at t0 = 6 sigma_t)"),
                pml_cells=PML_CELLS, n_cells=[NXY, NXY, idx["n_z"]], dtype="float32", backend=str(jax.default_backend()), devices=[str(d) for d in jax.devices()],
                conv_check_fs=conv_list, glass_extra_nm=glass_extra * 1e9)
    return objects, constraints, config, meta


# --------------------------------------------------------------------- segmented / resumable running
# A long run on this machine can outlive the container it runs in.  FDTD itself has no restart, but the
# whole dynamic state of a forward run is the ArrayContainer's fields (E, H and the PML/ADE auxiliaries)
# plus the detector accumulators; everything else is rebuilt deterministically by place_objects /
# apply_params.  So the run is executed in segments with fdtdx's custom_fdtd_forward, which takes an
# explicit (start_time, end_time) and does not reset the container, and the two dynamic sub-trees are
# written to disk after each segment.  Restarting the same command picks up from the last segment.
# The result is bit-identical to a single uninterrupted run apart from the order of floating-point
# reductions, because the state handed to the next segment is exactly the state the loop would have had.

STATE_PARTS = (("fields", lambda a: a.fields), ("det", lambda a: a.detector_states))


def _state_save(path, step, arrays):
    d = {"__step__": np.asarray(step, np.int64)}
    for tag, get in STATE_PARTS:
        leaves, _ = jax.tree_util.tree_flatten(get(arrays))
        for i, v in enumerate(leaves):
            d[f"{tag}/{i}"] = np.asarray(v)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **d)                      # write then rename: a reboot mid-write cannot corrupt it
    tmp.replace(path)


def _state_load(path, arrays):
    z = np.load(path)
    step = int(z["__step__"])
    for tag, get in STATE_PARTS:
        leaves, td = jax.tree_util.tree_flatten(get(arrays))
        new = jax.tree_util.tree_unflatten(td, [jnp.asarray(z[f"{tag}/{i}"]) for i in range(len(leaves))])
        arrays = arrays.aset("fields" if tag == "fields" else "detector_states", new)
    return step, arrays


def write_outputs(out, arrays, meta, steps_done, total_steps, timing):
    """Write phasors.npz / run_meta.json for the state reached so far.

    Called after every segment as well as at the end, so a long run that is interrupted (or that turns
    out to have converged before its nominal end) still leaves a usable spectrum on disk, marked with
    the number of steps actually simulated.  `complete` is the flag the drivers use to decide whether a
    job still needs work - the presence of phasors.npz no longer means the run finished.
    """
    save = {}
    for name, st in arrays.detector_states.items():
        for k, v in st.items():
            arr = np.asarray(v)
            save[f"{name}/{k}"] = arr[0] if (k == "phasor" and arr.shape[0] == 1) else arr
    tmp = out / "phasors.tmp.npz"
    np.savez_compressed(tmp, **save, z_edges_m=np.array(meta["z_edges_m"]), lam_spec_m=LAM_SPEC,
                        lam_field_m=np.array(meta["lam_field_m"]))
    tmp.replace(out / "phasors.npz")
    E = np.asarray(arrays.fields.E)
    m = dict(meta, n_steps_run=int(steps_done), steps_done=int(steps_done), n_steps_total=int(total_steps),
             complete=bool(steps_done >= total_steps), time_simulated_s=float(steps_done) * meta["dt_s"],
             timing=timing, max_abs_E_final=float(np.abs(E).max()), finite=bool(np.isfinite(E).all()))
    json.dump(m, open(out / "run_meta.json", "w"), indent=1)
    return m


def run(objects, constraints, config, bench_steps=None, segment_steps=None, state_path=None, on_segment=None):
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    objs, arrays, params, config, _ = fdtdx.place_objects(object_list=objects, config=config, constraints=constraints, key=key)
    arrays, objs, _ = fdtdx.apply_params(arrays, objs, params, key)
    t1 = time.time()
    if bench_steps:
        config = config.aset("time", bench_steps * config.time_step_duration)
    total = int(config.time_steps_total)
    if not segment_steps:
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objs, config=config, key=key)
        jax.block_until_ready(arrays.fields.E)
        return arrays, objs, config, dict(setup_s=t1 - t0, run_s=time.time() - t1, segments=0, resumed_from=0)
    arrays = arrays.reset()                 # run_fdtd resets internally; custom_fdtd_forward does not
    step, resumed = 0, 0
    if state_path is not None and state_path.exists():
        step, arrays = _state_load(state_path, arrays)
        resumed = step
        print(f"[fx] resuming from saved state at step {step}/{total}", flush=True)
    while step < total:
        end = min(step + int(segment_steps), total)
        st, arrays = custom_fdtd_forward(arrays=arrays, objects=objs, config=config, key=key,
                                         reset_container=False, record_detectors=True,
                                         start_time=step, end_time=end)
        jax.block_until_ready(arrays.fields.E)
        step = int(st)
        if state_path is not None:
            _state_save(state_path, step, arrays)
        if on_segment is not None:
            on_segment(arrays, step, total, dict(setup_s=t1 - t0, run_s=time.time() - t1))
        print(f"[fx] segment done: step {step}/{total} ({100 * step / total:.1f} %), "
              f"max|E| = {float(np.abs(np.asarray(arrays.fields.E)).max()):.3e}, "
              f"elapsed {time.time() - t1:.0f} s", flush=True)
    return arrays, objs, config, dict(setup_s=t1 - t0, run_s=time.time() - t1,
                                      segments=1, resumed_from=int(resumed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="final3")
    ap.add_argument("--no-ito", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--ito-cells", type=int, default=10)
    ap.add_argument("--time-fs", type=float, default=300.0)
    ap.add_argument("--field-lams", type=float, nargs="*", default=[MODELS["lambda_ZE_data_nm"], 1294.0, 1298.0, 1306.0, 1310.0])
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--courant", type=float, default=0.95, help="fraction of the exact 3D CFL limit of the rectilinear grid")
    ap.add_argument("--nxy", type=int, default=128, help="cells per period in x and y (128 = 1 cell per design pixel)")
    ap.add_argument("--no-subpixel", action="store_true", help="binary (>= 0.5 fill) cells instead of fill-fraction subpixel smoothing")
    ap.add_argument("--dz-asi", type=float, default=None, help="bulk a-Si:H z spacing [nm] (default = in-plane cell)")
    ap.add_argument("--glass-extra-nm", type=float, default=0.0,
                    help="extra bulk glass below the stack [nm].  Within a few nm of the glass Rayleigh cut-off "
                         "(n_glass P = 1251.2 nm) the (+-1,0) orders are evanescent with 1.3-1.9 um decay lengths, "
                         "so a large evanescent amplitude reaches the PML face and the CPML (a propagating-wave "
                         "absorber) mis-partitions the flux between the R and T planes, giving R > 1 with a "
                         "compensating T < 0 at fixed R + T = 1.  Deepening the glass removes it.")
    ap.add_argument("--conv-check-fs", type=float, nargs="*", default=None,
                    help="also record R/T flux phasors on DFT windows closed at these times [fs]; the early vs full "
                         "window comparison is the in-run convergence certificate, and several windows give the "
                         "run's own spectrum-vs-simulated-time convergence curve")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--bench-steps", type=int, default=None)
    ap.add_argument("--segment-steps", type=int, default=None,
                    help="run in segments of this many time steps, saving the full dynamic state after each "
                         "so the run resumes where it stopped if the machine restarts; re-running the same "
                         "command continues from the last saved segment")
    ap.add_argument("--restart", action="store_true", help="ignore any saved state and start from step 0")
    a = ap.parse_args()
    out = HERE / ("reference" if a.reference else a.design) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    objects, constraints, config, meta = build_scene(a.design, a.no_ito, a.ito_cells, a.time_fs * 1e-15, [l * 1e-9 for l in a.field_lams], reference=a.reference, stride=a.stride, courant=a.courant,
                                                     nxy=a.nxy, subpixel=not a.no_subpixel, dz_asi=(a.dz_asi * 1e-9 if a.dz_asi else None),
                                                     conv_check_fs=a.conv_check_fs, glass_extra=a.glass_extra_nm * 1e-9)
    print(f"[fx] {a.design} no_ito={a.no_ito} ref={a.reference} cells={meta['n_cells']} n_z={meta['idx']['n_z']} dt={meta['dt_s']*1e18:.2f} as steps={meta['n_steps']} stride={meta['dft_stride']} devices={meta['devices']}", flush=True)
    state_path = out / "state.npz" if a.segment_steps else None
    if a.restart and state_path is not None and state_path.exists():
        state_path.unlink()
    total_steps = int(config.time_steps_total)
    arrays, objs, config, timing = run(objects, constraints, config, bench_steps=a.bench_steps,
                                       segment_steps=a.segment_steps, state_path=state_path,
                                       on_segment=(lambda ar, st, tot, tm: write_outputs(out, ar, meta, st, tot, tm))
                                       if a.segment_steps else None)
    meta.update(timing=timing, bench_steps=a.bench_steps, n_steps_run=int(config.time_steps_total))
    m = write_outputs(out, arrays, meta, total_steps, total_steps, timing)
    print(f"[fx] done: setup {timing['setup_s']:.0f} s, run {timing['run_s']:.0f} s for {m['n_steps_run']} steps "
          f"({m['n_steps_run'] * np.prod(meta['n_cells']) / max(timing['run_s'], 1e-9):.2e} cell-steps/s); "
          f"final max|E| = {m['max_abs_E_final']:.3e} finite={m['finite']}", flush=True)
    if state_path is not None and state_path.exists():
        state_path.unlink()                 # the run finished; the resume state is no longer needed


if __name__ == "__main__":
    main()
