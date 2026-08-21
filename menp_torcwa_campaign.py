"""
menp_torcwa_campaign.py
=======================

MENP-based multipolar qualification of exactly THREE frozen binary
candidates from the ED/MD co-excitation search:

    P0870_H0100_seed029   (best [13,13] candidate, F_co ~ +4.086)
    P0830_H0100_seed011   (best order-convergence, F_co ~ +4.031/+3.993@15)
    P0900_H0100_seed011   (high-period representative, F_co ~ +4.002)

Method
------
* Dense full-3D complex E fields are reconstructed on the frozen binary
  geometry with torcwa `field_xy` on an (n_xy x n_xy x nz) grid spanning one
  unit cell x the full patterned-layer thickness (z endpoints included for
  trapezoidal integration, matching MENP's trapz convention).
* J = -i*omega*eps0*(n^2-1)*E exactly as MENP's E2J (TORCWA and MENP share
  the exp(-i*omega*t) convention; TORCWA E is in units where E_inc = 1,
  which matches MENP's |E0| = 1 cross-section normalization; positions are
  converted to meters, frequencies to Hz).
* Primary decomposition: the VALIDATED exact_me port (Alaee 2018 Table 2)
  in 'corrected' mode (symmetrized Qm, regularized kernels); toroidal_me
  provides T and the long-wavelength p for the anapole diagnostic.
* Background-medium note: MENP's cross sections assume an isolated
  scatterer in vacuum. Here the moments are PER-UNIT-CELL moments of the
  induced current computed from the true substrate-aware periodic fields;
  the C's are formal vacuum radiation weights used for relative comparison.
  The `substrate` stage quantifies the substrate's effect on the induced
  currents by re-solving with the input half-space set to air.

Stages (each checkpointed to results_ed_md_coexcitation/menp/<cand>/):
  fine       lambda = 1332.5 +/- 12 nm step 0.5, order [13,13], 96x96x21
  orders     [9,9],[11,11],[13,13],[15,15],[17,17] at 1332.5 nm
  grids      spatial-integration convergence at 1332.5 nm, [13,13]
  origins    origin sensitivity (no re-solve; reuses saved dense fields)
  substrate  input half-space air vs silica, [11,11] scan + [13,13] point
  material   parametric NIR a-Si:H check n_Si in {3.30, 3.45} (NO
             trustworthy tabulated local NIR dataset exists - documented)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torcwa                                     # noqa: E402
from menp_port import C0, exact_me, toroidal_me   # noqa: E402
import coexcite_ed_md_sweep as sweep              # noqa: E402

RESULTS = Path(__file__).resolve().parent / 'results_ed_md_coexcitation'
MENP_OUT = RESULTS / 'menp'

CANDIDATES = {
    'P0870_H0100_seed029': 'P0870_H0100_seed029_lam1332p5_refined',
    'P0830_H0100_seed011': 'P0830_H0100_seed011_lam1332p5_refined',
    'P0900_H0100_seed011': 'P0900_H0100_seed011_lam1332p5_refined',
}

LAM0 = 1332.5
FINE_LAMS = np.arange(LAM0 - 12.0, LAM0 + 12.0 + 0.01, 0.5)
ORDER_LIST = [[9, 9], [11, 11], [13, 13], [15, 15], [17, 17]]
GRID_LIST = [(64, 11), (64, 21), (96, 21), (128, 21), (96, 41), (128, 41)]
N_XY, NZ = 96, 21          # defaults: 96 > 2*(2*17+1) = 70 -> alias-free
                           # sampling of |E|^2 products up to order [17,17]


def load_candidate(name):
    d = RESULTS / 'refine' / CANDIDATES[name]
    cfg = json.loads((d / 'config.json').read_text())
    rho = np.load(d / 'rho_binary.npy')
    assert set(np.unique(rho)) <= {0.0, 1.0}
    return {'dir': d, 'cfg': cfg, 'rho': rho,
            'P': float(cfg['period_nm']), 'H': float(cfg['height_nm'])}


def solve_dense(rho_np, P, H, lam_nm, order, n_xy=N_XY, nz=NZ,
                substrate_eps=None, si_eps=None):
    """Solve one RCWA problem and reconstruct dense 3D complex E (and H)
    on the unit cell x layer volume. Returns dict of numpy arrays."""
    if substrate_eps is None:
        substrate_eps = sweep.SUBSTRATE_EPS
    if si_eps is None:
        si_eps = sweep.silicon_eps(lam_nm)
    if not torch.is_tensor(si_eps):
        si_eps = torch.tensor(si_eps, dtype=torch.complex64)
    rho = torch.tensor(rho_np, dtype=sweep.GEO_DTYPE, device=sweep.DEVICE)

    L = [float(P), float(P)]
    sim = torcwa.rcwa(freq=1.0 / lam_nm, order=list(order), L=L,
                      dtype=sweep.SIM_DTYPE, device=sweep.DEVICE)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    layer_eps = rho * si_eps + (1.0 - rho) * 1.0
    sim.add_layer(thickness=float(H), eps=layer_eps)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1.0, 0.0], direction='forward')

    x_ax = torch.linspace(0.0, float(P), n_xy, dtype=sweep.GEO_DTYPE,
                          device=sweep.DEVICE)
    z_ax = np.linspace(0.0, float(H), nz)
    E = np.empty((3, n_xy, n_xy, nz), np.complex64)
    Hf = np.empty((3, n_xy, n_xy, nz), np.complex64)
    with torch.no_grad():
        for kz, zp in enumerate(z_ax):
            Ev, Hv = sim.field_xy(0, x_ax, x_ax, z_prop=float(zp))
            for c in range(3):
                E[c, :, :, kz] = Ev[c].cpu().numpy()
                Hf[c, :, :, kz] = Hv[c].cpu().numpy()

    # refractive-index grid from the same binary mask (periodic sampling)
    nmask = rho_np.shape[0]
    idx = (np.floor(x_ax.cpu().numpy() / P * nmask).astype(int)) % nmask
    mask = rho_np[np.ix_(idx, idx)]
    n_si = np.sqrt(complex(si_eps))
    n2d = np.where(mask > 0.5, n_si, 1.0 + 0j)
    n3d = np.repeat(n2d[:, :, None], nz, axis=2).astype(np.complex64)

    # original-style 3-slice proxies from the same solve, on the original
    # 64-point exact-mean grid (exact for orders <= [15,15]; for [17,17]
    # the dense-volume proxies below are the alias-free reference)
    Ex3, Hz3, _ = sweep.compute_fields(sim, P, H)
    S_ED3, S_MD3 = sweep.compute_mode_scores(Ex3, Hz3)

    return {'x_nm': x_ax.cpu().numpy(), 'z_nm': z_ax, 'E': E, 'H': Hf,
            'n3d': n3d, 'P': P, 'Hh': H, 'lam_nm': lam_nm,
            'S_ED_3slice': float(S_ED3), 'S_MD_3slice': float(S_MD3),
            'S_ED_vol': float(np.mean(np.abs(E[0]) ** 2)),
            'S_MD_vol': float(np.mean(np.abs(Hf[2]) ** 2) / sweep.N_SUB ** 2)}


def decompose(fields, origin_nm=None, mode='corrected'):
    """Run the validated exact + toroidal decompositions about origin_nm
    (defaults to the unit-cell/layer center)."""
    P, Hh = fields['P'], fields['Hh']
    if origin_nm is None:
        origin_nm = (P / 2, P / 2, Hh / 2)
    x = (fields['x_nm'] - origin_nm[0]) * 1e-9
    y = (fields['x_nm'] - origin_nm[1]) * 1e-9
    z = (fields['z_nm'] - origin_nm[2]) * 1e-9
    f = np.array([C0 / (fields['lam_nm'] * 1e-9)])
    E4 = [fields['E'][c][..., None].astype(np.complex128) for c in range(3)]
    n4 = fields['n3d'][..., None].astype(np.complex128)
    ex = exact_me(x, y, z, f, *E4, n4, n4, n4, mode=mode)
    to = toroidal_me(x, y, z, f, *E4, n4, n4, n4, mode=mode)
    rec = {
        'lam_nm': fields['lam_nm'],
        'p': ex['p'][:, 0], 'm': ex['m'][:, 0],
        'T': to['T'][:, 0], 'p_lw': to['p'][:, 0],
        'Cp': float(ex['Cp'][0]), 'Cm': float(ex['Cm'][0]),
        'CQe': float(ex['CQe'][0]), 'CQm': float(ex['CQm'][0]),
        'Csum': float(ex['Csum'][0]),
        'CT': float(to['CT'][0]), 'CpT': float(to['CpT'][0]),
        'S_ED_3slice': fields['S_ED_3slice'], 'S_MD_3slice': fields['S_MD_3slice'],
        'S_ED_vol': fields['S_ED_vol'], 'S_MD_vol': fields['S_MD_vol'],
    }
    return rec


def rec_to_row(rec):
    row = {'lam_nm': rec['lam_nm']}
    for i, c in enumerate('xyz'):
        row[f'p{c}_re'] = float(np.real(rec['p'][i]))
        row[f'p{c}_im'] = float(np.imag(rec['p'][i]))
        row[f'm{c}_re'] = float(np.real(rec['m'][i]))
        row[f'm{c}_im'] = float(np.imag(rec['m'][i]))
        row[f'T{c}_re'] = float(np.real(rec['T'][i]))
        row[f'T{c}_im'] = float(np.imag(rec['T'][i]))
    for k in ['Cp', 'Cm', 'CQe', 'CQm', 'Csum', 'CT', 'CpT',
              'S_ED_3slice', 'S_MD_3slice', 'S_ED_vol', 'S_MD_vol']:
        row[k] = rec[k]
    return row


def save_rows(path, rows):
    import csv
    with open(path, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_partial(path, keyfn):
    """Load an existing (possibly partial) stage CSV; return (rows, keys)
    so a restarted stage can skip already-computed entries."""
    import csv as _csv
    rows, keys = [], set()
    if Path(path).exists():
        for r in _csv.DictReader(open(path)):
            rows.append(r)
            keys.add(keyfn(r))
    return rows, keys


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_fine(cand, name, out):
    path = out / 'fine_decomposition.csv'
    rows, done = load_partial(path, lambda r: round(float(r['lam_nm']), 2))
    if len(done) >= len(FINE_LAMS):
        log(f'{name}: fine complete - skip')
        return
    for lam in FINE_LAMS:
        if round(float(lam), 2) in done:
            continue
        t0 = time.time()
        fl = solve_dense(cand['rho'], cand['P'], cand['H'], float(lam), [13, 13])
        rec = decompose(fl)
        rows.append(rec_to_row(rec))
        log(f"{name}: fine lam={lam:7.1f} |px|={abs(rec['p'][0]):.3e} "
            f"|mz|={abs(rec['m'][2]):.3e} Cp={rec['Cp']:.3e} Cm={rec['Cm']:.3e} "
            f"({time.time()-t0:.0f}s)")
        if abs(lam - LAM0) < 1e-9:
            np.savez_compressed(out / 'dense_fields_target.npz',
                                **{k: v for k, v in fl.items()
                                   if k in ('x_nm', 'z_nm', 'E', 'H', 'n3d')},
                                P=fl['P'], Hh=fl['Hh'], lam_nm=fl['lam_nm'],
                                note='torcwa units E_inc=1; [13,13]; binary geometry')
        save_rows(path, rows)   # checkpoint every lambda


def stage_orders(cand, name, out):
    path = out / 'order_scan.csv'
    rows, done = load_partial(path, lambda r: int(r['order']))
    if len(done) >= len(ORDER_LIST):
        log(f'{name}: orders complete - skip')
        return
    for order in ORDER_LIST:
        if order[0] in done:
            continue
        t0 = time.time()
        try:
            fl = solve_dense(cand['rho'], cand['P'], cand['H'], LAM0, order)
            rec = rec_to_row(decompose(fl))
            rec['order'] = order[0]
            rows.append(rec)
            log(f"{name}: order {order} done ({time.time()-t0:.0f}s)")
            save_rows(path, rows)
        except Exception as e:
            log(f"{name}: order {order} FAILED: {e}")


def stage_grids(cand, name, out):
    path = out / 'grid_scan.csv'
    rows, done = load_partial(path, lambda r: (int(r['n_xy']), int(r['nz'])))
    if len(done) >= len(GRID_LIST):
        log(f'{name}: grids complete - skip')
        return
    for (nxy, nz) in GRID_LIST:
        if (nxy, nz) in done:
            continue
        fl = solve_dense(cand['rho'], cand['P'], cand['H'], LAM0, [13, 13],
                         n_xy=nxy, nz=nz)
        rec = rec_to_row(decompose(fl))
        rec['n_xy'], rec['nz'] = nxy, nz
        rows.append(rec)
        save_rows(path, rows)   # checkpoint per grid point
        log(f"{name}: grid {nxy}x{nxy}x{nz} done")


def stage_origins(cand, name, out):
    path = out / 'origin_scan.csv'
    if path.exists():
        log(f'{name}: origins exists - skip')
        return
    dfz = np.load(out / 'dense_fields_target.npz')
    fl = {k: dfz[k] for k in ('x_nm', 'z_nm', 'E', 'H', 'n3d')}
    fl.update({'P': float(dfz['P']), 'Hh': float(dfz['Hh']),
               'lam_nm': float(dfz['lam_nm']),
               'S_ED_3slice': np.nan, 'S_MD_3slice': np.nan,
               'S_ED_vol': np.nan, 'S_MD_vol': np.nan})
    P, Hh = fl['P'], fl['Hh']
    # J-weighted centroid origin
    Jw = np.abs(fl['E'][0]) ** 2 + np.abs(fl['E'][1]) ** 2 + np.abs(fl['E'][2]) ** 2
    Jw = Jw * (np.abs(fl['n3d'] ** 2 - 1) > 0.1)
    X, Y, Z = np.meshgrid(fl['x_nm'], fl['x_nm'], fl['z_nm'], indexing='ij')
    cx, cy, cz = (float((Jw * A).sum() / Jw.sum()) for A in (X, Y, Z))
    origins = {
        'center': (P / 2, P / 2, Hh / 2),
        'x+P8': (P / 2 + P / 8, P / 2, Hh / 2),
        'x-P8': (P / 2 - P / 8, P / 2, Hh / 2),
        'y+P8': (P / 2, P / 2 + P / 8, Hh / 2),
        'z+H4': (P / 2, P / 2, Hh / 2 + Hh / 4),
        'z-H4': (P / 2, P / 2, Hh / 2 - Hh / 4),
        'J-centroid': (cx, cy, cz),
    }
    rows = []
    for tag, o in origins.items():
        rec = rec_to_row(decompose(fl, origin_nm=o))
        rec['origin'] = tag
        rec['ox_nm'], rec['oy_nm'], rec['oz_nm'] = o
        rows.append(rec)
        log(f"{name}: origin {tag} = ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f}) nm done")
    save_rows(path, rows)


def stage_substrate(cand, name, out):
    path = out / 'substrate_scan.csv'
    rows, done = load_partial(
        path, lambda r: (r['substrate'], round(float(r['lam_nm']), 2), int(r['order'])))
    if len(done) >= 24:
        log(f'{name}: substrate complete - skip')
        return
    for sub_tag, sub_eps in [('silica', sweep.SUBSTRATE_EPS), ('air', 1.0)]:
        for lam in np.arange(LAM0 - 10, LAM0 + 10.01, 2.0):
            key = (sub_tag, round(float(lam), 2), 11)
            if key in done:
                continue
            fl = solve_dense(cand['rho'], cand['P'], cand['H'], float(lam),
                             [11, 11], substrate_eps=sub_eps)
            rec = rec_to_row(decompose(fl))
            rec['substrate'], rec['order'] = sub_tag, 11
            rows.append(rec)
            save_rows(path, rows)   # checkpoint per lambda
        if (sub_tag, round(LAM0, 2), 13) not in done:
            fl = solve_dense(cand['rho'], cand['P'], cand['H'], LAM0, [13, 13],
                             substrate_eps=sub_eps)
            rec = rec_to_row(decompose(fl))
            rec['substrate'], rec['order'] = sub_tag, 13
            rows.append(rec)
            save_rows(path, rows)
        log(f"{name}: substrate={sub_tag} done")


def stage_material(cand, name, out):
    """Parametric NIR a-Si:H sensitivity: NO trustworthy tabulated local
    dataset exists at 1332.5 nm (aSiH.txt ends at 999 nm), so n_Si = 3.30
    and 3.45 (literature NIR a-Si:H range) are scanned parametrically.
    The resonance shifts, so each n gets a proxy scan to relocate the
    joint peak, then a fine decomposition around it."""
    path = out / 'material_scan.csv'
    rows, done = load_partial(
        path, lambda r: (float(r['n_si']), round(float(r['lam_nm']), 2)))
    if len(done) >= 34:
        log(f'{name}: material complete - skip')
        return
    done_nsi = {k[0] for k in done}
    for n_si in [3.30, 3.45]:
        if sum(1 for k in done if k[0] == n_si) >= 17:
            continue
        si_eps = complex(n_si ** 2, 2 * n_si * 0.003)   # small NIR loss
        # coarse proxy relocation scan
        best_lam, best_fom = None, -1e30
        for lam in np.arange(1350.0, 1650.1, 5.0):
            fl = solve_dense(cand['rho'], cand['P'], cand['H'], float(lam),
                             [9, 9], si_eps=si_eps, n_xy=64, nz=7)
            fom = np.log(fl['S_ED_vol'] + 1e-12) + np.log(fl['S_MD_vol'] + 1e-12)
            if fom > best_fom:
                best_fom, best_lam = fom, float(lam)
        log(f"{name}: n_si={n_si} joint proxy peak near {best_lam} nm")
        for lam in np.arange(best_lam - 8, best_lam + 8.01, 1.0):
            fl = solve_dense(cand['rho'], cand['P'], cand['H'], float(lam),
                             [11, 11], si_eps=si_eps)
            rec = rec_to_row(decompose(fl))
            rec['n_si'], rec['order'] = n_si, 11
            rows.append(rec)
            save_rows(path, rows)
        log(f"{name}: material n_si={n_si} done")


STAGES = {'fine': stage_fine, 'orders': stage_orders, 'grids': stage_grids,
          'origins': stage_origins, 'substrate': stage_substrate,
          'material': stage_material}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True, choices=list(CANDIDATES))
    ap.add_argument('--stage', required=True, choices=list(STAGES))
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cand = load_candidate(args.candidate)
    out = MENP_OUT / args.candidate
    out.mkdir(parents=True, exist_ok=True)
    log(f"{args.candidate}: stage {args.stage} starting "
        f"(P={cand['P']}, H={cand['H']}, mask {cand['rho'].shape})")
    STAGES[args.stage](cand, args.candidate, out)
    log(f"{args.candidate}: stage {args.stage} complete")


if __name__ == '__main__':
    main()
