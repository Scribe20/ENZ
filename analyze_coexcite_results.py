"""
analyze_coexcite_results.py
===========================

Post-processing for the ED/MD co-excitation topology search
(`coexcite_ed_md_sweep.py`):

  --mode summary    build summary_discovery.csv (or summary_refine.csv)
  --mode select     rank by F_co + greedy geometric-diversity selection
                    (metric: 1 - IoU of binary masks resampled to a common
                    64x64 normalized-coordinate grid; threshold configurable)
  --mode contact    contact-sheet figure of representative topologies
  --mode spectra    per-candidate S_ED(lambda)/S_MD(lambda) plots + peak table
  --mode final      final compact representative-solution table

Pure post-processing: no simulation, no gradients.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / 'results_ed_md_coexcitation'
COMMON_GRID = 64
IOU_THRESHOLD = 0.30      # minimum (1 - IoU) distance to count as "distinct"


# ---------------------------------------------------------------------------

def load_runs(stage='discovery'):
    runs = []
    root = RESULTS_ROOT / stage
    if not root.exists():
        return runs
    for d in sorted(root.iterdir()):
        cfg_path = d / 'config.json'
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        cfg['_dir'] = d
        runs.append(cfg)
    return runs


def build_summary(stage='discovery'):
    runs = load_runs(stage)
    rows = []
    for cfg in runs:
        fs = cfg.get('final_scores', {})
        proj = fs.get('projected', {})
        binr = fs.get('binary', {})
        best = cfg.get('best', {})
        rows.append({
            'run_id': cfg['run_id'],
            'period_nm': cfg['period_nm'],
            'height_nm': cfg['height_nm'],
            'seed': cfg['seed'],
            'target_wavelength_nm': cfg['target_wavelength_nm'],
            'final_Fco': proj.get('F_co', float('nan')),
            'best_Fco': best.get('F_co', float('nan')),
            'final_S_ED': proj.get('S_ED', float('nan')),
            'final_S_MD': proj.get('S_MD', float('nan')),
            'balance': proj.get('balance', float('nan')),
            'binary_Fco': binr.get('F_co', float('nan')),
            'binary_S_ED': binr.get('S_ED', float('nan')),
            'binary_S_MD': binr.get('S_MD', float('nan')),
            'fill_fraction': cfg.get('fill_fraction', float('nan')),
            'nx': cfg['nx'], 'ny': cfg['ny'],
            'fourier_order_x': cfg['fourier_order'][0],
            'fourier_order_y': cfg['fourier_order'][1],
            'runtime_s': cfg.get('runtime_s', float('nan')),
            'status': cfg.get('status', 'unknown'),
        })
    rows.sort(key=lambda r: (-(r['final_Fco'] if np.isfinite(r['final_Fco']) else -1e9)))
    out = RESULTS_ROOT / f'summary_{stage}.csv'
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {out} ({len(rows)} runs, "
              f"{sum(r['status'] == 'completed' for r in rows)} completed)")
    return rows


# ---------------------------------------------------------------------------

def resample_mask(mask, n=COMMON_GRID):
    """Nearest-neighbour resample of a binary mask onto a common n x n
    normalized-coordinate grid."""
    ix = (np.arange(n) + 0.5) / n * mask.shape[0]
    iy = (np.arange(n) + 0.5) / n * mask.shape[1]
    return mask[ix.astype(int).clip(max=mask.shape[0] - 1)][:,
                iy.astype(int).clip(max=mask.shape[1] - 1)]


def geometry_distance(mask_a, mask_b):
    """1 - IoU on the common normalized grid (0 = identical, 1 = disjoint)."""
    a, b = mask_a > 0.5, mask_b > 0.5
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return 1.0 - np.logical_and(a, b).sum() / union


def select_candidates(n_min=10, n_max=15, threshold=IOU_THRESHOLD):
    rows = build_summary('discovery')
    ok = [r for r in rows if r['status'] == 'completed' and np.isfinite(r['final_Fco'])]
    masks = {}
    for r in ok:
        d = RESULTS_ROOT / 'discovery' / r['run_id']
        try:
            masks[r['run_id']] = resample_mask(np.load(d / 'rho_binary.npy'))
        except Exception:
            pass
    ok = [r for r in ok if r['run_id'] in masks]

    selected, sel_dists = [], []
    for r in ok:                       # ok is F_co-ranked already
        if len(selected) >= n_max:
            break
        dmin = min((geometry_distance(masks[r['run_id']], masks[s['run_id']])
                    for s in selected), default=1.0)
        if dmin >= threshold or len(selected) == 0:
            selected.append(r)
            sel_dists.append(dmin)
    # relax threshold if too few distinct candidates
    thr = threshold
    while len(selected) < n_min and thr > 0.05:
        thr *= 0.7
        for r in ok:
            if len(selected) >= n_min:
                break
            if any(s['run_id'] == r['run_id'] for s in selected):
                continue
            dmin = min(geometry_distance(masks[r['run_id']], masks[s['run_id']])
                       for s in selected)
            if dmin >= thr:
                selected.append(r)
                sel_dists.append(dmin)

    report = {
        'metric': '1 - IoU of binary masks on common %dx%d normalized grid' % (COMMON_GRID, COMMON_GRID),
        'threshold': threshold,
        'relaxed_final_threshold': thr,
        'n_completed': len(ok),
        'selected': [
            {'rank': i + 1, 'run_id': s['run_id'], 'final_Fco': s['final_Fco'],
             'final_S_ED': s['final_S_ED'], 'final_S_MD': s['final_S_MD'],
             'balance': s['balance'],
             'min_distance_to_previously_selected': round(d, 4)}
            for i, (s, d) in enumerate(zip(selected, sel_dists))],
        'full_ranked_list': [r['run_id'] for r in ok],
    }
    out = RESULTS_ROOT / 'selection_diversity.json'
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}: {len(selected)} diversity-selected candidates "
          f"(metric {report['metric']}, threshold {threshold})")
    for s in report['selected']:
        print(f"  #{s['rank']:2d} {s['run_id']}  F_co={s['final_Fco']:+.4f} "
              f"S_ED={s['final_S_ED']:.3f} S_MD={s['final_S_MD']:.3f} "
              f"bal={s['balance']:.3f} dmin={s['min_distance_to_previously_selected']}")
    return report


# ---------------------------------------------------------------------------

def contact_sheet(stage='refine', out_name='contact_sheet.png', max_panels=15):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    runs = [c for c in load_runs(stage) if c.get('status') == 'completed']
    runs.sort(key=lambda c: -c.get('final_scores', {}).get('projected', {})
              .get('F_co', -1e9))
    runs = runs[:max_panels]
    if not runs:
        print(f'no completed runs in stage {stage}')
        return
    ncol = 4
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.6 * nrow))
    axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis('off')
    for ax, cfg in zip(axs.flat, runs):
        mask = np.load(cfg['_dir'] / 'rho_binary.npy')
        fs = cfg.get('final_scores', {}).get('projected', {})
        ax.imshow(mask.T, origin='lower', cmap='gray_r', extent=[0, 1, 0, 1])
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"P={cfg['period_nm']:.0f} h={cfg['height_nm']:.0f} "
            f"seed={cfg['seed']}\n"
            f"F_co={fs.get('F_co', float('nan')):+.3f}  "
            f"S_ED={fs.get('S_ED', float('nan')):.2f}  "
            f"S_MD={fs.get('S_MD', float('nan')):.2f}", fontsize=8)
    fig.suptitle(f"ED/MD co-excitation - representative freeform topologies "
                 f"({stage}, lam0=%.1f nm)" % runs[0]['target_wavelength_nm'],
                 fontsize=11)
    fig.tight_layout()
    out = RESULTS_ROOT / out_name
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out} ({len(runs)} panels)")


# ---------------------------------------------------------------------------

def extract_peak(lams, vals):
    """Return (lambda_peak, is_interior). Peak must not sit on the sweep edge
    to count as meaningful."""
    i = int(np.argmax(vals))
    interior = 0 < i < len(vals) - 1
    if interior and i > 0 and i < len(vals) - 1:
        # parabolic sub-sample refinement
        y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-30:
            di = 0.5 * (y0 - y2) / denom
            di = float(np.clip(di, -1, 1))
            lam_peak = lams[i] + di * (lams[1] - lams[0])
        else:
            lam_peak = lams[i]
    else:
        lam_peak = lams[i]
    return float(lam_peak), bool(interior)


def spectra_report(stage='refine'):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    rows = []
    for cfg in load_runs(stage):
        spath = cfg['_dir'] / 'spectra.csv'
        if not spath.exists():
            continue
        dat = list(csv.DictReader(open(spath)))
        lams = np.array([float(r['wavelength_nm']) for r in dat])
        sed = np.array([float(r['S_ED']) for r in dat])
        smd = np.array([float(r['S_MD']) for r in dat])
        t0 = np.array([float(r['T0']) for r in dat])
        lam_ed, ed_ok = extract_peak(lams, sed)
        lam_md, md_ok = extract_peak(lams, smd)
        split = abs(lam_ed - lam_md) if (ed_ok and md_ok) else float('nan')
        rows.append({'run_id': cfg['run_id'],
                     'lambda_peak_ED': lam_ed if ed_ok else float('nan'),
                     'lambda_peak_MD': lam_md if md_ok else float('nan'),
                     'ED_peak_interior': ed_ok, 'MD_peak_interior': md_ok,
                     'ED_MD_peak_splitting_nm': split})

        fig, ax1 = plt.subplots(figsize=(7, 4.2))
        ax1.plot(lams, sed, 'o-', color='tab:red', label='S_ED', ms=3)
        ax1.plot(lams, smd, 's-', color='tab:blue', label='S_MD', ms=3)
        ax1.axvline(cfg['target_wavelength_nm'], color='k', ls='--', lw=1,
                    label='target lam0')
        ax1.set_xlabel('wavelength (nm)')
        ax1.set_ylabel('modal-response proxy')
        ax1.set_yscale('log')
        ax2 = ax1.twinx()
        ax2.plot(lams, t0, '-', color='0.6', lw=1, label='T0')
        ax2.set_ylabel('0th-order transmission', color='0.4')
        ax2.set_ylim(0, 1.05)
        ax1.legend(loc='upper left', fontsize=8)
        fig.suptitle(f"{cfg['run_id']}  (diagnostic sweep, order "
                     f"{cfg['fourier_order']})", fontsize=9)
        fig.tight_layout()
        fig.savefig(cfg['_dir'] / 'spectra.png', dpi=130, bbox_inches='tight')
        plt.close(fig)

    out = RESULTS_ROOT / f'spectra_peaks_{stage}.csv'
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {out} ({len(rows)} candidates)")
    return rows


# ---------------------------------------------------------------------------

def final_table(stage='refine'):
    peaks = {r['run_id']: r for r in spectra_report(stage)}
    rows = []
    for cfg in load_runs(stage):
        if cfg.get('status') != 'completed':
            continue
        fs = cfg.get('final_scores', {})
        proj, binr = fs.get('projected', {}), fs.get('binary', {})
        pk = peaks.get(cfg['run_id'], {})
        rows.append({
            'run_id': cfg['run_id'],
            'period_nm': cfg['period_nm'], 'height_nm': cfg['height_nm'],
            'seed': cfg['seed'],
            'Fco_projected': proj.get('F_co', float('nan')),
            'Fco_binary': binr.get('F_co', float('nan')),
            'S_ED': proj.get('S_ED', float('nan')),
            'S_MD': proj.get('S_MD', float('nan')),
            'balance': proj.get('balance', float('nan')),
            'lambda_peak_ED': pk.get('lambda_peak_ED', float('nan')),
            'lambda_peak_MD': pk.get('lambda_peak_MD', float('nan')),
            'ED_MD_peak_splitting_nm': pk.get('ED_MD_peak_splitting_nm', float('nan')),
        })
    rows.sort(key=lambda r: -(r['Fco_projected'] if np.isfinite(r['Fco_projected']) else -1e9))
    out = RESULTS_ROOT / f'final_summary_{stage}.csv'
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {out}")
        hdr = ('run_id', 'Fco_projected', 'Fco_binary', 'S_ED', 'S_MD',
               'balance', 'lambda_peak_ED', 'lambda_peak_MD')
        print(('%-42s' + '%12s' * 7) % hdr)
        for r in rows:
            print('%-42s%+12.4f%+12.4f%12.4f%12.4f%12.3f%12.1f%12.1f' % (
                r['run_id'], r['Fco_projected'], r['Fco_binary'], r['S_ED'],
                r['S_MD'], r['balance'], r['lambda_peak_ED'],
                r['lambda_peak_MD']))
    return rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['summary', 'select', 'contact', 'spectra', 'final'])
    ap.add_argument('--stage', default='discovery',
                    choices=['discovery', 'refine'])
    ap.add_argument('--threshold', type=float, default=IOU_THRESHOLD)
    args = ap.parse_args()
    if args.mode == 'summary':
        build_summary(args.stage)
    elif args.mode == 'select':
        select_candidates(threshold=args.threshold)
    elif args.mode == 'contact':
        contact_sheet(stage=args.stage,
                      out_name=f'contact_sheet_{args.stage}.png')
    elif args.mode == 'spectra':
        spectra_report(args.stage)
    elif args.mode == 'final':
        final_table(args.stage)


if __name__ == '__main__':
    main()
