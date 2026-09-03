"""Partial-solution leaderboards (spec sec 7) + small corrected angular
set for promising candidates (spec sec 6) + the mining report.

usage: python pr_leaderboard.py
Reads results/perfect_r_workspace_candidates.csv; writes
results/workspace_partial_champions.csv,
results/mining_angular_set.csv, reports/PERFECT_R_WORKSPACE_MINING.md
"""
import json

import numpy as np
import pandas as pd
import torch

import pr_core as pr
import wf_core as wf
import rt_core as rc

RES = pr.HERE / 'results'
REP = pr.HERE / 'reports'
CAMPS = {'rtfree': pr.HERE.parent / 'rtype_freeform_campaign',
         'widefov': pr.HERE.parent / 'rtype_widefov_campaign'}

BOARDS = [('highest R_cross', 'Rcross', False),
          ('highest total reflection', 'Rtot', False),
          ('largest min(|r_x|,|r_y|)', 'min_r', False),
          ('smallest transmission', 'T', True),
          ('smallest R_co', 'co', True),
          ('smallest absorption', 'A', True),
          ('smallest phase error from pi', 'phase_err_deg', True),
          ('highest F_ideal', 'F', False)]


def load_rho(row):
    if row.campaign == 'paper':
        from wf_preflight import rect_rho
        return rect_rho()
    d = CAMPS[row.campaign] / row.stage / row.tag
    if (d / 'rho_binary.npy').exists():
        return torch.tensor(np.load(d / 'rho_binary.npy'))
    from pr_mine import recover_checkpoint
    return recover_checkpoint(d, row.P)[0]


def main():
    torch.set_num_threads(4)
    df = pd.read_csv(RES / 'perfect_r_workspace_candidates.csv')
    df['min_r'] = df[['abs_rx', 'abs_ry']].min(axis=1)
    # useful-R filters for the leaderboards that would otherwise be won
    # by trivial states (e.g. "smallest T" by an absorber)
    useful = df[df.Rcross >= 0.30]
    lines = ['# Perfect-R workspace mining', '',
             f'{len(df)} historical geometries re-evaluated at theta=0, '
             'order [9,9], corrected conventions (exact Jones, F_ideal).',
             f'{int((df.source != "final").sum())} recovered from '
             'checkpoint-only (incomplete) run directories.', '',
             '## Partial-solution leaderboards (top 3 each; '
             '"useful" boards require R_cross >= 0.30)', '']
    champs = []
    for title, col, asc in BOARDS:
        pool = df if col in ('Rcross', 'Rtot', 'F', 'min_r') else useful
        top = pool.sort_values(col, ascending=asc).head(3)
        lines.append(f'### {title}')
        for _, r in top.iterrows():
            lines.append(f'- {r.tag} ({r.campaign}/{r.stage}, {r.source}): '
                         f'{col}={r[col]:.3f} | F={r.F:.3f} Rc={r.Rcross:.3f} '
                         f'T={r["T"]:.3f} co={r.co:.3f} A={r.A:.3f} '
                         f'|rx|,|ry|={r.abs_rx:.2f},{r.abs_ry:.2f} '
                         f'err={r.phase_err_deg:.0f} isl={r.n_islands}')
            champs.append({'board': title, **r.to_dict()})
        lines.append('')
    pd.DataFrame(champs).to_csv(RES / 'workspace_partial_champions.csv',
                                index=False)
    # overlooked: incomplete checkpoints beating recorded finals
    ck = df[df.source != 'final'].sort_values('F', ascending=False).head(5)
    lines += ['## Checkpoint-only states (never recorded as finals)', '']
    for _, r in ck.iterrows():
        lines.append(f'- {r.tag} [{r.source}]: F={r.F:.3f} T={r["T"]:.3f} '
                     f'co={r.co:.3f} A={r.A:.3f}')
    # small corrected angular set for the promising set
    prom = pd.concat([df.sort_values('F', ascending=False).head(6),
                      useful.sort_values('T').head(3),
                      useful.sort_values('A').head(3)]).drop_duplicates('tag')
    rows = []
    for _, r in prom.iterrows():
        rho = load_rho(r)
        for th in (0.0, 20.0, 40.0, 50.0):
            for ph in (0.0, 45.0, 90.0):
                with torch.no_grad():
                    Rj, Tj = wf.jones_angle(rho, r.P, r.H, th, ph,
                                            order=(9, 9))
                m = pr.scalars(pr.port_metrics(Rj, Tj))
                rows.append({'tag': r.tag, 'theta': th, 'phi': ph, **m})
        sub = pd.DataFrame([x for x in rows if x['tag'] == r.tag])
        print(f'{r.tag}: F0={sub.F.iloc[0]:.3f} minF(0-50)={sub.F.min():.3f}'
              f' maxT={sub["T"].max():.3f}', flush=True)
    ang = pd.DataFrame(rows)
    ang.to_csv(RES / 'mining_angular_set.csv', index=False)
    lines += ['', '## Corrected small angular set (theta 0/20/40/50 x '
              'phi 0/45/90, exact p/s basis)', '',
              '| tag | F(0) | min F | mean F | max T | max co |',
              '|---|---|---|---|---|---|']
    for tag, sub in ang.groupby('tag'):
        lines.append(f'| {tag} | {sub.F.iloc[0]:.3f} | {sub.F.min():.3f} | '
                     f'{sub.F.mean():.3f} | {sub["T"].max():.3f} | '
                     f'{sub.co.max():.3f} |')
    (REP / 'PERFECT_R_WORKSPACE_MINING.md').write_text('\n'.join(lines))
    print('LEADERBOARD_DONE', flush=True)


if __name__ == '__main__':
    main()
