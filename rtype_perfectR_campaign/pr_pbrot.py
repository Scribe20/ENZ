"""Physical-rotation PB law vs incident angle for perfect-R finalists
(spec secs 24, 31): rotate the hard-binary motif alpha = 0..180 (15-deg
steps) at theta = 0/30/45/50 (phi = 0), fit phase_cross = phase0 +
s*alpha (ideal |s| = 2), and record the rotated-operator fidelity
F(U_alpha) at every state.
usage: python pr_pbrot.py <rho.npy> <P> <H> <label>
Appends results/pb_matrix_fidelity.csv
"""
import csv, math, sys
import numpy as np, scipy.ndimage as ndi, torch
import pr_core as pr, rt_core as rc, wf_core as wf

def main(path, P, H, label):
    rho = torch.tensor(np.load(path))
    rows = []
    for th in (0.0, 30.0, 45.0, 50.0):
        phs, amps, fids = [], [], []
        for al in range(0, 181, 15):
            rr = ndi.rotate(rho.numpy(), al, reshape=False, order=0, mode='constant', cval=0.0)
            rr = torch.tensor((rr > 0.5).astype(np.float32))
            with torch.no_grad():
                Rj, Tj = wf.jones_angle(rr, P, H, th, 0.0, order=(9, 9))
            Rc = rc.circular(Rj)
            phs.append(math.degrees(float(torch.angle(Rc[0, 1]))))
            amps.append(0.5 * float(torch.abs(Rc[0, 1]) ** 2 + torch.abs(Rc[1, 0]) ** 2))
            fids.append(float(pr.fidelity_state(Rj, float(al), 0.0)))
            rows.append({'label': label, 'theta': th, 'alpha': al, 'phase_deg': phs[-1],
                         'R_cross': amps[-1], 'F_Ualpha': fids[-1], 'T': float(0.5 * (torch.abs(Tj) ** 2).sum())})
        a = np.arange(0, 181, 15, dtype=float)
        ph = np.degrees(np.unwrap(np.radians(phs)))
        cf = np.polyfit(a, ph, 1); rms = float(np.std(ph - np.polyval(cf, a)))
        rows.append({'label': label, 'theta': th, 'alpha': -1, 'phase_deg': cf[0], 'R_cross': rms,
                     'F_Ualpha': float(np.min(fids)), 'T': float(np.mean(fids))})
        print(f'{label} th={th:.0f}: slope={cf[0]:+.3f} rms={rms:.1f} F(U_alpha) min/mean={min(fids):.3f}/{np.mean(fids):.3f} Rc range {min(amps):.3f}-{max(amps):.3f}', flush=True)
    out = pr.HERE / 'results' / 'pb_matrix_fidelity.csv'
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)
    print('PBROT_DONE', label, flush=True)

if __name__ == '__main__':
    torch.set_num_threads(1)
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
