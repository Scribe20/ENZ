"""
method_validation.py — Gates 2-4 numerical validation (run before pilot)
========================================================================
A. torch exact moments  ==  corrected MENP port (machine precision,
   identical dense fields).
B. Exact channel identity (freestanding layer, air background):
   E_sc^{±} = -(Z0/2A) ∫ J_x e^{∓ikz} dV  vs  TORCWA (t-1, r).
C. Multipole truncation quality: px / (my & Qxz) terms vs the exact
   channel integral; even/odd parity split.
D. Objective gradient smoke test (F_ED_EQ differentiable wrt rho).
Results appended to METHOD_VALIDATION.md.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))

import ed_eq_core as core                      # noqa: E402
from menp_port import exact_me                 # noqa: E402

EPS0, C0 = core.EPS0, core.C0
Z0 = 376.730313668

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def test_structure(seed=5, n=48):
    torch.manual_seed(seed)
    rho = torch.rand(n, n)
    rho_f = torch.fft.fft2(rho)
    kx = torch.fft.fftfreq(n)
    W = torch.exp(-((kx[:, None] ** 2 + kx[None, :] ** 2) / (2 * 0.06 ** 2)))
    rho = torch.real(torch.fft.ifft2(rho_f * W))
    rho = (rho > rho.median()).float()
    return rho


def main():
    P, h, lam = 700.0, 250.0, 1332.5
    order = [9, 9]
    eps_si = core.si_eps(lam)
    rho = test_structure()
    log(f'validation structure: P={P} h={h} lam={lam} order={order} '
        f'eps_si={eps_si:.4f} fill={rho.mean():.3f}')

    # ---- A: torch moments vs MENP port on identical fields -------------
    sim = core.build_sim(rho, P, h, lam, order)
    x_ax, z_ax, E, _ = core.fields_3d(sim, P, h, 64, 15)
    nmask = rho.shape[0]
    idx = (torch.floor(x_ax / P * nmask).long()) % nmask
    rho_s = rho[idx][:, idx]
    eps3 = (rho_s * (eps_si - 1.0) + 1.0)[:, :, None].expand(64, 64, 15)
    mo = core.torch_moments(E, eps3, x_ax, z_ax, lam)

    xm = (x_ax.numpy() - P / 2) * 1e-9
    zm = (z_ax.numpy() - h / 2) * 1e-9
    f = np.array([C0 / (lam * 1e-9)])
    E4 = [E[c].detach().numpy().astype(np.complex128)[..., None] for c in range(3)]
    n4 = np.sqrt(eps3.detach().numpy().astype(np.complex128))[..., None]
    ex = exact_me(xm, xm, zm, f, *E4, n4, n4, n4, mode='corrected')
    pairs = [('px', ex['p'][0, 0]), ('py', ex['p'][1, 0]), ('pz', ex['p'][2, 0]),
             ('mx', ex['m'][0, 0]), ('my', ex['m'][1, 0]), ('mz', ex['m'][2, 0]),
             ('Qxx', ex['Qe']['xx'][0]), ('Qxz', ex['Qe']['xz'][0]),
             ('Qyz', ex['Qe']['yz'][0]), ('Qxy', ex['Qe']['xy'][0])]
    worst = 0.0
    for tag, ref in pairs:
        v = complex(mo[tag])
        rel = abs(v - ref) / (abs(ref) + 1e-300)
        worst = max(worst, rel)
    log(f'A. torch-vs-MENP worst relative moment difference: {worst:.3e} '
        f'({"PASS" if worst < 1e-6 else "FAIL"}) [float32 fields, float64 MENP]')

    # ---- B/C: freestanding exact channel identity ----------------------
    def channel_case(P_, h_, lam_, rho_, tagpfx):
        simf = core.build_sim(rho_, P_, h_, lam_, order, substrate_eps=1.0)
        x_ax, z_ax, E, _ = core.fields_3d(simf, P_, h_, 96, 21)
        nm = rho_.shape[0]
        i96 = (torch.floor(x_ax / P_ * nm).long()) % nm
        eps3 = (rho_[i96][:, i96] * (eps_si - 1.0) + 1.0)[:, :, None].expand(96, 96, 21)
        k = 2 * math.pi / (lam_ * 1e-9)
        omega = 2 * math.pi * C0 / (lam_ * 1e-9)
        chi = eps3 - 1.0
        Jx = (-1j * omega * EPS0) * chi * E[0].to(torch.complex128)
        xm = ((x_ax - P_ / 2) * 1e-9).to(torch.float64)
        zm = ((z_ax - h_ / 2) * 1e-9).to(torch.float64)
        A_cell = (P_ * 1e-9) ** 2
        Zc = zm.reshape(1, 1, -1)

        def tz(F):
            F = torch.trapezoid(F, xm, dim=0)
            F = torch.trapezoid(F, xm, dim=0)
            return torch.trapezoid(F, zm, dim=0)

        E_up = complex(-(Z0 / (2 * A_cell)) * tz(Jx * torch.exp(-1j * k * Zc)))
        E_dn = complex(-(Z0 / (2 * A_cell)) * tz(Jx * torch.exp(+1j * k * Zc)))
        amps = core.channel_amplitudes(simf)
        t_bare = complex(np.exp(1j * k * h_ * 1e-9))
        best = {}
        for chan, rec, raw, sub in [('up', E_up, complex(amps['txx']), t_bare),
                                    ('dn', E_dn, complex(amps['rxx']), 0.0)]:
            errs = {}
            for mult in (0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0):
                ph = np.exp(1j * k * mult * h_ * 1e-9)
                errs[mult] = abs(rec - (raw - sub) * ph) / (abs(rec) + 1e-300)
            mbest = min(errs, key=errs.get)
            best[chan] = (mbest, errs[mbest])
            log(f'B. {tagpfx} {chan}: best ref phase = e^(i k {mbest} h), '
                f'residual {errs[mbest]:.3f} (next-best {sorted(errs.values())[1]:.3f})')
        mo96 = core.torch_moments(E, eps3, x_ax * 1.0, z_ax * 1.0, lam_)
        px_ = complex(mo96['px']); my_ = complex(mo96['my']); Qxz_ = complex(mo96['Qxz'])
        even = -(Z0 / (2 * A_cell)) * (-1j * omega * px_)
        odd = -(Z0 / (2 * A_cell)) * (-1j * k) * (-(1j * omega / 6) * Qxz_ + my_)
        evx, odx = (E_up + E_dn) / 2, (E_up - E_dn) / 2
        log(f'C. {tagpfx} parity: even rel {abs(even-evx)/abs(evx):.3f} '
            f'(kh/2={k*h_*1e-9/2:.2f}); odd rel {abs(odd-odx)/abs(odx):.3f}')
        return best

    b1 = channel_case(P, h, lam, rho, 'cfg1(h250)')
    b2 = channel_case(660.0, 140.0, 1290.0, test_structure(seed=9), 'cfg2(h140)')
    consistent = all(b1[c][0] == b2[c][0] for c in ('up', 'dn'))
    log(f'B. reference-plane convention consistent across configs: {consistent}')

    # ---- D: gradient smoke test ----------------------------------------
    rho_g = test_structure(seed=7, n=40).requires_grad_(True)
    F, S_px, S_Q, _ = core.eval_objective(rho_g, 650.0, 200.0, lam, [5, 5],
                                          n_xy=40, nz=5)
    F.backward()
    g = rho_g.grad
    ok = (g is not None and torch.all(torch.isfinite(g))
          and float(torch.linalg.norm(g)) > 0 and float(S_px) > 0 and float(S_Q) > 0)
    log(f'D. gradient smoke: F={float(F):+.4f} S_px={float(S_px):.3e} '
        f'S_Qxz={float(S_Q):.3e} |grad|={float(torch.linalg.norm(g)):.3e} '
        f'({"PASS" if ok else "FAIL"})')

    with open(_HERE / 'METHOD_VALIDATION.md', 'a') as fo:
        fo.write('\n## method_validation.py run\n```\n' + '\n'.join(LOG) + '\n```\n')


if __name__ == '__main__':
    main()
