"""AR-audit figures: baseline comparison, both directions, Argand
mechanism diagrams (sections 11-12), and the single six-panel
'is it actually antireflective?' figure (section 16)."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stage_core as sc

R = sc.RESULTS
F = R / 'figures'
LAM0 = 1332.5
H_STAR = 227.2
R_FRESNEL = 0.18699 ** 2


def cxr(row, stem):
    return complex(row[stem + '_re'], row[stem + '_im'])


def argand_path(row):
    """Returns labeled cumulative complex-r points for one keypoint row."""
    rbg = cxr(row, 'rbg')
    rfull = cxr(row, 'rxx')
    Ed = cxr(row, 'E_dn')
    ev, oQ, om = cxr(row, 'even_px'), cxr(row, 'odd_Q'), cxr(row, 'odd_m')
    lam = float(row['lam_nm'])
    k = 2 * np.pi / (lam * 1e-9)
    P_nm = 550.0
    A = (P_nm * 1e-9) ** 2
    e2 = -(sc.Z0 / (2 * A)) * (-k ** 2 / 2) * cxr(row, 'I2')
    g_dn = Ed / (rfull - rbg)
    pieces = {'ED': ev / g_dn, 'EQ': -oQ / g_dn, 'MD': -om / g_dn,
              '2nd': e2 / g_dn}
    cum = [('bg', rbg)]
    z = rbg
    for name in ('ED', 'EQ', 'MD', '2nd'):
        z = z + pieces[name]
        cum.append((f'+{name}', z))
    cum.append(('full', rfull))
    return rbg, pieces, cum, rfull


def draw_argand(ax, row, title):
    rbg, pieces, cum, rfull = argand_path(row)
    z = 0
    z_pts = [c for _, c in cum[:-1]]
    labels = [n for n, _ in cum[:-1]]
    prev = 0 + 0j
    colors = ['#666', 'tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    for (name, z), c in zip(cum[:-1], colors):
        ax.annotate('', xy=(z.real, z.imag), xytext=(prev.real, prev.imag),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.8))
        mid = (prev + z) / 2
        ax.text(mid.real, mid.imag, name, fontsize=7.5, color=c)
        prev = z
    ax.annotate('', xy=(rfull.real, rfull.imag),
                xytext=(prev.real, prev.imag),
                arrowprops=dict(arrowstyle='->', color='k', lw=1.2,
                                linestyle=':'))
    ax.plot([rfull.real], [rfull.imag], 'k*', ms=11,
            label=f'full r, |r|={abs(rfull):.3f}')
    ax.plot([0], [0], 'ko', ms=5, mfc='none', label='r = 0')
    lim = max(0.35, abs(rbg) * 1.6, abs(rfull) * 1.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Re r'); ax.set_ylabel('Im r')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.2)


def main():
    df = pd.read_csv(R / 'ar_rt_all.csv')
    d9 = df[df.order == 9]
    kp = pd.read_csv(R / 'ar_keypoints.csv')
    bare = d9[d9.case == 'bare'].sort_values('lam_nm')

    def spec(case, h, col='f_R'):
        p = d9[(d9.case == case) & (np.isclose(d9.h_nm, h))] \
            .sort_values('lam_nm')
        return p.lam_nm.to_numpy(), p[col].to_numpy()

    # section 7: per-h comparison (R and T)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    for ax, h in zip(axs.ravel(), [225.0, H_STAR, 235.0, 250.0]):
        lam, Rp = spec('p0550', h)
        ax.semilogy(lam, Rp, 'k-', lw=2, label='P0550 freeform')
        ax.semilogy(bare.lam_nm, bare.f_R, '--', color='tab:blue', lw=1.5,
                    label='bare silica')
        lu, Ru = spec('uniform', h)
        ax.semilogy(lu, Ru, '-', color='tab:orange', lw=1.2,
                    label='uniform a-Si film')
        if h in (H_STAR, 235.0):
            ls, Rs = spec('simple', h)
            ax.semilogy(ls, Rs, '-', color='tab:green', lw=1.0,
                        label='fill-matched disk')
        ax.set_title(f'h = {h:g} nm', fontsize=10)
        ax.set_ylim(1e-4, 1.1)
        ax.grid(alpha=0.25)
        ax.axvline(LAM0, color='k', ls=':', lw=0.7)
    axs[0, 0].legend(fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel('wavelength (nm)')
    for ax in axs[:, 0]:
        ax.set_ylabel('R')
    fig.suptitle('reflection vs baselines (substrate-side incidence)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(F / 'ar_comparison_R.png', dpi=200)
    fig.savefig(F / 'ar_comparison_R.pdf')
    plt.close(fig)

    # section 12 figure: Argand pair h=221 vs h*=227.2 at lam0 + h* at Rmin
    rows = {r['tag']: r for _, r in kp.iterrows()}
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6))
    draw_argand(axs[0], rows['h221_lam0'],
                'h = 221 nm, 1332.5 nm\n(naive 180-deg ED-EQ crossing)')
    draw_argand(axs[1], rows[f'h{H_STAR:g}_lam0'],
                f'h = {H_STAR} nm, 1332.5 nm\n(R-minimum thickness)')
    tagm = f'h{H_STAR:g}_lamRmin'
    draw_argand(axs[2], rows[tagm],
                f"h = {H_STAR} nm, {rows[tagm]['lam_nm']:.0f} nm\n"
                '(deepest reflection minimum)')
    fig.suptitle('complex-r construction: background + multipole ladder '
                 '(per-row exact port coupling); star = full TORCWA',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(F / 'ar_argand_mechanism.png', dpi=200)
    fig.savefig(F / 'ar_argand_mechanism.pdf')
    plt.close(fig)

    # section 16: the single six-panel figure
    fig = plt.figure(figsize=(13, 12.5))
    gs = fig.add_gridspec(3, 2, hspace=0.34, wspace=0.22)
    # A topology
    axA = fig.add_subplot(gs[0, 0])
    rho = np.load(R / 'geometry' / 'p0550_rho_binary_exact.npy')
    axA.imshow(rho.T, origin='lower', cmap='gray_r', extent=[0, 550, 0, 550],
               interpolation='nearest')
    axA.set_title('A  P0550 hard-binary topology (a-Si dark, P = 550 nm, '
                  'fill 0.62)', fontsize=9)
    axA.set_xlabel('x (nm)'); axA.set_ylabel('y (nm)')
    # B baselines at h*
    axB = fig.add_subplot(gs[0, 1])
    lam, Rp = spec('p0550', H_STAR)
    axB.semilogy(lam, Rp, 'k-', lw=2.2, label=f'P0550, h = {H_STAR} nm')
    axB.semilogy(bare.lam_nm, bare.f_R, '--', color='tab:blue', lw=1.6,
                 label='bare silica (0.035)')
    lu, Ru = spec('uniform', H_STAR)
    axB.semilogy(lu, Ru, color='tab:orange', lw=1.2, label='uniform film')
    ls, Rs = spec('simple', H_STAR)
    axB.semilogy(ls, Rs, color='tab:green', lw=1.0, label='fill-matched disk')
    axB.set_ylim(1e-4, 1.1); axB.grid(alpha=0.25)
    axB.axvline(LAM0, color='k', ls=':', lw=0.7)
    axB.legend(fontsize=7)
    axB.set_title('B  is it antireflective? R vs all baselines', fontsize=9)
    axB.set_xlabel('wavelength (nm)'); axB.set_ylabel('R')
    # C composition at h* (interpolated + exact keypoints)
    axC = fig.add_subplot(gs[1, 0])
    sw = pd.read_csv(R / 'p0550_phase_sweep.csv')
    sw5 = sw[sw.lam_nm % 5 == 0]
    w = (H_STAR - 225.0) / 10.0
    lo = sw5[sw5.h_nm == 225.0].sort_values('lam_nm')
    hi = sw5[sw5.h_nm == 235.0].sort_values('lam_nm')
    lamC = lo.lam_nm.to_numpy()
    for c, col in (('f_ED', 'tab:blue'), ('f_EQ', 'tab:red'),
                   ('f_MD', 'tab:green'), ('f_MQ', 'tab:purple')):
        v = (1 - w) * lo[c].to_numpy() + w * hi[c].to_numpy()
        axC.plot(lamC, v, color=col, lw=1.6, label=c)
    for tag, mk in ((f'h{H_STAR:g}_lam0', 'o'), (f'h{H_STAR:g}_lamRmin', 's')):
        r = rows[tag]
        for c, col in (('f_ED', 'tab:blue'), ('f_EQ', 'tab:red'),
                       ('f_MD', 'tab:green'), ('f_MQ', 'tab:purple')):
            axC.plot([r['lam_nm']], [r[c]], mk, color=col, ms=6, mec='k',
                     mew=0.5)
    axC.set_ylim(0, 1); axC.grid(alpha=0.25); axC.legend(fontsize=7, ncol=4)
    axC.set_title('C  exact family fractions at h* (lines: interpolated '
                  'family; markers: exact solves)', fontsize=9)
    axC.set_xlabel('wavelength (nm)'); axC.set_ylabel('fraction')
    # D Argand at best point
    axD = fig.add_subplot(gs[1, 1])
    draw_argand(axD, rows[tagm],
                f'D  complex-r cancellation at {rows[tagm]["lam_nm"]:.0f} nm,'
                f' h = {H_STAR} nm')
    # E thickness family R
    axE = fig.add_subplot(gs[2, 0])
    for h, c in ((225.0, 'tab:blue'), (H_STAR, 'k'), (235.0, 'tab:orange'),
                 (250.0, 'tab:red')):
        lam, Rp = spec('p0550', h)
        axE.semilogy(lam, Rp, color=c, lw=1.9 if h == H_STAR else 1.1,
                     label=f'h = {h:g}')
    axE.axhline(R_FRESNEL, color='gray', ls='--', lw=1, label='bare')
    axE.set_ylim(1e-4, 1); axE.grid(alpha=0.25); axE.legend(fontsize=7)
    axE.axvline(LAM0, color='k', ls=':', lw=0.7)
    axE.set_title('E  thickness family', fontsize=9)
    axE.set_xlabel('wavelength (nm)'); axE.set_ylabel('R')
    # F both directions
    axF = fig.add_subplot(gs[2, 1])
    p = d9[(d9.case == 'p0550') & (np.isclose(d9.h_nm, H_STAR))] \
        .sort_values('lam_nm')
    axF.semilogy(p.lam_nm, p.f_R, 'k-', lw=1.8, label='silica-side incidence')
    axF.semilogy(p.lam_nm, p.b_R, color='tab:orange', ls='--', lw=1.4,
                 label='air-side incidence')
    dmax = float(np.max(np.abs(p.f_R - p.b_R)))
    axF.text(0.03, 0.05, f'max |R_f - R_b| = {dmax:.1e}\n'
             '(reciprocal quasi-lossless 2-port)',
             transform=axF.transAxes, fontsize=8)
    axF.axhline(R_FRESNEL, color='gray', ls='--', lw=1)
    axF.set_ylim(1e-4, 1); axF.grid(alpha=0.25); axF.legend(fontsize=7)
    axF.set_title('F  both illumination directions (h*)', fontsize=9)
    axF.set_xlabel('wavelength (nm)'); axF.set_ylabel('R')
    fig.suptitle('P0550 antireflection audit - single-figure answer',
                 fontsize=12)
    fig.savefig(F / 'ar_single_answer.png', dpi=200)
    fig.savefig(F / 'ar_single_answer.pdf')
    plt.close(fig)
    print('AR_FIGS_DONE')


if __name__ == '__main__':
    main()
