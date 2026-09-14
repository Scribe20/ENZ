"""Figure utilities: geometry, band diagrams along G-X-M-G with the official grid extrema, field profiles."""
import numpy as np
from pwem import *
from topopt import field_on_grid
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import linalg as sla

def plot_geometry(eps, ax=None, title='Permittivity'):
    if ax is None: fig, ax = plt.subplots(figsize=(4, 4))
    ext = [0, A_NM, 0, A_NM]
    ax.imshow(eps, origin='lower', cmap='gray_r', vmin=1, vmax=12.5, extent=ext, interpolation='nearest')
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)'); ax.set_title(title)
    return ax

def band_diagram(eps, ax=None, npts=30, nbands=10, Mmax=9, gap=None, title=None, formulation='official', ylim=(0, 0.8)):
    K, s, ticks = kpath_GXMG(npts)
    btm, bte = compute_bands(eps, K, Mmax=Mmax, nbands=nbands, formulation=formulation)
    if ax is None: fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(s, bte, color='tab:red', lw=1.3)
    ax.plot(s, btm, color='tab:blue', lw=1.3)
    ax.axhline(TARGET, color='k', ls='--', lw=0.9)
    if gap:
        ax.axhspan(gap['w_low'], gap['w_high'], color='purple', alpha=0.25, lw=0)
    for t in ticks[1:-1]: ax.axvline(t, color='gray', lw=0.6)
    ax.set_xticks(ticks); ax.set_xticklabels(['Γ', 'X', 'M', 'Γ']); ax.set_xlim(s[0], s[-1]); ax.set_ylim(*ylim)
    ax.set_ylabel('ωa/2πc'); ax.set_xlabel('Bloch wave vector')
    if title: ax.set_title(title)
    return ax, (s, btm, bte, ticks)

def mode_field(eps, k, pol, n, Mmax=9):
    """return (omega, field on grid) : Ez for TM (complex), Hz for TE, plus |D| or |E| intensities as needed.
    Fields are the Bloch-periodic parts u(r) times the phase exp(i k.r)."""
    from pwem import _matrices
    st = Structure(eps, Mmax); st.eps_inv_mat
    A_tm, A_te, kG, mag = _matrices(st, k, 'official')
    A = A_tm if pol == 'tm' else A_te
    lam, V = sla.eigh(A, subset_by_index=[0, n], driver='evr')
    v = V[:, n]; w = np.sqrt(max(lam[n], 0)) / (2 * np.pi)
    X, Y = meshgrid_xy(st.N)
    phase = np.exp(1j * (k[0] * X + k[1] * Y))
    if pol == 'tm':
        c = st.eps_inv_mat @ (mag * v)            # Ez coefficients (up to a constant): c = E^{-1} D^{1/2} v ~ lambda * c_Ez
        Ez = field_on_grid(c, st.basis, st.N) * phase
        return w, dict(Ez=Ez, energy=np.real(st.eps * np.abs(Ez)**2))
    else:
        Hz = field_on_grid(v, st.basis, st.N) * phase
        Dt = field_on_grid(kG * v[:, None], st.basis, st.N)  # ~ (k+G) v_G -> proportional to D_t (rotated); |D|^2
        D2 = np.abs(Dt[0])**2 + np.abs(Dt[1])**2
        return w, dict(Hz=Hz, D2=D2, energy=D2 / st.eps)   # electric energy ~ |D|^2/eps
