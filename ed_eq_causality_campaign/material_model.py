"""
material_model.py — dispersive NIR material models for the ED-EQ campaign
=========================================================================

PROVENANCE (all fetched from the refractiveindex.info database GitHub
mirror, polyanskiy/refractiveindex.info-database @ master, path
database/data/main/Si/nk/; files stored in ./material/; CC0 license):

  primary  'aSi_Franta2013'   Franta, Necas, Ohlidal, Appl. Opt. 52, 7962
                              (2013) via refractiveindex.info; amorphous
                              silicon, universal-dispersion-model fit to
                              wide-range ellipsometry; tabulated nk
                              0.138-26.9 um.  n(1332.5 nm) = 3.6518,
                              k = 6.0e-6.
  bracket  'aSi_Pierce1972'   Pierce & Spicer, PRB 5, 3017 (1972) /
                              Palik handbook tabulation; evaporated a-Si
                              60-nm film; 0.0103-2.07 um.
                              n(1332.5 nm) = 3.513, k = 0 (below gap).
  bracket  'aSi_Karaman2025'  Karaman et al. 2025 (refractiveindex.info
                              "Amorphous silicon" shelf); 0.21-2.5 um,
                              20 C.  n(1332.5 nm) = 3.814.
  control  'cSi_Franta2017'   Franta et al. 2017, 25 C; 0.031-310 um.
                              n(1332.5 nm) = 3.5021 (matches c-Si
                              literature - used as a parsing sanity check
                              and optional crystalline-Si control).

MATERIAL-IDENTITY STATEMENT (recorded per campaign contract section 3):
The project's baseline example used a hydrogenated a-Si:H film model
(Materials.aSiH, tabulated only to 999 nm - the source of the previous
campaign's clamping problem).  NO authoritative tabulated NIR a-Si:H
dataset exists locally or in the refractiveindex.info main shelf.  The
scientifically defensible choice is unhydrogenated amorphous silicon with
an explicit uncertainty band:

    n(1332.5 nm) = 3.65 (primary, Franta 2013)
    bracket [3.51 (Pierce) ... 3.81 (Karaman)]
    - PECVD a-Si:H literature values (n ~ 3.4-3.6 at 1.3-1.5 um,
      H-content dependent) fall at/below the low edge of this band.
    k(1332.5 nm): sub-gap; primary tabulation 6e-6.  Real a-Si:H films
    show sub-gap absorption up to alpha ~ 1-100 /cm (k up to ~1e-3);
    the campaign therefore carries a separate "realistic-loss" scenario
    k_loss = 1e-4 and a lossless diagnostic k -> 0 (contract section 15).

INTERPOLATION RULE: cubic spline in wavelength on the tabulated (lam, n)
and (lam, k) columns, evaluated strictly INSIDE the tabulated range.
Out-of-range evaluation raises ValueError - clamping is forbidden by the
campaign contract (the previous campaign's clamped-endpoint mistake).
"""

from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

_MAT_DIR = Path(__file__).resolve().parent / 'material'

_FILES = {
    'aSi_Franta2013': 'Franta.yml',
    'aSi_Pierce1972': 'Pierce.yml',
    'aSi_Karaman2025': 'Karaman-20C.yml',
    'cSi_Franta2017': 'Franta-25C.yml',
}

PRIMARY = 'aSi_Franta2013'
K_LOSS_SCENARIO = 1e-4          # documented realistic sub-gap loss scenario


class Material:
    def __init__(self, key):
        self.key = key
        d = yaml.safe_load(open(_MAT_DIR / _FILES[key]))
        tab = next(b for b in d['DATA'] if 'tabulated' in b['type'])
        arr = np.array([[float(v) for v in ln.split()]
                        for ln in tab['data'].strip().splitlines()])
        self.lam_um = arr[:, 0]
        self._n = CubicSpline(arr[:, 0], arr[:, 1])
        kcol = arr[:, 2] if arr.shape[1] > 2 else np.zeros(len(arr))
        self._k = CubicSpline(arr[:, 0], kcol)
        self.lam_min_nm = self.lam_um.min() * 1000
        self.lam_max_nm = self.lam_um.max() * 1000
        self.reference = d.get('REFERENCES', '').splitlines()[0:2]

    def nk(self, lam_nm, k_override=None, lossless=False):
        """Complex refractive index at lam_nm. Raises outside the
        tabulated range (clamping forbidden)."""
        lam_um = np.asarray(lam_nm, float) / 1000.0
        if np.any(lam_um < self.lam_um.min()) or np.any(lam_um > self.lam_um.max()):
            raise ValueError(
                f'{self.key}: {lam_nm} nm outside tabulated range '
                f'[{self.lam_min_nm:.0f}, {self.lam_max_nm:.0f}] nm - '
                f'refusing to extrapolate/clamp')
        n = self._n(lam_um)
        k = 0.0 if lossless else (k_override if k_override is not None
                                  else np.maximum(self._k(lam_um), 0.0))
        return n + 1j * k

    def eps(self, lam_nm, **kw):
        return self.nk(lam_nm, **kw) ** 2


def primary():
    return Material(PRIMARY)


if __name__ == '__main__':
    for key in _FILES:
        m = Material(key)
        nk = m.nk(1332.5)
        print(f'{key:16s} range [{m.lam_min_nm:7.1f},{m.lam_max_nm:9.1f}] nm  '
              f'n+ik(1332.5) = {nk:.5f}  eps = {nk**2:.4f}')
    m = primary()
    print('primary lossless :', m.nk(1332.5, lossless=True))
    print('primary k-loss   :', m.nk(1332.5, k_override=K_LOSS_SCENARIO))
    try:
        m.nk(3000.0 if m.lam_max_nm < 3000 else 1e6)
    except ValueError as e:
        print('out-of-range guard OK:', str(e)[:80])
