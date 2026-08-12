"""ITO permittivity handling.

Loads the supplied digitized permittivity CSV, builds interpolants, checks
passivity under the exp(-i*omega*t) convention (passive -> Im(eps) > 0), and
computes the material ENZ (zero-crossing) wavelength by interpolation.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

import config


class ITOMaterial:
    def __init__(self, csv_path=config.ITO_CSV):
        self.df = pd.read_csv(csv_path)
        self.columns = list(self.df.columns)

        # Column audit: the task description anticipates possibly-swapped
        # "as_labeled" columns plus "recommended_physical" columns.  Detect
        # what is actually present rather than assuming.
        rec_re = [c for c in self.columns if "epsilon_real" in c and "recommended" in c]
        rec_im = [c for c in self.columns if "epsilon_imag" in c and "recommended" in c]
        lab_re = [c for c in self.columns if "epsilon_real" in c and "recommended" not in c]
        lab_im = [c for c in self.columns if "epsilon_imag" in c and "recommended" not in c]
        if rec_re and rec_im:
            self.re_col, self.im_col = rec_re[0], rec_im[0]
            self.used_recommended = True
        elif lab_re and lab_im:
            self.re_col, self.im_col = lab_re[0], lab_im[0]
            self.used_recommended = False
        else:
            raise ValueError(f"Cannot identify permittivity columns in {self.columns}")
        self.labeled_cols_present = bool(lab_re and lab_im)

        self.wl = self.df["wavelength_nm"].to_numpy(float)
        self.eps_re = self.df[self.re_col].to_numpy(float)
        self.eps_im = self.df[self.im_col].to_numpy(float)

        self._sp_re = CubicSpline(self.wl, self.eps_re)
        self._sp_im = CubicSpline(self.wl, self.eps_im)

    # ------------------------------------------------------------------
    def eps(self, wl_nm):
        """Complex relative permittivity at wavelength(s) in nm (cubic spline)."""
        wl_nm = np.asarray(wl_nm, float)
        if np.any(wl_nm < self.wl[0]) or np.any(wl_nm > self.wl[-1]):
            raise ValueError("wavelength outside tabulated ITO range "
                             f"[{self.wl[0]}, {self.wl[-1]}] nm")
        return self._sp_re(wl_nm) + 1j * self._sp_im(wl_nm)

    # ------------------------------------------------------------------
    def passivity_report(self):
        """Under exp(-i*omega*t), a passive medium has Im(eps) > 0."""
        im_min = self.eps_im.min()
        monotonic_re = np.all(np.diff(self.eps_re) < 0)
        return {
            "Im_eps_min": im_min,
            "passive_everywhere": bool(im_min > 0),
            "Re_eps_monotonically_decreasing": bool(monotonic_re),
        }

    # ------------------------------------------------------------------
    def zero_crossing_nm(self):
        """Material ENZ wavelength: Re(eps)=0, from spline root bracketing.

        This is a MATERIAL property, not an electromagnetic eigenmode.
        """
        sign = np.sign(self.eps_re)
        idx = np.where(np.diff(sign) != 0)[0]
        if len(idx) != 1:
            raise RuntimeError(f"expected exactly one Re(eps)=0 crossing, found {len(idx)}")
        i = idx[0]
        return brentq(self._sp_re, self.wl[i], self.wl[i + 1], xtol=1e-9)


if __name__ == "__main__":
    m = ITOMaterial()
    print("columns present:", m.columns)
    print("used columns   :", m.re_col, "|", m.im_col,
          "(recommended-physical)" if m.used_recommended else "(as-labeled)")
    print("as-labeled columns also present:", m.labeled_cols_present)
    print("passivity      :", m.passivity_report())
    lam0 = m.zero_crossing_nm()
    print(f"material ENZ zero crossing: {lam0:.3f} nm")
    print(f"eps at crossing: {m.eps(lam0):.6f}")
