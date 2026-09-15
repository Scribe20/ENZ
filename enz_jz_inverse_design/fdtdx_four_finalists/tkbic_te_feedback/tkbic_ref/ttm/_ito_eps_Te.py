# -*- coding: utf-8 -*-
"""ITO eps(lambda, Te) — delta-Drude(Kane 비포물선) 전자온도 의존 유전율 (2026-09-04).

RCWA 캠페인(_rcwa_fast.make_eps_ito_Te), meep, _fit_meep_materials 와 **동일한 식**:

    E          = hc / lam_nm                       [eV], hc = 1239.841984 eV nm
    m_ratio(Te)= 0.35 * (1 + 2*0.4191*(1 + (pi^2/4)*(kB*Te)^2)),  kB = 8.617333262e-5 eV/K
    hwp(Te)    = HWP * sqrt(m_ratio(300)/m_ratio(Te))              (HWP = 1.775882 eV)
    eps(lam,Te)= eps_meas(lam) - (hwp(Te)^2 - HWP^2) / (E*(E + i*HGAM))   (HGAM = 0.126846 eV)

즉 측정 eps 에 "Drude 항의 변화분"만 얹는다 (delta-Drude). Te=300 K 에서는 정확히
eps_meas 로 환원된다 (아래 self-test 에서 |delta|<1e-10 확인). EPS_INF 는 식에 들어가지
않으며 참고용으로만 보관한다.

단위/규약
- lam_nm: 진공 파장 [nm]. eps 는 상대 유전율(복소, Im>0 = 손실, exp(-i w t) 규약 —
  Lumerical/RCWA 와 동일). n+ik = sqrt(eps) (주값, Im>=0).
- eps_meas = (n+ik)^2, n,k 는 Materials_data/ITO_nk.csv (6열 헤더 csv; 헤더 인식 로더
  `load_nk_table` 을 _lumerical_tkbic_build 에서 재사용).
- sampled_table_for_lumerical(): Lumerical "Sampled 3D data" 의 'sampled data' 2열
  [f_Hz, eps(complex)] — 주파수 오름차순 (기존 _lumerical_tkbic_build.eps_table 과 동일).

ASSUM
- hc = 1239.841984 eV nm (공통 컨텍스트 지정값). _rcwa_fast 계열 노트북은 1239.84193 을
  쓰며 상대차 4e-8 → eps 차이 ~1e-8 수준 (self-test 에 수치 기록).
- 상수 HWP/HGAM 는 tkbic_meep_matfits.json 의 global_drude 값 (파일이 있으면 그 값을
  읽고 없으면 아래 하드코딩 기본값 사용; 둘은 1e-6 이내 동일).
"""
import json
import sys
from pathlib import Path

import numpy as np

HC_EVNM = 1239.841984
KB_EV = 8.617333262e-5
M0_RATIO = 0.35
KANE_C = 0.4191
HWP_DEFAULT = 1.775882      # eV
HGAM_DEFAULT = 0.126846     # eV
EPS_INF_DEFAULT = 3.42674   # 참고용 (식에 미사용)

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent                      # package root (patched for share)
MATFITS_JSON = _PKG / "cache" / "tkbic_meep_matfits.json"
MATDIR = _PKG / "Materials_data"
ITO_NK_CSV = MATDIR / "ITO_nk.csv"


def _load_drude_consts():
    """global_drude 상수 (hwp, hgam, eps_inf) — json 우선, 없으면 기본값."""
    if MATFITS_JSON.exists():
        try:
            g = json.loads(MATFITS_JSON.read_text(encoding="utf-8"))["global_drude"]
            return float(g["hwp_eV"]), float(g["hgam_eV"]), float(g["eps_inf"])
        except Exception:
            pass
    return HWP_DEFAULT, HGAM_DEFAULT, EPS_INF_DEFAULT


HWP, HGAM, EPS_INF = _load_drude_consts()


def _load_nk_table(path):
    """Header-aware nk loader (same logic as the notebook load_nk_table): returns (lam_nm, n, k)."""
    path = Path(path)
    rows, header = [], None
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "%")):
            continue
        parts = line.replace(",", " ").split()
        try:
            values = [float(x) for x in parts]
        except ValueError:
            if any(ch.isalpha() for ch in line) and "," in line:
                header = [x.strip().lower() for x in line.split(",")]
            continue
        if header and "n" in header and "k" in header:
            li = header.index("wavelength_nm") if "wavelength_nm" in header else 0
            ni, ki = header.index("n"), header.index("k")
        else:
            li, ni, ki = 0, 1, 2
        if len(values) <= max(li, ni, ki):
            continue
        rows.append([values[li], values[ni], values[ki]])
    d = np.asarray(rows, float)
    o = np.argsort(d[:, 0])
    return d[o, 0], d[o, 1], d[o, 2]


_NK_CACHE = {}


def _nk(path=ITO_NK_CSV):
    key = str(path)
    if key not in _NK_CACHE:
        _NK_CACHE[key] = _load_nk_table(path)
    return _NK_CACHE[key]


def eps_meas_table(path=ITO_NK_CSV):
    """측정 ITO (lam_nm, eps_meas) — lam 오름차순. eps_meas=(n+ik)^2."""
    lam, n, k = _nk(path)
    return lam, (n + 1j * k) ** 2


def eps_meas(lam_nm, path=ITO_NK_CSV):
    """eps_meas 를 lam_nm 에 보간 (n,k 각각 선형보간 후 제곱 — RCWA _eps_np 와 동일)."""
    lam, n, k = _nk(path)
    lam_nm = np.asarray(lam_nm, float)
    return (np.interp(lam_nm, lam, n) + 1j * np.interp(lam_nm, lam, k)) ** 2


def m_ratio(Te):
    Te = np.asarray(Te, float)
    return M0_RATIO * (1.0 + 2.0 * KANE_C * (1.0 + (np.pi ** 2 / 4.0) * (KB_EV * Te) ** 2))


def hwp_of(Te):
    return HWP * np.sqrt(m_ratio(300.0) / m_ratio(Te))


def eps_ito_Te(lam_nm, Te, path=ITO_NK_CSV):
    """eps_ITO(lam_nm, Te) (복소 ndarray/스칼라). Te=300 -> eps_meas 정확히."""
    E = HC_EVNM / np.asarray(lam_nm, float)
    d = (hwp_of(float(Te)) ** 2 - HWP ** 2) / (E * (E + 1j * HGAM))
    return eps_meas(lam_nm, path) - d


def nk_of_eps(eps):
    """n+ik = sqrt(eps) 주값 (Im>=0 강제)."""
    nk = np.sqrt(np.asarray(eps, complex))
    return np.where(nk.imag < 0, -nk, nk)


def sampled_table_for_lumerical(Te, lo=900.0, hi=1700.0, path=ITO_NK_CSV):
    """Lumerical 'sampled data' 2열 [f_Hz, eps] (주파수 오름차순), lam in [lo,hi] nm.

    측정 파장 격자를 그대로 쓴다 (기존 _lumerical_tkbic_build.eps_table 과 동일 규약)."""
    C0, NM = 299792458.0, 1e-9
    lam, _ = eps_meas_table(path)
    m = (lam >= lo) & (lam <= hi)
    lam_s = lam[m]
    eps = eps_ito_Te(lam_s, Te, path)
    freq = C0 / (lam_s * NM)
    o = np.argsort(freq)
    return np.column_stack([freq[o], eps[o]])


def table_to_lam_nk(table):
    """[f_Hz, eps] 테이블 -> (lam_nm 오름차순, n, k) — check_fit 목표용."""
    C0, NM = 299792458.0, 1e-9
    table = np.asarray(table)
    lam = C0 / (table[:, 0].real * NM)
    nk = nk_of_eps(table[:, 1])
    o = np.argsort(lam)
    return lam[o], nk.real[o], nk.imag[o]


def _selftest(verbose=True):
    out = {}
    lam_chk = np.array([1250.0, 1292.0, 1300.0, 1350.0])
    d300 = np.max(np.abs(eps_ito_Te(lam_chk, 300.0) - eps_meas(lam_chk)))
    assert d300 < 1e-10, d300
    out["max|eps(300K)-eps_meas|"] = float(d300)
    e0 = complex(eps_ito_Te(1292.0, 300.0))
    assert abs(e0 - (0.0528 + 0.4187j)) < 0.02, e0
    # ---- _rcwa_fast.make_eps_ito_Te 와 대조 ----------------------------------
    lam_m, n_m, k_m = _nk()

    def _eps_np(l, n_i, k_i):
        return (np.interp(l, lam_m, n_i) + 1j * np.interp(l, lam_m, k_i)) ** 2

    try:
        if str(_HERE) not in sys.path:
            sys.path.append(str(_HERE))
        import _rcwa_fast  # noqa: F401  (torch 의존이면 여기서 실패)
        ns = dict(np=np, HWP_FIT=HWP, HGAM_FIT=HGAM, hc_eVnm=1239.84193,
                  _eps_np=_eps_np, ito_n_i=n_m, ito_k_i=k_m)
        ref_fn = _rcwa_fast.make_eps_ito_Te(ns)
        ref_mode = "_rcwa_fast.make_eps_ito_Te import (ns 주입, hc=1239.84193)"
    except Exception as ex:  # torch 등 문제 -> 식 전사
        ref_mode = f"식 전사 (import 실패: {type(ex).__name__}: {str(ex)[:60]})"
        kB = 8.617333262e-5

        def _mr(Te):
            return 0.35 * (1 + 2 * 0.4191 * (1.0 + (np.pi ** 2 / 4) * (kB * Te) ** 2))

        def ref_fn(l, Te):
            E = 1239.84193 / np.asarray(l, float)
            hw = HWP * np.sqrt(_mr(300.0) / _mr(Te))
            return _eps_np(l, n_m, k_m) - (hw ** 2 - HWP ** 2) / (E * (E + 1j * HGAM))
    pts = [(1292.0, 3000.0), (1250.0, 1500.0), (1350.0, 6000.0)]
    dmax = 0.0
    rows = []
    for l, Te in pts:
        a, b = complex(eps_ito_Te(l, Te)), complex(ref_fn(l, Te))
        dmax = max(dmax, abs(a - b))
        rows.append((l, Te, a, b))
    out["ref_mode"] = ref_mode
    out["max|eps-eps_ref| (3 pts)"] = float(dmax)
    # 물리 경향: 1292 nm 에서 Te 상승 -> Re eps 양으로 이동, eps''/|eps|^2 감소
    trend = []
    for Te in [300, 1500, 3000, 6000, 8000]:
        e = complex(eps_ito_Te(1292.0, Te))
        trend.append((Te, e, e.imag / abs(e) ** 2))
    assert trend[-1][1].real > trend[0][1].real
    assert trend[-1][2] < trend[0][2]
    if verbose:
        print(f"[selftest] HWP={HWP:.6f} eV HGAM={HGAM:.6f} eV (EPS_INF={EPS_INF:.5f}, 미사용)")
        print(f"[selftest] max|eps(300K)-eps_meas| = {d300:.2e}  (<1e-10 OK)")
        print(f"[selftest] 기준 대조: {ref_mode}")
        for l, Te, a, b in rows:
            print(f"    lam={l:.0f} Te={Te:5.0f}: eps={a.real:+.5f}{a.imag:+.5f}i  "
                  f"ref={b.real:+.5f}{b.imag:+.5f}i  |d|={abs(a-b):.2e}")
        print("[selftest] 1292 nm 경향 (Te, eps, eps''/|eps|^2):")
        for Te, e, q in trend:
            print(f"    {Te:5d} K: {e.real:+.4f}{e.imag:+.4f}i   {q:.3f}")
        E1, E2 = HC_EVNM / 1292.0, 1239.84193 / 1292.0
        hw = hwp_of(3000.0)
        dd = abs((hw ** 2 - HWP ** 2) / (E1 * (E1 + 1j * HGAM))
                 - (hw ** 2 - HWP ** 2) / (E2 * (E2 + 1j * HGAM)))
        print(f"[selftest] hc 1239.841984 vs 1239.84193 -> |d eps|(1292 nm, 3000 K) = {dd:.2e}")
    return out


if __name__ == "__main__":
    _selftest()
