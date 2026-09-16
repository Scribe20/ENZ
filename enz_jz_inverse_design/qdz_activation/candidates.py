"""Registry of the FROZEN geometries evaluated by the activation extension (nothing is re-optimized).

Sources (all read-only):
  * qdz_parent/outputs/certified/<tag>/certification.json   exact-pole certified parent candidates (10)
  * qdz_parent/frontier/<tag>/result.json                   frontier runs WITHOUT a certification (8):
                                                            evaluated with a [7,7] exact-pole pass here,
                                                            labelled certified=False
  * qdz_parent/baselines/<name>/certification.json          the Jz/Fz finalists under the parent metrics
  * reference geometries: the SNU deck cylinder (SOURCE_AUDIT.md section 8; d = 405.7, h = 970.5,
    P = 588 nm) rasterised exactly as in nonlinear_activation_best/build_lookup.py, and the bare
    air / ITO / glass film (rho = None), both as controls that were never optimized in any campaign.
Every candidate carries: tag, kind, rho path, P, h, pad, Q_target, the certification dict (or None),
the SHA256 of the binary, and the certified Q_r / lambda_r / FWHM / eta_Dz / eta_Dz_P when present.
"""
import numpy as np

import common as cm                       # noqa: F401
import config                             # noqa: E402

NX = config.NX


def disc(nx, P, d):
    """Same rasterisation as nonlinear_activation_best/build_lookup.py and qdz_parent/baselines.py."""
    xg = (np.arange(nx) + 0.5) / nx * P
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    return (((X - P / 2) ** 2 + (Y - P / 2) ** 2) < (d / 2) ** 2).astype(float)


def load_rho(path):
    a = np.load(path)
    assert a.shape == (NX, NX), (path, a.shape)
    a = (np.asarray(a, dtype=float) > 0.5).astype(float)          # uint8 or float64 hard-binary
    return a


def _cert_summary(c):
    if not c:
        return {}
    mE = c.get("metrics", {}).get("lambda_E", {}).get("by_order", [{}])[-1]
    mR = c.get("metrics", {}).get("lambda_r", {}).get("by_order", [{}])[-1]
    near = c.get("pole_nearest_lambda_E") or {}
    qp = c.get("q_proxy_at_certification", {})
    return dict(Q_r=c.get("Q_r"), lambda_r=c.get("lambda_r"), fwhm_r_nm=near.get("fwhm_nm"),
                detuning_nm=c.get("detuning_nm"), detuning_in_linewidths=c.get("detuning_in_linewidths"),
                pole_refine_converged=(c.get("local_refine") or {}).get("converged"),
                eta_Dz_lamE=mE.get("eta_Dz"), eta_Dz_P_lamE=mE.get("eta_Dz_P"), U_mid_lamE=mE.get("U_mid"),
                R_parent_lamE=mE.get("R"), T_parent_lamE=mE.get("T"),
                eta_Dz_lamr=mR.get("eta_Dz"), R_parent_lamr=mR.get("R"), T_parent_lamr=mR.get("T"),
                eta_order_conv_rel=c.get("metrics", {}).get("lambda_E", {}).get("order_convergence_rel"),
                parent_poles=[dict(lam=p["lambda_nm"], Q=p["Q"], fwhm=p["fwhm_nm"]) for p in c.get("parent_poles", [])],
                cert_order=(c.get("orders") or [[None]])[-1],
                Q_proxy_well_posed=qp.get("well_posed"), Q_proxy=qp.get("Q_proxy"),
                Q_proxy_rel_err=qp.get("rel_err_vs_pole"),
                multipoles_lamE={k: v for k, v in (c.get("multipoles", {}).get("lambda_E", {}) or {}).items() if k.startswith("frac")},
                sha256_cert=c.get("sha256"))


def registry(include=("certified", "uncertified", "baselines", "references")):
    cands = []
    seen = set()
    if "certified" in include:
        for p in sorted((cm.QDZ / "outputs" / "certified").glob("*/certification.json")):
            c = cm.jload(p)
            tag = c["tag"]
            run = next((d for d in (cm.QDZ / "frontier" / tag, cm.QDZ / "pilot" / tag) if (d / "rho_hard_binary.npy").exists()), None)
            if run is None:
                continue
            r = cm.jload(run / "result.json")
            cands.append(dict(tag=tag, kind="qdz_certified", rho_path=str(run / "rho_hard_binary.npy"), P=float(c["P"]),
                              h=float(c["h"]), pad=float(r["pad_frac"]), Q_target=float(r["Q_target"]), certified=True,
                              cert_path=str(p), cert=_cert_summary(c), origin=f"qdz_parent/{run.relative_to(cm.QDZ)}"))
            seen.add(tag)
    if "uncertified" in include:
        for p in sorted((cm.QDZ / "frontier").glob("*/result.json")):
            r = cm.jload(p)
            if r.get("campaign") != "qdz_parent" or r["tag"] in seen:
                continue
            cands.append(dict(tag=r["tag"], kind="qdz_uncertified", rho_path=str(p.parent / "rho_hard_binary.npy"),
                              P=float(r["P"]), h=float(r["h"]), pad=float(r["pad_frac"]), Q_target=float(r["Q_target"]),
                              certified=False, cert_path=None, cert={},
                              proxy_end_of_run=dict(Q_proxy=r["hard"].get("Q_proxy"), lambda_r_proxy=r["hard"].get("lambda_r_proxy"),
                                                    curvature_u=r["hard"].get("curvature_u"), well_posed=bool(r["hard"].get("well_posed")),
                                                    acquired=bool(max(r["history"]["curvature_u"]) > 0.1)),
                              origin=f"qdz_parent/frontier/{r['tag']}"))
    if "baselines" in include:
        for p in sorted((cm.QDZ / "baselines").glob("*/certification.json")):
            c = cm.jload(p)
            name = p.parent.name
            prov = c.get("provenance", "")
            if not prov.endswith(".npy"):
                continue
            cands.append(dict(tag=f"jzfz_{name}", kind="jzfz_baseline", rho_path=prov, P=float(c["P"]), h=float(c["h"]),
                              pad=(0.08 if name == "final2" else 0.12), Q_target=None, certified=True, cert_path=str(p),
                              cert=_cert_summary(c), origin=f"Jz/Fz finalist {name} (parent-certified in qdz_parent/baselines)"))
    if "references" in include:
        cands.append(dict(tag="deck_cylinder_P588_h970", kind="reference", rho_path=None, geom=dict(disc_d=405.7), P=588.0,
                          h=970.5, pad=0.0, Q_target=None, certified=False, cert_path=None, cert={},
                          origin="SNU deck fabricated cylinder (SOURCE_AUDIT.md section 8), never optimized here"))
        cands.append(dict(tag="bare_ito_film_P825", kind="reference", rho_path="NONE", P=825.0, h=100.0, pad=0.0,
                          Q_target=None, certified=False, cert_path=None, cert={},
                          origin="air / ITO(23) / glass with an empty design layer (forward.build_sim rho=None): analytic control"))
    return cands


def get_rho(c):
    if c.get("rho_path") == "NONE":
        return None
    if c.get("rho_path"):
        return load_rho(c["rho_path"])
    if c.get("geom", {}).get("disc_d"):
        return disc(NX, c["P"], c["geom"]["disc_d"])
    raise ValueError(c["tag"])


def by_tag(tag):
    for c in registry():
        if c["tag"] == tag:
            return c
    raise KeyError(tag)
