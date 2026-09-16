"""Exact LOADED certification of selected finalists with the existing, never-run qdz_parent machinery
(hybrid_certify.certify_loaded): real-ITO R/T/A/F_c spectrum at the map order, loaded AAA poles, the parent
poles of the same geometry, and the loss-scaling continuation gamma(s) = gamma_rad + s gamma_nr through
Im(eps_ITO) x s, s in {1, .5, .25, .1, .03, 0} (analysis.track_branch / fit_gamma) -> exact Q_rad, Q_nr and
gamma_rad/gamma_nr of the loaded mode, which is the exact counterpart of Stage B's first-order
`critical_coupling_estimate`.

    python loaded_certify_finalists.py --tags <tag> [<tag> ...] [--order 7 7] [--threads 4]
Outputs: outputs/loaded_certification/<tag>/loaded_certification.json (+ loaded_spectrum.npz)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

import common as cm                       # noqa: F401
import config                             # noqa: E402
import forward as fwd                     # noqa: E402
import candidates as cd                   # noqa: E402
import hybrid_certify as hc               # noqa: E402  (qdz_parent, unchanged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--order", type=int, nargs=2, default=[7, 7])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-track", action="store_true")
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    for tag in a.tags:
        c = cd.by_tag(tag); rho = torch.as_tensor(cd.get_rho(c), dtype=config.GEO_DTYPE)
        out = cm.OUT / "loaded_certification" / tag
        res = hc.certify_loaded(rho, c["P"], c["h"], tag, out, order=list(a.order), track=not a.no_track)
        res["meta"] = dict(candidate=c, rho_sha256=cm.sha256_array(cd.get_rho(c)), provenance=cm.provenance(order=a.order))
        cm.jdump(res, out / "loaded_certification.json")
        gf = res.get("gamma_fit") or {}
        print(f"[loaded-cert {tag}] loaded poles: {[(round(p['lambda_nm'], 1), round(p['Q'], 1)) for p in res['loaded_poles']]} | "
              f"gamma fit: Q_rad={gf.get('Q_rad')} Q_nr={gf.get('Q_nr')} ratio={gf.get('gamma_ratio')} resid={gf.get('linearity_resid')} n={gf.get('n_levels')}", flush=True)


if __name__ == "__main__":
    main()
