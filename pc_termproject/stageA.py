"""Stage A: parametric exploration of physically distinct C4v topology families at the exact grading basis
(Mmax=9, official formulation) on the 66-point irreducible wedge of the official 21x11 grid (exact for C4v)."""
import numpy as np, json, time, sys, itertools
from pwem import *

LOG = "/home/user/ENZ/pc_termproject/logs/stageA.jsonl"
import os; os.makedirs(os.path.dirname(LOG), exist_ok=True)

def fam_circle_rod(r):            return geom_circle(r)
def fam_square_rod(s):            return geom_square(s)
def fam_diamond_rod(s):           return geom_square(s, rot45=True)
def fam_circle_hole(r):           return ~geom_circle(r) if r <= 0.5 else ~(geom_circle(r) | geom_corner_circle(0)) 
def fam_circle_hole_big(r):       # holes overlap for r>0.5 -> use periodic union of neighbouring holes
    X, Y = meshgrid_xy(); m = np.zeros_like(X, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            m |= ((X - dx)**2 + (Y - dy)**2) <= r**2
    return ~m
def fam_square_hole(s):           return ~geom_square(s)
def fam_diamond_hole(s):          return ~geom_square(s, rot45=True)
def fam_rod_veins(r, w):          return geom_circle(r) | geom_veins(w)
def fam_rod_diagveins(r, w):      return geom_circle(r) | geom_veins(w, diag=True)
def fam_cross(L, w):              return geom_cross(L, w)
def fam_cross45(L, w):            return geom_cross(L, w, rot45=True)
def fam_ring(ro, ri):             return geom_ring(ro, ri)
def fam_hole_rod(rh, rr):         return (~geom_circle(rh)) | geom_circle(rr)          # Si frame with circular hole + isolated centre rod
def fam_frame_rod(w, rr):         return geom_edge_veins(w) | geom_circle(rr)          # square Si frame (veins on cell edges) + centre rod
def fam_diamondhole_rod(s, rr):   return (~geom_square(s, rot45=True)) | geom_circle(rr)
def fam_corner_center(r1, r2):    return geom_corner_circle(r1) | geom_circle(r2)      # two-sublattice rods (corner + centre)
def fam_veins_cornerrod(w, r1):   return geom_veins(w) | geom_corner_circle(r1)        # veins through centre + rods at corners (rods sit in the air pockets)
def fam_square_rod_veins(s, w):   return geom_square(s) | geom_veins(w)
def fam_diamond_rod_veins(s, w):  return geom_square(s, rot45=True) | geom_veins(w)
def fam_diamond_rod_diagveins(s, w): return geom_square(s, rot45=True) | geom_veins(w, diag=True)

FAMILIES = [
    ("circle_rod",      fam_circle_rod,      [np.round(np.arange(0.08, 0.50, 0.02), 3)]),
    ("square_rod",      fam_square_rod,      [np.round(np.arange(0.15, 0.95, 0.05), 3)]),
    ("diamond_rod",     fam_diamond_rod,     [np.round(np.arange(0.15, 0.72, 0.03), 3)]),
    ("circle_hole",     fam_circle_hole_big, [np.round(np.arange(0.20, 0.72, 0.02), 3)]),
    ("square_hole",     fam_square_hole,     [np.round(np.arange(0.30, 0.97, 0.03), 3)]),
    ("diamond_hole",    fam_diamond_hole,    [np.round(np.arange(0.30, 0.72, 0.03), 3)]),
    ("rod_veins",       fam_rod_veins,       [np.round(np.arange(0.08, 0.38, 0.03), 3), np.round(np.arange(0.02, 0.24, 0.02), 3)]),
    ("rod_diagveins",   fam_rod_diagveins,   [np.round(np.arange(0.08, 0.38, 0.03), 3), np.round(np.arange(0.02, 0.24, 0.02), 3)]),
    ("cross",           fam_cross,           [np.round(np.arange(0.3, 1.01, 0.1), 3), np.round(np.arange(0.05, 0.36, 0.05), 3)]),
    ("cross45",         fam_cross45,         [np.round(np.arange(0.3, 1.45, 0.1), 3), np.round(np.arange(0.05, 0.36, 0.05), 3)]),
    ("ring",            fam_ring,            [np.round(np.arange(0.20, 0.51, 0.05), 3), np.round(np.arange(0.05, 0.41, 0.05), 3)]),
    ("hole_rod",        fam_hole_rod,        [np.round(np.arange(0.30, 0.50, 0.025), 3), np.round(np.arange(0.05, 0.28, 0.025), 3)]),
    ("frame_rod",       fam_frame_rod,       [np.round(np.arange(0.04, 0.25, 0.04), 3), np.round(np.arange(0.05, 0.36, 0.025), 3)]),
    ("diamondhole_rod", fam_diamondhole_rod, [np.round(np.arange(0.40, 0.72, 0.05), 3), np.round(np.arange(0.05, 0.28, 0.025), 3)]),
    ("corner_center",   fam_corner_center,   [np.round(np.arange(0.05, 0.36, 0.05), 3), np.round(np.arange(0.05, 0.36, 0.05), 3)]),
    ("veins_cornerrod", fam_veins_cornerrod, [np.round(np.arange(0.04, 0.25, 0.04), 3), np.round(np.arange(0.05, 0.36, 0.05), 3)]),
    ("square_rod_veins",fam_square_rod_veins,[np.round(np.arange(0.15, 0.66, 0.05), 3), np.round(np.arange(0.02, 0.24, 0.04), 3)]),
    ("diamond_rod_veins",fam_diamond_rod_veins,[np.round(np.arange(0.15, 0.66, 0.05), 3), np.round(np.arange(0.02, 0.24, 0.04), 3)]),
    ("diamond_rod_diagveins",fam_diamond_rod_diagveins,[np.round(np.arange(0.15, 0.66, 0.05), 3), np.round(np.arange(0.02, 0.24, 0.04), 3)]),
]

def run():
    only = sys.argv[1:] 
    total = sum(int(np.prod([len(g) for g in grids])) for _, _, grids in FAMILIES if not only or _ in only)
    print(f"Stage A: {total} evaluations planned", flush=True)
    done = 0; t_start = time.time()
    with open(LOG, "a") as f:
        for name, fn, grids in FAMILIES:
            if only and name not in only: continue
            best = None
            for params in itertools.product(*grids):
                params = [float(p) for p in params]
                mask = fn(*params)
                if not is_c4v(mask):
                    print("  WARNING non-C4v mask", name, params); 
                fill = fill_fraction(mask)
                if fill < 0.02 or fill > 0.98:
                    continue
                r = evaluate(mask_to_eps(mask), Mmax=9, nbands=10, wedge=True)
                rec = dict(family=name, params=params, fill=fill, score=r['score'], gap=r['gap'],
                           tm_gaps=r['tm_gaps'][:5], te_gaps=r['te_gaps'][:5],
                           tm_gap_at_target=r['tm_gap_at_target'], te_gap_at_target=r['te_gap_at_target'],
                           runtime=r['runtime_s'])
                f.write(json.dumps(rec) + "\n"); f.flush()
                done += 1
                g = r['gap']
                gs = f"gap [{g['w_low']:.4f},{g['w_high']:.4f}] TM{g['n_tm']+1}-{g['n_tm']+2}/TE{g['n_te']+1}-{g['n_te']+2}" if g else "no complete gap at target"
                tmg = r['tm_gap_at_target']; teg = r['te_gap_at_target']
                print(f"[{done}/{total} {time.time()-t_start:6.0f}s] {name:22s} {str(params):26s} fill={fill:.3f} score={100*r['score']:6.2f}%  {gs}"
                      f"  | TM@t={'-' if tmg is None else f'[{tmg[0]:.3f},{tmg[1]:.3f}]'} TE@t={'-' if teg is None else f'[{teg[0]:.3f},{teg[1]:.3f}]'}", flush=True)
                if best is None or r['score'] > best[0]:
                    best = (r['score'], params, fill, gs)
            print(f"=== best {name}: score={100*best[0]:.2f}% params={best[1]} fill={best[2]:.3f} {best[3]}" if best else f"=== {name}: nothing", flush=True)

if __name__ == "__main__":
    run()
