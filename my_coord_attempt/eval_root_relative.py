"""
Post-hoc root-relative / Procrustes wrist evaluation + Dyn-HaMR ablation report.

Reads the *_traj_align_eval.npz files written by run_merged_ablation.py
(`--save_eval_npz`) and reports, per estimator, three flavours of wrist MPJPE:

  world     : raw world-frame error (what the original log printed). Includes the
              whole-body root/trajectory placement error.
  root-rel  : after subtracting each body's own pelvis root. Removes translational
              placement; keeps global-orientation error.
  procrustes: after a per-sequence similarity (scale+R+t) fit of the predicted
              wrist track onto GT. Pure trajectory-shape / articulation error.

This is the metric that matches what the eye sees when matching the rendered ball
to the GT mesh's hand (a local comparison, not a world-frame one).

Two modes:
  Single dir : python eval_root_relative.py --dir <run_eval_dir>
  Ablation   : python eval_root_relative.py --full_dir <A> --ablation_dir <B>
               Prints the FULL run, the no-Dyn-HaMR run, and their delta so you
               can read off the causal value of Dyn-HaMR's 3D/hand contribution.

The npz arrays are recomputed here from the stored wrist/root tracks, so the
numbers are independent of whatever metric the run happened to print.
"""

import os
import glob
import argparse
import numpy as np


# --------------------------------------------------------------------------
# Metric primitives — inlined (kept byte-for-byte identical to the definitions
# in run_merged_ablation.py) so this tool is fully standalone and does NOT
# import the heavy runner (which needs PYTHONPATH=. and the UniEgoMotion cwd).
# --------------------------------------------------------------------------
def mean_l2(a, b, mask=None):
    if mask is None:
        return float(np.linalg.norm(a - b, axis=-1).mean())
    if mask.sum() == 0:
        return np.nan
    return float(np.linalg.norm(a[mask] - b[mask], axis=-1).mean())


def _procrustes_align_traj(pred, gt, mask):
    """Per-sequence rigid (scale+R+t) Procrustes of `pred` onto `gt` over valid
    frames, applied to the whole sequence. Identity fallback if <3 valid frames."""
    if mask.sum() < 3:
        return pred.copy()
    A = pred[mask]
    B = gt[mask]
    mu_A = A.mean(0); mu_B = B.mean(0)
    AA = A - mu_A;    BB = B - mu_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    var_A = (AA ** 2).sum()
    scale = float(np.sum(S) / var_A) if var_A > 1e-12 else 1.0
    t = mu_B - scale * (R @ mu_A)
    return scale * (pred @ R.T) + t


# Estimators we score, and which stored arrays back them.
#   key -> (left_array, right_array, root_array_or_None)
# root_array None means "no body root stored" -> root-rel is skipped for it.
ESTIMATORS = [
    ("uem",      "pred_left_wrist",     "pred_right_wrist",     "pred_root"),
    ("dyn_hamr", "dyn_left_aligned",    "dyn_right_aligned",    "pred_root"),
    ("guided",   "guided_left_wrist",   "guided_right_wrist",   "pred_root"),
    ("smpl_opt", "smpl_opt_left_wrist", "smpl_opt_right_wrist", "smpl_opt_root"),
]


def _get(npz, key):
    return npz[key] if key in npz.files else None


def eval_one_clip(npz):
    """Return {estimator: {variant_side: value}} for one clip's npz."""
    gt_l = _get(npz, "gt_left_wrist")
    gt_r = _get(npz, "gt_right_wrist")
    gt_root = _get(npz, "gt_root")
    lv = _get(npz, "left_valid").astype(bool)
    rv = _get(npz, "right_valid").astype(bool)

    out = {}
    for name, lk, rk, rootk in ESTIMATORS:
        l = _get(npz, lk)
        r = _get(npz, rk)
        if l is None or r is None:
            continue
        n = min(len(l), len(r), len(gt_l), len(gt_r), len(lv), len(rv))
        l, r = l[:n], r[:n]
        gl, gr = gt_l[:n], gt_r[:n]
        lvn, rvn = lv[:n], rv[:n]
        pred_root = _get(npz, rootk)
        gtr = gt_root

        rec = {}
        # World frame.
        rec["L_world"] = mean_l2(l, gl, lvn)
        rec["R_world"] = mean_l2(r, gr, rvn)
        # Root-relative (needs both roots).
        if pred_root is not None and gtr is not None:
            pr = pred_root[:n]
            gtrn = gtr[:n]
            rec["L_rootrel"] = mean_l2(l - pr, gl - gtrn, lvn)
            rec["R_rootrel"] = mean_l2(r - pr, gr - gtrn, rvn)
        # Procrustes.
        rec["L_proc"] = mean_l2(_procrustes_align_traj(l, gl, lvn), gl, lvn)
        rec["R_proc"] = mean_l2(_procrustes_align_traj(r, gr, rvn), gr, rvn)
        out[name] = rec
    return out


def aggregate_dir(eval_dir):
    """Mean each metric across all clips in a dir. Returns {est: {metric: mean}}."""
    files = sorted(glob.glob(os.path.join(eval_dir, "*_traj_align_eval.npz")))
    if not files:
        raise SystemExit(f"[ERR] No *_traj_align_eval.npz under {eval_dir}")
    acc = {}  # est -> metric -> list
    for f in files:
        npz = np.load(f, allow_pickle=True)
        per = eval_one_clip(npz)
        for est, rec in per.items():
            for k, v in rec.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                acc.setdefault(est, {}).setdefault(k, []).append(float(v))
    agg = {est: {k: float(np.mean(vs)) for k, vs in d.items()}
           for est, d in acc.items()}
    return agg, len(files)


VARIANTS = ["L_world", "L_rootrel", "L_proc", "R_world", "R_rootrel", "R_proc"]
EST_ORDER = ["uem", "dyn_hamr", "guided", "smpl_opt"]


def _fmt(v):
    return f"{v:7.4f}" if v is not None and not np.isnan(v) else "   --  "


def print_table(title, agg, n_clips):
    print(f"\n=== {title}  (mean over {n_clips} clips, meters) ===")
    header = "estimator   | " + " | ".join(f"{v:>10s}" for v in VARIANTS)
    print(header)
    print("-" * len(header))
    for est in EST_ORDER:
        if est not in agg:
            continue
        row = [f"{est:11s}"]
        for v in VARIANTS:
            val = agg[est].get(v)
            row.append(f"{_fmt(val):>10s}")
        print(" | ".join(row))


def print_upper_bound(agg, n_clips):
    """Dyn-HaMR's best-case value: the Procrustes columns ARE the per-clip
    optimal similarity (scale+R+t) fit of each estimator's wrist track DIRECTLY
    onto GT. For dyn_hamr this is the theoretical upper bound -- the smallest
    wrist error Dyn-HaMR could ever achieve on these clips if it were placed and
    scaled perfectly. Comparing it to uem's same-column number answers 'is there
    any per-frame signal in Dyn-HaMR that UEM lacks?' If dyn_hamr_proc is not
    clearly below uem_proc, Dyn-HaMR has no extractable advantage, no matter how
    cleverly it is aligned or weighted."""
    print(f"\n=== Dyn-HaMR upper bound: optimal-to-GT (Procrustes) wrist error "
          f"(mean over {n_clips} clips, meters) ===")
    print("    Each number = per-clip similarity fit of that track ONTO GT, then")
    print("    residual. Lower = better. This strips ALL placement/scale handicap.\n")
    print(f"  {'estimator':11s} | {'L_proc':>8s} | {'R_proc':>8s}")
    print("  " + "-" * 34)
    for est in EST_ORDER:
        if est not in agg:
            continue
        l = agg[est].get("L_proc"); r = agg[est].get("R_proc")
        print(f"  {est:11s} | {_fmt(l):>8s} | {_fmt(r):>8s}")
    u = agg.get("uem", {}); d = agg.get("dyn_hamr", {})
    if u and d and all(k in u and k in d for k in ("L_proc", "R_proc")):
        dl = d["L_proc"] - u["L_proc"]
        dr = d["R_proc"] - u["R_proc"]
        print(f"\n  dyn_hamr - uem (optimal-to-GT):  L {dl:+.4f}   R {dr:+.4f}")
        verdict = ("Dyn-HaMR beats UEM even at its best -> real extractable signal"
                   if (dl < -0.002 or dr < -0.002) else
                   "Dyn-HaMR is NOT better than UEM even at its optimal fit -> "
                   "no extractable per-frame advantage; guiding toward it cannot help")
        print(f"  Verdict: {verdict}")


def print_delta(full, ablt):
    print("\n=== Dyn-HaMR causal value: (no_dyn_hamr) - (full)  [+ = Dyn-HaMR helps] ===")
    print("    Positive delta means removing Dyn-HaMR made the error WORSE,")
    print("    i.e. Dyn-HaMR was helping by that many meters.\n")
    header = "estimator   | " + " | ".join(f"{v:>10s}" for v in VARIANTS)
    print(header)
    print("-" * len(header))
    for est in EST_ORDER:
        if est not in full or est not in ablt:
            continue
        row = [f"{est:11s}"]
        for v in VARIANTS:
            a = full[est].get(v)
            b = ablt[est].get(v)
            if a is None or b is None or np.isnan(a) or np.isnan(b):
                row.append(f"{'--':>10s}")
            else:
                row.append(f"{(b - a):+10.4f}")
        print(" | ".join(row))
    print("\n  Read smpl_opt (the final mesh wrist) for the bottom line: that delta")
    print("  is what Dyn-HaMR's 3D + finger contribution is worth on the shipped output.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="Single eval dir to summarise.")
    ap.add_argument("--full_dir", default=None,
                    help="Eval dir of the FULL pipeline run.")
    ap.add_argument("--ablation_dir", default=None,
                    help="Eval dir of the no_dyn_hamr run.")
    args = ap.parse_args()

    if args.dir:
        agg, n = aggregate_dir(args.dir)
        print_table(f"Run: {args.dir}", agg, n)
        print_upper_bound(agg, n)
        return

    if args.full_dir and args.ablation_dir:
        full, nf = aggregate_dir(args.full_dir)
        ablt, na = aggregate_dir(args.ablation_dir)
        print_table("FULL pipeline (Dyn-HaMR ON)", full, nf)
        print_table("ABLATION (Dyn-HaMR OFF, 2D reproj only)", ablt, na)
        print_upper_bound(full, nf)
        print_delta(full, ablt)
        return

    ap.error("Provide --dir, or both --full_dir and --ablation_dir.")


if __name__ == "__main__":
    main()
