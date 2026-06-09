#!/usr/bin/env python3
"""Render the method pipeline diagram to PNG + PDF for the report.

Standalone, matplotlib-only. Writes:
  my_coord_attempt/pipeline_figure.png
  my_coord_attempt/pipeline_figure.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette ----
C_INPUT = "#dbeafe"; E_INPUT = "#3b82f6"   # blue  - inputs
C_HAND  = "#dcfce7"; E_HAND  = "#22c55e"   # green - hand evidence
C_PROC  = "#fef9c3"; E_PROC  = "#eab308"   # yellow- processing/fusion
C_DIFF  = "#fae8ff"; E_DIFF  = "#a855f7"   # purple- diffusion
C_OUT   = "#fee2e2"; E_OUT   = "#ef4444"   # red   - output

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.5); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, style="-", color="#444", lw=1.8, ls="solid"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=16,
                 lw=lw, color=color, linestyle=ls,
                 connectionstyle="arc3,rad=0.0"))

# ---------------- INPUTS (left) ----------------
box(0.2, 5.7, 2.6, 1.0, "Ego video\n(Aria RGB)", C_INPUT, E_INPUT, bold=True)
box(0.2, 4.3, 2.6, 1.0, "Camera\ntrajectory", C_INPUT, E_INPUT, bold=True)

# ---------------- HAND EVIDENCE ----------------
box(3.5, 6.5, 3.2, 0.95, "Dyn-HaMR [5]\n3D wrists + MANO fingers", C_HAND, E_HAND, fs=9.5, bold=True)
box(3.5, 5.35, 3.2, 0.95, "MediaPipe\n2D wrist pixels (+conf.)", C_HAND, E_HAND, fs=9.5, bold=True)

# ---------------- ALIGN + RELIABILITY ----------------
box(7.4, 6.5, 2.9, 0.95, "Align to UEM frame\n(similarity transform)", C_PROC, E_PROC, fs=9)
box(7.4, 5.35, 2.9, 0.95, "Reliability weighting\nconsistency + proj-error", C_PROC, E_PROC, fs=9)

# ---------------- UEM DIFFUSION (center) ----------------
box(3.3, 2.7, 4.2, 1.6,
    "UniEgoMotion [2]\nconditional motion diffusion\n(denoising loop)",
    C_DIFF, E_DIFF, fs=10.5, bold=True)

# in-diffusion guidance callout
box(8.0, 3.05, 2.7, 0.95, "(i) In-diffusion\nwrist guidance", C_DIFF, E_DIFF, fs=9.5, bold=True)

# ---------------- POST-INFERENCE ----------------
box(2.6, 0.5, 3.0, 1.45,
    "(ii) Wrist fusion\nprior+dyn+vel+acc+sep",
    C_PROC, E_PROC, fs=9, bold=True)
box(6.1, 0.5, 3.0, 1.45,
    "Body re-posing (arms)\n3D + 2D reproj + orient",
    C_PROC, E_PROC, fs=9, bold=True)
box(9.6, 0.5, 3.0, 1.45,
    "Finger transfer\n(Dyn-HaMR MANO)",
    C_HAND, E_HAND, fs=9, bold=True)

# ---------------- OUTPUT ----------------
box(10.9, 3.6, 1.9, 1.5, "Guided\nSMPL-X\nbody", C_OUT, E_OUT, fs=10.5, bold=True)

# ---------------- ARROWS ----------------
# inputs -> hand evidence (video) and -> UEM (both)
arrow(2.8, 6.1, 3.5, 6.97)      # video -> dynhamr
arrow(2.8, 5.85, 3.5, 5.85)     # video -> mediapipe
arrow(2.8, 4.8, 3.3, 3.9)       # trajectory -> UEM
arrow(2.8, 5.6, 3.3, 3.7)       # video features -> UEM
# hand evidence -> align/reliability
arrow(6.7, 6.97, 7.4, 6.97)
arrow(6.7, 5.82, 7.4, 5.82)
# align/reliability -> in-diffusion guidance
arrow(9.35, 6.5, 9.35, 4.0)
# in-diffusion guidance <-> UEM loop (bidirectional pair)
arrow(8.0, 3.62, 7.5, 3.62)
arrow(7.5, 3.28, 8.0, 3.28, color="#a855f7")
# UEM -> post-inference fusion
arrow(4.0, 2.7, 4.0, 1.95)
# reliability also feeds fusion (dashed)
arrow(7.5, 5.35, 4.3, 1.95, color="#eab308", ls="dashed", lw=1.3)
# fusion -> reposing -> fingers
arrow(5.6, 1.2, 6.1, 1.2)
arrow(9.1, 1.2, 9.6, 1.2)
# mediapipe also feeds reposing (dashed)
arrow(7.6, 5.35, 7.6, 1.95, color="#22c55e", ls="dashed", lw=1.3)
# fingers -> output
arrow(11.1, 1.95, 11.6, 3.6)

# ---------------- legend ----------------
ax.text(0.2, 1.7, "Stage colors:", fontsize=9, fontweight="bold")
for i,(c,e,lab) in enumerate([
        (C_INPUT,E_INPUT,"input"),(C_HAND,E_HAND,"hand evidence"),
        (C_PROC,E_PROC,"alignment / optimization"),
        (C_DIFF,E_DIFF,"diffusion"),(C_OUT,E_OUT,"output")]):
    yy = 1.35 - i*0.27
    ax.add_patch(FancyBboxPatch((0.2, yy), 0.32, 0.18,
                 boxstyle="round,pad=0.01,rounding_size=0.05", fc=c, ec=e, lw=1.4))
    ax.text(0.62, yy+0.09, lab, fontsize=8.5, va="center")
ax.text(0.2, 1.35-5*0.27, "dashed = reliability/2D weighting feed",
        fontsize=8, va="center", style="italic", color="#555")

plt.tight_layout()
for ext in ["png", "pdf"]:
    out = f"/work/courses/digital_human/team7/my_coord_attempt/pipeline_figure.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
