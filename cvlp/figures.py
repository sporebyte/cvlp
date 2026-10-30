"""QC panels for a template run."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_all(params, scores, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mrcfile

    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(exist_ok=True)
    pr = params["_profiles"]

    # radial profile and per-orbit site profiles (copia panel A)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(pr["radii"], pr["profile"], lw=2)
    for k, c in (("inner_layer", "#888"), ("gap_radius", "#d62728"), ("outer_layer", "#2ca02c"),
                 ("mask_r_outer", "#9467bd")):
        if params[k] is not None:
            ax[0].axvline(params[k], ls="--", color=c, label=f"{k} {params[k]:.0f} A")
    ax[0].set(xlabel="radius (A)", ylabel="mean density", title="spherical average")
    ax[0].legend(fontsize=8)
    for n, p in pr["site_prof"].items():
        ax[1].plot(pr["site_r"], p, lw=2, label=f"{n}: peak {pr['site_r'][np.argmax(p)]:.1f} A")
    lo = params["gap_radius"] - 20
    ax[1].set(xlabel="radius (A)", title="density on each orbit's sites (inner layer removed)",
              xlim=(lo, params["mask_r_outer"] + 30))
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "A_map_profiles.png", dpi=130); plt.close(fig)

    # lattice test
    if scores:
        labels = list(dict.fromkeys(s["label"] for s in scores))
        morph = list(dict.fromkeys(s["morphology"] for s in scores))
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), 3.5))
        x = np.arange(len(labels))
        w = 0.8 / len(morph)
        for i, mo in enumerate(morph):
            sc = {s["label"]: s["score"] for s in scores if s["morphology"] == mo}
            ax.bar(x - 0.4 + w * (i + 0.5), [sc[l] for l in labels], w, label=f"{mo} on the lattice")
        ax.axhline(0.8, color="k", lw=0.8, ls="--")
        ax.set_xticks(x, [l.replace(" (h,k)=", "\n") for l in labels], fontsize=7)
        ax.set(ylim=(0, 1.05), ylabel="min(recall, precision)",
               title=f"lattice test (finest passing wins): T={params['T']} (h,k)=({params['h']},{params['k']})")
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout(); plt.savefig(fig_dir / "B_lattice_test.png", dpi=130); plt.close(fig)

    # the templates: top view (z = 0 slice) and side view (xz), mask outline
    files = params["files"]["references"]
    mask = mrcfile.read(Path(out) / params["files"]["mask"])
    b = mask.shape[0] // 2
    fig, ax = plt.subplots(2, len(files), figsize=(2.6 * len(files), 5.4), squeeze=False)
    for col, (n, f) in enumerate(files.items()):
        v = mrcfile.read(Path(out) / f)
        for row, (img, m, t) in enumerate(((v[b], mask[b], "top (z=0)"), (v[:, b, :], mask[:, b, :], "side (xz), out = up"))):
            a = ax[row, col]
            a.imshow(img, origin="lower", cmap="gray")
            a.contour(m, levels=[0.5], colors="#ff7f0e", linewidths=0.8)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{n} {t}" if row == 0 else t, fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "C_templates.png", dpi=130); plt.close(fig)
    return fig_dir
