"""QC panels for a template run."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _figure(**kw):
    """A Figure that is not managed by pyplot, so writing PNGs never changes the
    backend of a running GUI."""
    from matplotlib.figure import Figure
    return Figure(**kw)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def write_all(params, scores, out):

    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(exist_ok=True)
    pr = params["_profiles"]

    # radial profile and per-orbit site profiles (copia panel A)
    fig = _figure(figsize=(12, 4)); ax = fig.subplots(1, 2)
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
    _save(fig, fig_dir / "A_map_profiles.png")

    # lattice test
    if scores:
        labels = list(dict.fromkeys(s["label"] for s in scores))
        morph = list(dict.fromkeys(s["morphology"] for s in scores))
        fig = _figure(figsize=(max(6, 0.5 * len(labels)), 3.5)); ax = fig.subplots()
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
        _save(fig, fig_dir / "B_lattice_test.png")

    files = params["files"]["references"]
    rows = 4 if params["files"].get("references_unfiltered") else 2
    fig = _figure(figsize=(2.6 * len(files), 2.7 * rows))
    template_panels(fig, params, out)
    _save(fig, fig_dir / "C_templates.png")
    return fig_dir


def template_panels(fig, params, out, unfiltered=True):
    """Top view (z = 0 slice) and side view (xz) of every template with the mask
    outline; below them the unfiltered cuts (5-slice slabs, the fine grid is
    thin). Draws into `fig` (also used by the GUI)."""
    import mrcfile

    files = params["files"]["references"]
    unf = params["files"].get("references_unfiltered", {}) if unfiltered else {}
    mask = mrcfile.read(Path(out) / params["files"]["mask"])
    b = mask.shape[0] // 2
    rows = 4 if unf else 2
    ax = fig.subplots(rows, len(files), squeeze=False)
    for col, (n, f) in enumerate(files.items()):
        v = mrcfile.read(Path(out) / f)
        panels = [(v[b], mask[b], f"{n} template, top (z=0)"), (v[:, b, :], mask[:, b, :], "template, side (xz), out = up")]
        if n in unf:
            u = mrcfile.read(Path(out) / unf[n])
            c = u.shape[0] // 2
            panels += [(u[c - 2:c + 3].mean(0), None, f"{n} unfiltered, top"),
                       (u[:, c - 2:c + 3, :].mean(1), None, "unfiltered, side")]
        for row, (img, m, t) in enumerate(panels):
            a = ax[row, col]
            a.imshow(img, origin="lower", cmap="gray")
            if m is not None:
                a.contour(m, levels=[0.5], colors="#ff7f0e", linewidths=0.8)
            a.set_title(t, fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    return ax
