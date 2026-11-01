"""3D view of the references: rotate the unfiltered cut and the template side by side.

    cvlp-view EMD-19708          (or: python -m cvlp.viewer EMD-19708) - any cvlp output folder
    or the "3D view" button in cvlp-gui

Each class is shown twice, in the same box (in A) and the same orientation:
left, the unfiltered cut from the map (original resolution, on the fine
grid); right, the low-passed template as it goes to RELION, optionally only
inside the mask. Drag either panel to rotate; the other follows. The sliders
set each surface's contour level, in standard deviations of that box.
"""
from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import mrcfile
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

MAX_FACES = 60000       # per surface; more makes dragging sluggish in matplotlib


def surface(vol, apix, level, max_faces=MAX_FACES):
    """Isosurface of a (z, y, x) volume -> (verts xyz in A from the box centre, faces).
    The marching-cubes step grows until the surface has at most max_faces triangles."""
    from skimage.measure import marching_cubes
    if not vol.min() < level < vol.max():
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    step = 1
    while True:
        v, f, _, _ = marching_cubes(vol, level, spacing=(apix,) * 3, step_size=step)
        if len(f) <= max_faces or step >= 6:
            break
        step += 1
    c = (np.array(vol.shape) - 1) / 2 * apix
    return (v - c)[:, ::-1], f                        # (z, y, x) -> (x, y, z), centred


def default_level(vol, contour=None):
    """The EMDB author contour if there is one (the unfiltered cuts keep the map's
    density scale), else Otsu's threshold."""
    if contour is not None and vol.min() < contour < vol.max():
        return float(contour)
    from skimage.filters import threshold_otsu
    return float(threshold_otsu(vol))


class Viewer(tk.Toplevel):
    def __init__(self, master, folder, params=None):
        super().__init__(master)
        self.folder = Path(folder)
        self.p = params or json.load(open(self.folder / "params.json"))
        self.title(f"cvlp 3D view: {self.folder.name}")
        self.geometry("1300x780")
        files = self.p["files"]
        self.names = list(files["references"])
        self.mask = mrcfile.read(self.folder / files["mask"]).astype(np.float32)
        self.contour = ((self.p.get("emdb") or {}).get("contour"))
        self.cache = {}
        self.v = dict(name=tk.StringVar(value=self.names[0]), lev_u=tk.DoubleVar(value=0.0),
                      lev_t=tk.DoubleVar(value=0.0), masked=tk.BooleanVar(value=True))
        self._build()
        self.load_class()

    # ------------------------------------------------------------------ layout
    def _build(self):
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Label(bar, text="class").pack(side="left")
        cb = ttk.Combobox(bar, textvariable=self.v["name"], values=self.names, width=8, state="readonly")
        cb.pack(side="left", padx=(4, 16))
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_class())
        ttk.Checkbutton(bar, text="template only inside the mask (what RELION uses)", variable=self.v["masked"],
                        command=lambda: self.redraw("t", fresh=True)).pack(side="left")
        for txt, el, az in (("top", 90, -90), ("side", 0, -90), ("oblique", 25, -60)):
            ttk.Button(bar, text=txt, width=7, command=lambda el=el, az=az: self.set_view(el, az)).pack(side="right")
        ttk.Label(bar, text="view:").pack(side="right", padx=(0, 4))

        self.fig = Figure(figsize=(12, 6.4))
        self.ax = {"u": self.fig.add_subplot(1, 2, 1, projection="3d"),
                   "t": self.fig.add_subplot(1, 2, 2, projection="3d")}
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._sync)

        sl = ttk.Frame(self, padding=6)
        sl.pack(fill="x")
        self.lbl = {}
        for col, (k, txt) in enumerate((("u", "unfiltered level"), ("t", "template level"))):
            f = ttk.Frame(sl)
            f.grid(row=0, column=col, sticky="ew", padx=10)
            sl.columnconfigure(col, weight=1)
            ttk.Label(f, text=txt, width=16).pack(side="left")
            s = ttk.Scale(f, from_=-1.0, to=8.0, variable=self.v[f"lev_{k}"], orient="horizontal")
            s.pack(side="left", fill="x", expand=True)
            s.bind("<ButtonRelease-1>", lambda e, k=k: self.redraw(k, fresh=True))
            self.lbl[k] = ttk.Label(f, width=26)
            self.lbl[k].pack(side="left")
        ttk.Label(self, foreground="#666", padding=(6, 0, 6, 6),
                  text="Drag a panel to rotate (the other follows); right-drag to zoom. Levels are in "
                       "standard deviations of the box; release the slider to redraw.").pack(anchor="w")

    # ------------------------------------------------------------------ data
    def load_class(self):
        n = self.v["name"].get()
        if n not in self.cache:
            files = self.p["files"]
            t = mrcfile.open(self.folder / files["references"][n], permissive=True)
            tvol, tapix = t.data.astype(np.float32), float(t.voxel_size.x)
            t.close()
            unf = files.get("references_unfiltered", {}).get(n)
            if unf:
                u = mrcfile.open(self.folder / unf, permissive=True)
                uvol, uapix = u.data.astype(np.float32), float(u.voxel_size.x)
                u.close()
            else:
                uvol, uapix = None, None
            self.cache[n] = dict(t=(tvol, tapix), u=(uvol, uapix))
        for k in ("u", "t"):
            vol = self._vol(k)
            if vol is None:
                continue
            lev = default_level(vol, self.contour if k == "u" else None)
            self.v[f"lev_{k}"].set(round((lev - vol.mean()) / vol.std(), 2))
        self.redraw("u", fresh=True)
        self.redraw("t", fresh=True)

    def _vol(self, k):
        d = self.cache[self.v["name"].get()]
        vol, _ = d[k]
        if k == "t" and vol is not None and self.v["masked"].get():
            vol = vol * self.mask
        return vol

    # ------------------------------------------------------------------ drawing
    def redraw(self, k, fresh=False):
        n = self.v["name"].get()
        ax = self.ax[k]
        el, az = getattr(ax, "elev", 25), getattr(ax, "azim", -60)
        ax.cla()
        vol = self._vol(k)
        apix = self.cache[n][k][1]
        title = {"u": f"{n}: unfiltered, {apix:.2f} A/px",
                 "t": f"{n}: template, {apix:.1f} A/px, low-pass {self._lowpass_txt()}"}[k]
        ax.set_title(title, fontsize=9)
        if vol is None:
            ax.text2D(0.5, 0.5, "no unfiltered cut in this folder", transform=ax.transAxes, ha="center")
            self.canvas.draw_idle()
            return
        level = vol.mean() + self.v[f"lev_{k}"].get() * vol.std()
        verts, faces = surface(vol, apix, level)
        if len(faces):
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], lw=0, antialiased=False, shade=True,
                            color="#c9a227" if k == "u" else "#4a90c2")
        half = vol.shape[0] / 2 * apix
        for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            set_lim(-half, half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x (A)", fontsize=7); ax.set_ylabel("y (A)", fontsize=7); ax.set_zlabel("z, out (A)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=el, azim=az)
        self.lbl[k].configure(text=f"{self.v[f'lev_{k}'].get():+.2f} sd = {level:.3g}  ({len(faces):,} triangles)")
        self.canvas.draw_idle()

    def _lowpass_txt(self):
        lp = self.p.get("lowpass")
        if not isinstance(lp, dict):
            return "copia"
        return "copia Gaussian" if lp["mode"] == "copia" else f"{lp['cutoff_A']:g} A"

    def _sync(self, event):
        """Mirror the dragged panel's view onto the other one."""
        if event.button is None or event.inaxes not in self.ax.values():
            return
        src = event.inaxes
        dst = self.ax["t"] if src is self.ax["u"] else self.ax["u"]
        if (dst.elev, dst.azim) != (src.elev, src.azim):
            dst.view_init(elev=src.elev, azim=src.azim)
            self.canvas.draw_idle()

    def set_view(self, el, az):
        for ax in self.ax.values():
            ax.view_init(elev=el, azim=az)
        self.canvas.draw_idle()


def main(argv=None):
    import matplotlib
    matplotlib.use("TkAgg")
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 1
    folder = Path(argv[0]).expanduser()
    if not (folder / "params.json").exists():
        print(f"{folder}: no params.json - give a cvlp output folder")
        return 1
    root = tk.Tk()
    root.withdraw()
    v = Viewer(root, folder)
    v.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
