"""cvlp-gui: load a map, look at the cutouts, set the parameters, save.

    cvlp-gui            (or: python -m cvlp.gui)

1. Load: an EMDB id, or your own map file (or both: a local copy of an entry).
   Downloads, finds the icosahedral frame and the lattice. Slow (~1 min), once.
2. Preview: cuts the templates with the current parameters into a hidden
   preview folder and shows them - low-passed with the mask outline, and
   unfiltered from the map. Re-run after changing a parameter (~20-60 s).
   "3D view" opens the references as rotatable surfaces (cvlp.viewer).
3. Save: copies the previewed files into the output folder and writes
   params.json and the QC figures, exactly as the command line would.

The preview runs the same code as `cvlp`, so what you see is what you save.
Long steps run in a background thread; the log at the bottom follows them.
"""
from __future__ import annotations

import queue
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import cli, figures, pipeline, recipe

PREVIEW = ".preview"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("cvlp: capsomer templates")
        self.geometry("1400x900")
        self.q = queue.Queue()
        self.state = None            # pipeline state after Load
        self.params = None           # params of the current preview
        self.preview_args = None     # the template parameters it was made with
        self.busy = False
        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)

        self.v = {k: tk.StringVar(value=d) for k, d in (
            ("emdb", ""), ("map", ""), ("out", ""), ("T", ""), ("hk", ""),
            ("apix", "11.8"), ("box", ""), ("mask", ""), ("arc", f"{recipe.MAX_ARC_DEG:g}"),
            ("lowpass", "nyquist"), ("gap", ""), ("prefix", ""))}
        self.v["unfiltered"] = tk.BooleanVar(value=True)

        src = ttk.LabelFrame(left, text="1  Map", padding=6)
        src.pack(fill="x")
        self._row(src, "EMDB id", "emdb", hint="e.g. EMD-19708")
        self._row(src, "or map file", "map", browse=self._pick_map)
        self._row(src, "output folder", "out", browse=self._pick_out, hint="default: the id or the file name")
        self.btn_load = ttk.Button(src, text="Load map and find lattice", command=self.on_load)
        self.btn_load.pack(fill="x", pady=(6, 0))

        lat = ttk.LabelFrame(left, text="Lattice", padding=6)
        lat.pack(fill="x", pady=8)
        self.lbl_lattice = ttk.Label(lat, text="not loaded", wraplength=330, justify="left")
        self.lbl_lattice.pack(fill="x")
        self._row(lat, "T (blank: from map)", "T")
        self._row(lat, "(h,k) (blank: from map)", "hk", hint="e.g. 2,1")
        self.btn_lattice = ttk.Button(lat, text="Re-check lattice", command=self.on_lattice, state="disabled")
        self.btn_lattice.pack(fill="x", pady=(6, 0))

        par = ttk.LabelFrame(left, text="2  Templates", padding=6)
        par.pack(fill="x")
        self._row(par, "pixel size (A/px)", "apix", hint="of your subtomograms")
        self._row(par, "box (px)", "box", hint="blank: auto")
        self._row(par, "mask diameter (A)", "mask", hint="blank: auto")
        self._row(par, "max arc (deg)", "arc")
        f = ttk.Frame(par); f.pack(fill="x", pady=1)
        ttk.Label(f, text="low-pass", width=20).pack(side="left")
        ttk.Combobox(f, textvariable=self.v["lowpass"], values=("copia", "nyquist", "30", "40"),
                     width=12).pack(side="left", fill="x", expand=True)
        ttk.Label(par, text="copia = Gaussian (50% at ~4.5 px), nyquist = Fourier to 2 px, or a resolution in A",
                  wraplength=330, foreground="#666").pack(fill="x")
        self._row(par, "gap radius (A)", "gap", hint="blank: measured")
        self._row(par, "RELION prefix", "prefix", hint="blank: the output folder")
        ttk.Checkbutton(par, text="show unfiltered cutouts", variable=self.v["unfiltered"],
                        command=self._draw).pack(anchor="w", pady=(4, 0))
        self.btn_preview = ttk.Button(par, text="Preview cutouts", command=self.on_preview, state="disabled")
        self.btn_preview.pack(fill="x", pady=(6, 0))
        self.btn_3d = ttk.Button(par, text="3D view (rotate unfiltered and template)", command=self.on_3d,
                                 state="disabled")
        self.btn_3d.pack(fill="x", pady=(4, 0))

        sv = ttk.LabelFrame(left, text="3  Save", padding=6)
        sv.pack(fill="x", pady=8)
        self.btn_save = ttk.Button(sv, text="Save templates to output folder", command=self.on_save, state="disabled")
        self.btn_save.pack(fill="x")

        self.lbl_info = ttk.Label(right, text="Load a map to start.", justify="left")
        self.lbl_info.pack(fill="x")
        self.fig = Figure(figsize=(9, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        logf = ttk.Frame(right)
        logf.pack(fill="x")
        self.log = tk.Text(logf, height=11, font=("Menlo", 10), wrap="none")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="x", expand=True)

    def _row(self, parent, label, key, browse=None, hint=None):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=1)
        ttk.Label(f, text=label, width=20).pack(side="left")
        ttk.Entry(f, textvariable=self.v[key], width=18).pack(side="left", fill="x", expand=True)
        if browse:
            ttk.Button(f, text="...", width=3, command=browse).pack(side="left")
        if hint:
            ttk.Label(parent, text=hint, foreground="#666").pack(anchor="e")

    def _pick_map(self):
        p = filedialog.askopenfilename(title="Map file", filetypes=[("maps", "*.mrc *.map *.ccp4"), ("all", "*")])
        if p:
            self.v["map"].set(p)

    def _pick_out(self):
        p = filedialog.askdirectory(title="Output folder")
        if p:
            self.v["out"].set(p)

    # ------------------------------------------------------------------ inputs
    def _num(self, key, kind=float):
        s = self.v[key].get().strip()
        return kind(s) if s else None

    def _out(self):
        s = self.v["out"].get().strip()
        if s:
            return Path(s).expanduser()
        return cli.default_out(self.v["emdb"].get().strip() or None, self.v["map"].get().strip() or None)

    def _template_args(self):
        return dict(apix=self._num("apix"), box=self._num("box", int), mask_diameter=self._num("mask"),
                    max_arc=self._num("arc") or recipe.MAX_ARC_DEG,
                    lowpass=cli.lowpass_arg(self.v["lowpass"].get().strip() or "copia"),
                    gap_radius=self._num("gap"),
                    relion_prefix=self.v["prefix"].get().strip() or str(self._out()))

    # ------------------------------------------------------------------ actions
    def on_load(self):
        emdb_id = self.v["emdb"].get().strip() or None
        map_path = self.v["map"].get().strip() or None
        if not emdb_id and not map_path:
            return messagebox.showinfo("cvlp", "Give an EMDB id or choose a map file.")
        out, apix = self._out(), self._num("apix") or 11.8
        T, hk, gap = self._num("T", int), self.v["hk"].get().strip() or None, self._num("gap")

        def job():
            st = pipeline.load(out, emdb_id=emdb_id, map_path=map_path and Path(map_path).expanduser(),
                               apix=apix, log=self._log)
            pipeline.lattice(st, T=T, hk=hk, gap_radius=gap, log=self._log)
            return st
        self._run(job, self._loaded)

    def _loaded(self, st):
        self.state = st
        if not self.v["out"].get().strip():
            self.v["out"].set(str(st["out"]))
        self._show_lattice()
        for b in (self.btn_lattice, self.btn_preview):
            b.configure(state="normal")
        self.on_preview()

    def _show_lattice(self):
        st = self.state
        m = st["model"]
        t = f"T={m['T']}  (h,k)=({m['h']},{m['k']})  {', '.join(f'{n} x{c}' for n, c in m['counts'].items())}"
        if st["morphology"]:
            t += f"\n{st['morphology']} on the lattice"
        e = st["entry"]
        if e and e.get("triangulation"):
            t += f"\nEMDB entry: T={e['triangulation']}"
        sm = st["surface"]
        t += f"\nsurface: vertex {sm['vertex']:.0f} A, face {sm['face']:.0f} A" + (" (faceted)" if st["faceted"] else "")
        self.lbl_lattice.configure(text=t)

    def on_lattice(self):
        st, T, hk, gap = self.state, self._num("T", int), self.v["hk"].get().strip() or None, self._num("gap")
        self._run(lambda: pipeline.lattice(st, T=T, hk=hk, gap_radius=gap, log=self._log),
                  lambda st: (self._show_lattice(), self.on_preview()))

    def on_preview(self):
        try:
            st, kw = self.state, self._template_args()
        except ValueError as e:
            return messagebox.showerror("cvlp", f"Check the parameters: {e}")
        prev = Path(st["out"]) / PREVIEW
        shutil.rmtree(prev, ignore_errors=True)
        self.preview_args = kw
        self._run(lambda: pipeline.templates(st, prev, write_figures=False, log=self._log, **kw), self._previewed)

    def _previewed(self, params):
        self.params = params
        self.btn_save.configure(state="normal")
        self.btn_3d.configure(state="normal")
        self._draw()

    def on_3d(self):
        from .viewer import Viewer
        Viewer(self, Path(self.state["out"]) / PREVIEW, self.params)

    def on_save(self):
        st, params = self.state, self.params
        out, prev = self._out(), Path(st["out"]) / PREVIEW
        try:
            changed = self._template_args() != self.preview_args
        except ValueError:
            changed = True
        if changed and not messagebox.askyesno(
                "cvlp", "The parameters changed since the preview. Save the previewed templates anyway?\n"
                        "(Choose No and press Preview to apply the new ones.)"):
            return

        def job():
            out.mkdir(parents=True, exist_ok=True)
            shutil.copytree(prev, out, dirs_exist_ok=True)
            figures.write_all(params, st["scores"], out)
            return out
        self._run(job, lambda o: self._log(f"=== saved to {o}/  (templates, mask, star files, params.json, figures)"))

    # ------------------------------------------------------------------ drawing
    def _draw(self):
        if not self.params:
            return
        p = self.params
        self.fig.clear()
        figures.template_panels(self.fig, p, Path(self.state["out"]) / PREVIEW, unfiltered=self.v["unfiltered"].get())
        self.fig.tight_layout()
        self.canvas.draw_idle()
        lp = p["lowpass"]
        lp_txt = (f"copia Gaussian (50% at {lp['half_amplitude_A']} A)" if lp["mode"] == "copia"
                  else f"Fourier to {lp['cutoff_A']:g} A (50% at {lp['half_amplitude_A']} A)")
        names = list(p["correlation_in_mask"])
        cc = "   ".join(f"{a}-{b} {p['correlation_in_mask'][a][b]:.2f}" for i, a in enumerate(names)
                        for b in names[i + 1:])
        self.lbl_info.configure(text=(
            f"T={p['T']}   box {p['box']} px x {p['bin_apix']} A = {p['box'] * p['bin_apix']:.0f} A   "
            f"mask {p['mask_diameter']:.0f} A (+-{p['mask_arc_deg']:.0f} deg)   low-pass {lp_txt}\n"
            f"correlation in the mask (lower = easier to tell apart):  {cc}"
            + ("\nsmall capsid: oversampling works best on larger VLPs (T>=7)" if p.get("small_capsid") else "")))

    # ------------------------------------------------------------------ threading
    def _log(self, s):
        self.q.put(("log", s))

    def _run(self, job, done):
        if self.busy:
            return
        self._set_busy(True)

        def work():
            try:
                self.q.put(("done", (done, job())))
            except Exception as e:                       # shown, not raised: the GUI stays up
                self.q.put(("error", e))
        threading.Thread(target=work, daemon=True).start()

    def _set_busy(self, busy):
        self.busy = busy
        st = "disabled" if busy else "normal"
        self.btn_load.configure(state=st)
        for b in (self.btn_lattice, self.btn_preview):
            b.configure(state=st if self.state else "disabled")
        self.btn_save.configure(state=st if self.params else "disabled")
        self.btn_3d.configure(state=st if self.params else "disabled")
        self.configure(cursor="watch" if busy else "")

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", val + "\n")
                    self.log.see("end")
                elif kind == "done":
                    self._set_busy(False)
                    val[0](val[1])
                else:
                    self._set_busy(False)
                    self.log.insert("end", f"ERROR: {val}\n")
                    self.log.see("end")
                    messagebox.showerror("cvlp", str(val))
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main():
    matplotlib.use("TkAgg")
    App().mainloop()


if __name__ == "__main__":
    main()
