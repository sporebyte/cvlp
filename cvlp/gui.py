"""cvlp-gui: load a map, look at the cutouts, set the parameters, save.

    cvlp-gui            (or: python -m cvlp.gui)

1. Load: an EMDB id, or your own map file (or both: a local copy of an entry).
   Downloads, finds the icosahedral frame and the lattice. Slow (~1 min), once.
2. Preview: cuts the templates with the current parameters into a hidden
   preview folder and shows them - low-passed with the mask outline, and
   unfiltered from the map. Re-run after changing a parameter (~20-60 s).
   "Open figures folder" shows the QC figures in the file browser, among them
   the radial density profiles of the map being cut (A_map_profiles.png).
3. Save: copies the previewed files (templates, unfiltered references, mask,
   star files, params.json, figures) into the output folder, exactly as the
   command line would write them.

The preview runs the same code as `cvlp`, so what you see is what you save.
Long steps run in a background thread; the log at the bottom follows them.
"""
from __future__ import annotations

import queue
import shutil
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import cli, figures, pipeline, recipe

PREVIEW = ".preview"


def open_folder(path):
    """Show a folder in the system file browser (Finder, Explorer, or xdg-open)."""
    import os
    import subprocess
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform.startswith("win"):
        os.startfile(str(path))
    else:
        subprocess.Popen(["xdg-open", str(path)])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("cvlp: capsomer templates")
        w, h = min(1400, self.winfo_screenwidth() - 60), min(900, self.winfo_screenheight() - 110)
        self.geometry(f"{w}x{h}+20+20")             # fits a laptop screen, not just a big monitor
        self.minsize(900, 500)
        self.q = queue.Queue()
        self.state = None            # pipeline state after Load
        self.params = None           # params of the current preview
        self.preview_args = None     # the template parameters it was made with
        self.saved_to = None         # where that preview was saved, if it was
        self.busy = False
        self._build()
        self._poll_id = self.after(100, self._poll)

    def destroy(self):
        self.after_cancel(self._poll_id)
        super().destroy()

    # ------------------------------------------------------------------ layout
    def _scrolling_panel(self, parent):
        """A frame that scrolls vertically when it is taller than the window
        (the scrollbar only shows then). -> the frame to fill."""
        holder = ttk.Frame(parent)
        holder.pack(side="top", fill="both", expand=True)
        bg = ttk.Style().lookup("TFrame", "background") or self.cget("background")
        cv = tk.Canvas(holder, highlightthickness=0, background=bg)
        sb = ttk.Scrollbar(holder, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(cv, padding=8)
        win = cv.create_window((0, 0), window=body, anchor="nw")

        def fit(_=None):
            cv.configure(scrollregion=cv.bbox("all"), width=body.winfo_reqwidth())
            if body.winfo_reqheight() > cv.winfo_height() > 1:
                sb.pack(side="right", fill="y")
            else:
                sb.pack_forget()
                cv.yview_moveto(0)
        body.bind("<Configure>", fit)
        cv.bind("<Configure>", lambda e: (cv.itemconfigure(win, width=e.width), fit()))

        def wheel(e):                                   # macOS deltas are small, Windows' are +-120
            step = -e.delta if sys.platform == "darwin" else -e.delta // 120
            if sb.winfo_ismapped():
                cv.yview_scroll(step, "units")
        cv.bind("<Enter>", lambda e: (cv.bind_all("<MouseWheel>", wheel),
                                      cv.bind_all("<Button-4>", lambda e: cv.yview_scroll(-1, "units")),
                                      cv.bind_all("<Button-5>", lambda e: cv.yview_scroll(1, "units"))))
        cv.bind("<Leave>", lambda e: [cv.unbind_all(s) for s in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
        return body

    def _build(self):
        side = ttk.Frame(self)
        side.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)
        # Save is pinned to the bottom so it is always on screen; the rest scrolls if it does not fit
        sv = ttk.LabelFrame(side, text="3  Save", padding=6)
        sv.pack(side="bottom", fill="x", padx=8, pady=8)
        left = self._scrolling_panel(side)

        self.v = {k: tk.StringVar(value=d) for k, d in (
            ("emdb", ""), ("map", ""), ("out", ""), ("T", ""), ("hk", ""),
            ("apix", "11.8"), ("box", ""), ("mask", ""), ("arc", f"{recipe.MAX_ARC_DEG:g}"),
            ("lowpass", "nyquist"), ("gap", ""), ("prefix", ""))}
        self.v["unfiltered"] = tk.BooleanVar(value=True)

        src = ttk.LabelFrame(left, text="1  Map", padding=6)
        src.pack(fill="x")
        self._row(src, "EMDB id", "emdb", hint="EMD-19708")
        self._row(src, "or map file", "map", browse=self._pick_map)
        self._row(src, "output folder", "out", browse=self._pick_out, hint="blank: id/file")
        self.btn_load = ttk.Button(src, text="Load map and find lattice", command=self.on_load)
        self.btn_load.pack(fill="x", pady=(6, 0))

        lat = ttk.LabelFrame(left, text="Lattice", padding=6)
        lat.pack(fill="x", pady=8)
        self.lbl_lattice = ttk.Label(lat, text="not loaded", wraplength=330, justify="left")
        self.lbl_lattice.pack(fill="x")
        self._row(lat, "T", "T", hint="blank: map")
        self._row(lat, "(h,k)", "hk", hint="e.g. 2,1")
        self.btn_lattice = ttk.Button(lat, text="Re-check lattice", command=self.on_lattice, state="disabled")
        self.btn_lattice.pack(fill="x", pady=(6, 0))

        par = ttk.LabelFrame(left, text="2  Templates", padding=6)
        par.pack(fill="x")
        self._row(par, "pixel size (A/px)", "apix", hint="subtomograms")
        self._row(par, "box (px)", "box", hint="blank: auto")
        self._row(par, "mask diameter (A)", "mask", hint="blank: auto")
        self._row(par, "max arc (deg)", "arc", hint="small capsids")
        f = ttk.Frame(par); f.pack(fill="x", pady=1)
        ttk.Label(f, text="low-pass", width=17).pack(side="left")
        ttk.Combobox(f, textvariable=self.v["lowpass"], values=("copia", "nyquist", "30", "40"),
                     width=12).pack(side="left", fill="x", expand=True)
        ttk.Label(par, text="copia = Gaussian (50% at ~4.5 px), nyquist = Fourier to 2 px, or a resolution in A",
                  wraplength=330, foreground="#666").pack(fill="x")
        self._row(par, "gap radius (A)", "gap", hint="blank: measured")
        self._row(par, "RELION prefix", "prefix", hint="blank: output")
        ttk.Checkbutton(par, text="show unfiltered cutouts", variable=self.v["unfiltered"],
                        command=self._draw).pack(anchor="w", pady=(4, 0))
        self.btn_preview = ttk.Button(par, text="Preview cutouts", command=self.on_preview, state="disabled")
        self.btn_preview.pack(fill="x", pady=(6, 0))
        self.btn_figs = ttk.Button(par, text="Open figures folder", command=self.on_figures, state="disabled")
        self.btn_figs.pack(fill="x", pady=(4, 0))
        ttk.Label(par, text="radial density of the map, lattice test, templates", foreground="#666").pack(anchor="e")

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
        ttk.Label(f, text=label, width=17).pack(side="left")
        if hint:                                         # on the same row, to keep the panel short
            ttk.Label(f, text=hint, foreground="#666", width=12).pack(side="right")
        if browse:
            ttk.Button(f, text="...", width=3, command=browse).pack(side="right")
        ttk.Entry(f, textvariable=self.v[key], width=9).pack(side="left", fill="x", expand=True)

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
        self.saved_to = None
        self._run(lambda: pipeline.templates(st, prev, log=self._log, **kw), self._previewed)

    def _previewed(self, params):
        self.params = params
        self.btn_save.configure(state="normal")
        self.btn_figs.configure(state="normal")
        self._draw()

    def on_figures(self):
        """The saved figures if this preview was saved, else the preview's."""
        folder = (self.saved_to or Path(self.state["out"]) / PREVIEW) / "figures"
        if not folder.is_dir():
            return messagebox.showinfo("cvlp", f"No figures yet in {folder}")
        open_folder(folder)
        self._log(f"  opened {folder}")

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
            shutil.copytree(prev, out, dirs_exist_ok=True)       # the preview already holds the figures
            return out

        def done(o):
            self.saved_to = o
            self._log(f"=== saved to {o}/  (templates, unfiltered references, mask, star files, params.json, figures)")
        self._run(job, done)

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
        self.btn_figs.configure(state=st if self.params else "disabled")
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
        self._poll_id = self.after(100, self._poll)


def main():
    matplotlib.use("TkAgg")
    App().mainloop()


if __name__ == "__main__":
    main()
