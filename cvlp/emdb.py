"""Fetch an EMDB entry: its metadata (REST API) and its primary map (FTP over https).

Downloads are cached in the run's output folder (emdb/), so a re-run with other
template parameters does not download again.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import urllib.request
from pathlib import Path

import mrcfile
import numpy as np

API = "https://www.ebi.ac.uk/emdb/api/entry/{id}"
FTP = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{id}/map/{file}"


def normalise_id(s):
    """'EMD-19708', 'emd_19708', '19708' -> 'EMD-19708'."""
    m = re.fullmatch(r"(?:emd[-_]?)?(\d{4,6})", str(s).strip(), flags=re.I)
    if not m:
        raise ValueError(f"not an EMDB id: {s!r}")
    return f"EMD-{m.group(1)}"


def _get(url, dest=None):
    req = urllib.request.Request(url, headers={"User-Agent": "cvlp"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if dest is None:
            return r.read()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)


def _find(obj, key):
    """All values stored under `key` anywhere in a nested JSON object."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out += _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            out += _find(v, key)
    return out


def _value(x):
    return x.get("valueOf_") if isinstance(x, dict) else x


def fetch_entry(emdb_id, cache_dir):
    """Metadata summary of an entry; the raw JSON is kept as emdb/<id>.json."""
    eid = normalise_id(emdb_id)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    raw = cache / f"{eid}.json"
    if not raw.exists():
        raw.write_bytes(_get(API.format(id=eid)))
    d = json.loads(raw.read_text())
    m = d["map"]
    point_groups = [p for p in _find(d, "point_group") if p]
    res = [_value(r) for r in _find(d, "resolution")]
    tri = [int(t) for t in _find(d, "triangulation") if str(t).isdigit()]
    title = _value(d.get("admin", {}).get("title")) or _value(d.get("sample", {}).get("name")) or ""
    if not tri:                                   # e.g. "... icosahedral capsid (T=9)"
        tri = [int(t) for t in re.findall(r"\bT\s*=\s*(\d+)", title)]
    return dict(
        id=eid,
        title=title,
        map_file=m["file"],
        apix=float(_value(m["pixel_spacing"]["x"])),
        dims=[int(m["dimensions"][k]) for k in ("col", "row", "sec")],
        point_group=point_groups[0] if point_groups else None,
        resolution=float(res[0]) if res and res[0] else None,
        contour=next((c["level"] for c in m.get("contour_list", {}).get("contour", []) if c.get("primary")), None),
        triangulation=tri[0] if tri else None,
        method=_find(d, "method")[0] if _find(d, "method") else None,
    )


def download_map(entry, cache_dir):
    """Primary map, gunzipped -> path. Skips the download if it is already there."""
    cache = Path(cache_dir)
    gz = cache / entry["map_file"]
    out = cache / entry["map_file"].removesuffix(".gz")
    if out.exists():
        return out
    url = FTP.format(id=entry["id"], file=entry["map_file"])
    print(f"  downloading {url}")
    _get(url, gz)
    with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 20)
    gz.unlink()
    return out


def read_map(path, max_apix=None):
    """Map -> (vol zyx float32, apix). Honours the header's axis order
    (mapc/mapr/maps), which EMDB maps are not guaranteed to have as 1/2/3.

    With max_apix, a finely sampled map is block-averaged by the largest whole
    factor that keeps the pixel at or below max_apix, slab by slab from a
    memory map, so a 2 GB deposition never has to fit in memory twice.
    """
    with mrcfile.mmap(str(path), permissive=True, mode="r") as m:
        apix = np.array([m.voxel_size.x, m.voxel_size.y, m.voxel_size.z], float)
        order = (int(m.header.mapc), int(m.header.mapr), int(m.header.maps))
        f = int(max_apix // apix[0]) if max_apix else 1
        if f >= 2:
            nz, ny, nx = (s // f for s in m.data.shape)
            vol = np.empty((nz, ny, nx), np.float32)
            for z in range(nz):
                slab = np.asarray(m.data[z * f:(z + 1) * f, :ny * f, :nx * f], np.float32)
                vol[z] = slab.reshape(f, ny, f, nx, f).mean(axis=(0, 2, 4))
            apix = apix * f
        else:
            vol = np.array(m.data, np.float32)
    if order != (1, 2, 3):
        # data axes are (sec, row, col) = axes (maps, mapr, mapc); want (z, y, x) = (3, 2, 1)
        file_axes = [order[2], order[1], order[0]]
        vol = np.transpose(vol, [file_axes.index(a) for a in (3, 2, 1)])
    if not np.allclose(apix, apix[0], rtol=1e-3):
        raise ValueError(f"{path}: anisotropic pixel size {apix}")
    if len(set(vol.shape)) != 1:
        raise ValueError(f"{path}: map is not cubic {vol.shape}")
    return np.ascontiguousarray(vol), float(apix[0])
