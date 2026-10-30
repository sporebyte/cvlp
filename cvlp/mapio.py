"""Reading and writing maps and RELION reference lists.

load_map / write_mrc are teact's (io/tomogram.py). Centre convention: the
symmetry centre of a map is voxel (n-1)/2, in xyz - what scipy's
map_coordinates needs, and what every cut in cvlp assumes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mrcfile
import numpy as np


@lru_cache(maxsize=8)
def load_map(path):
    """MRC -> (vol zyx float, apix, centre xyz in voxels, (n-1)/2 convention)."""
    with mrcfile.open(str(path), permissive=True) as m:
        vol = np.asarray(m.data, float)
        apix = float(m.voxel_size.x)
    if not apix or apix == 1.0:
        raise ValueError(f"{path}: suspicious pixel size {apix}")
    return vol, apix, (np.asarray(vol.shape, float)[::-1] - 1) / 2


def write_mrc(path, arr, apix):
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.asarray(arr, np.float32))
        m.voxel_size = apix


def write_reference_star(out_path, ref_paths):
    """The --ref list for relion_refine: one rlnReferenceImage per class."""
    lines = ["", "data_model_classes", "", "loop_", "_rlnReferenceImage #1", *map(str, ref_paths), ""]
    Path(out_path).write_text("\n".join(lines))
