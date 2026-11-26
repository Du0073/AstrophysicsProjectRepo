"""Hubble Sequence taxonomy mapping for SDSS Galaxy10 benchmarks & CNN classifiers."""

from __future__ import annotations

import numpy as np

# Official Galaxy10 SDSS 10-class names (from astroNN.datasets.galaxy10sdss)
GALAXY10_SDSS_NAMES: dict[int, str] = {
    0: "Disk, Face-on, No Spiral",
    1: "Smooth, Completely round",
    2: "Smooth, in-between round",
    3: "Smooth, Cigar shaped",
    4: "Disk, Edge-on, Rounded Bulge",
    5: "Disk, Edge-on, Boxy Bulge",
    6: "Disk, Edge-on, No Bulge",
    7: "Disk, Face-on, Tight Spiral",
    8: "Disk, Face-on, Medium Spiral",
    9: "Disk, Face-on, Loose Spiral",
}

# 5-class Hubble Sequence taxonomy mapped cleanly from Galaxy10 SDSS
HUBBLE_CLASSES: tuple[str, ...] = (
    "Elliptical",
    "Lenticular/Edge-on",
    "Tight Spiral (Sa)",
    "Medium Spiral (Sb)",
    "Loose Spiral / Irregular (Sc/Irr)",
)

GALAXY10_TO_HUBBLE_INDEX: dict[int, int] = {
    1: 0,  # Smooth, Completely round -> Elliptical
    2: 0,  # Smooth, in-between round -> Elliptical
    3: 0,  # Smooth, Cigar shaped -> Elliptical
    0: 1,  # Disk, Face-on, No Spiral -> Lenticular/Edge-on (S0)
    4: 1,  # Disk, Edge-on, Rounded Bulge -> Lenticular/Edge-on
    5: 1,  # Disk, Edge-on, Boxy Bulge -> Lenticular/Edge-on
    6: 1,  # Disk, Edge-on, No Bulge -> Lenticular/Edge-on
    7: 2,  # Disk, Face-on, Tight Spiral -> Tight Spiral (Sa)
    8: 3,  # Disk, Face-on, Medium Spiral -> Medium Spiral (Sb)
    9: 4,  # Disk, Face-on, Loose Spiral -> Loose Spiral / Irregular (Sc/Irr)
}


def map_galaxy10_to_hubble(labels_10: np.ndarray) -> np.ndarray:
    """Convert integer Galaxy10 SDSS labels (0..9) to 5-class Hubble Sequence indices (0..4)."""
    arr = np.asarray(labels_10, dtype=np.int64)
    out = np.empty_like(arr)
    for g10_idx, hubble_idx in GALAXY10_TO_HUBBLE_INDEX.items():
        out[arr == g10_idx] = hubble_idx
    return out


def aggregate_10class_probs_to_hubble(probs_10: np.ndarray) -> np.ndarray:
    """Sum 10-class Galaxy10 probability vectors into 5-class Hubble Sequence probabilities."""
    p10 = np.asarray(probs_10, dtype=np.float32)
    single = p10.ndim == 1
    if single:
        p10 = p10[None, :]
    p5 = np.zeros((p10.shape[0], len(HUBBLE_CLASSES)), dtype=np.float32)
    for g10_idx, hubble_idx in GALAXY10_TO_HUBBLE_INDEX.items():
        p5[:, hubble_idx] += p10[:, g10_idx]
    return p5[0] if single else p5
