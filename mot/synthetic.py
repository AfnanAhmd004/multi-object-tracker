"""Synthetic detection streams with noise, missed detections and false positives."""
from __future__ import annotations

import numpy as np


def scenario(n_objects: int = 5, n_frames: int = 80, size=(640, 480), miss_rate: float = 0.1,
             fp_rate: float = 1.0, noise: float = 2.0, seed: int = 0):
    """Returns (detections_per_frame, ground_truth_per_frame) where gt is a list of (id, box)."""
    rng = np.random.default_rng(seed)
    W, H = size
    pos = rng.uniform([50, 50], [W - 50, H - 50], (n_objects, 2))
    vel = rng.uniform(-6, 6, (n_objects, 2))
    wh = rng.uniform([30, 40], [60, 90], (n_objects, 2))
    dets, gts = [], []
    for _ in range(n_frames):
        pos += vel
        for i in range(2):  # bounce
            out = (pos[:, i] < 30) | (pos[:, i] > (W, H)[i] - 30)
            vel[out, i] *= -1
        boxes = np.c_[pos - wh / 2, pos + wh / 2]
        gts.append([(i, boxes[i].copy()) for i in range(n_objects)])
        seen = boxes[rng.random(n_objects) > miss_rate]
        seen = seen + rng.normal(0, noise, seen.shape)
        n_fp = rng.poisson(fp_rate)
        fp_c = rng.uniform([0, 0], [W, H], (n_fp, 2)); fp_wh = rng.uniform(20, 60, (n_fp, 2))
        dets.append(np.r_[seen, np.c_[fp_c - fp_wh / 2, fp_c + fp_wh / 2]])
    return dets, gts
