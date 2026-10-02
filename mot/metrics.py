"""CLEAR-MOT style metrics: MOTA, ID switches, precision and recall."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker import iou_matrix


def clear_mot(gts_per_frame, hyps_per_frame, iou_threshold: float = 0.5) -> dict[str, float]:
    fn = fp = idsw = tp = n_gt = 0
    last_match: dict[int, int] = {}
    for gts, hyps in zip(gts_per_frame, hyps_per_frame):
        n_gt += len(gts)
        g_ids = [g for g, _ in gts]; h_ids = [h for h, _ in hyps]
        gb = np.array([b for _, b in gts]).reshape(-1, 4); hb = np.array([b for _, b in hyps]).reshape(-1, 4)
        iou = iou_matrix(gb, hb)
        pairs = []
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            pairs = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_threshold]
        tp += len(pairs)
        fn += len(gts) - len(pairs)
        fp += len(hyps) - len(pairs)
        for i, j in pairs:
            g, h = g_ids[i], h_ids[j]
            if g in last_match and last_match[g] != h:
                idsw += 1
            last_match[g] = h
    return {
        "MOTA": 1 - (fn + fp + idsw) / max(1, n_gt),
        "id_switches": idsw,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, n_gt),
    }
