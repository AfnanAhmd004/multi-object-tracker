"""SORT-style tracker: Kalman prediction, IoU cost, Hungarian assignment and track lifecycle."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import BoxKalman


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


@dataclass
class TrackState:
    id: int
    kf: BoxKalman
    hits: int = 1
    age: int = 0
    misses: int = 0

    @property
    def box(self):
        return self.kf.box


class SortTracker:
    def __init__(self, iou_threshold: float = 0.3, min_hits: int = 3, max_age: int = 5):
        self.iou_threshold, self.min_hits, self.max_age = iou_threshold, min_hits, max_age
        self.tracks: list[TrackState] = []
        self._next_id = 0

    def update(self, detections: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Consume one frame of [x1, y1, x2, y2] detections; return confirmed (id, box) pairs."""
        detections = np.asarray(detections, float).reshape(-1, 4)
        preds = np.array([t.kf.predict() for t in self.tracks]).reshape(-1, 4)
        for t in self.tracks:
            t.age += 1
        matches, unmatched_dets = self._associate(preds, detections)
        matched_tracks = set()
        for ti, di in matches:
            tr = self.tracks[ti]
            tr.kf.update(detections[di]); tr.hits += 1; tr.misses = 0
            matched_tracks.add(ti)
        for ti, tr in enumerate(self.tracks):
            if ti not in matched_tracks:
                tr.misses += 1
        for di in unmatched_dets:
            self.tracks.append(TrackState(self._next_id, BoxKalman(detections[di])))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return [(t.id, t.box) for t in self.tracks if t.hits >= self.min_hits and t.misses == 0]

    def _associate(self, preds, dets):
        if len(preds) == 0 or len(dets) == 0:
            return [], list(range(len(dets)))
        iou = iou_matrix(preds, dets)
        rows, cols = linear_sum_assignment(-iou)
        matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= self.iou_threshold]
        matched = {c for _, c in matches}
        return matches, [d for d in range(len(dets)) if d not in matched]
