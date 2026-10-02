"""Constant-velocity Kalman filter over bounding boxes.

State: [cx, cy, w, h, vx, vy, vw, vh]; measurement: [cx, cy, w, h].
"""
from __future__ import annotations

import numpy as np


class BoxKalman:
    def __init__(self, box_xyxy: np.ndarray, dt: float = 1.0, q: float = 1.0, r: float = 4.0):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4) * dt
        self.H = np.eye(4, 8)
        self.Q = np.diag([q, q, q, q, 10 * q, 10 * q, q, q]) * 0.1
        self.R = np.eye(4) * r
        self.x = np.r_[xyxy_to_cxcywh(box_xyxy), np.zeros(4)]
        self.P = np.diag([10, 10, 10, 10, 1000, 1000, 1000, 1000.0])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return cxcywh_to_xyxy(self.x[:4])

    def update(self, box_xyxy: np.ndarray) -> None:
        z = xyxy_to_cxcywh(box_xyxy)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return cxcywh_to_xyxy(self.x[:4])


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], float)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
