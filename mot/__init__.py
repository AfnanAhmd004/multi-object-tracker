"""multi-object-tracker: SORT-style multi-object tracking with Kalman filtering and Hungarian matching."""
from .kalman import BoxKalman
from .metrics import clear_mot
from .synthetic import scenario
from .tracker import SortTracker, iou_matrix

__all__ = ["BoxKalman", "SortTracker", "clear_mot", "iou_matrix", "scenario"]
