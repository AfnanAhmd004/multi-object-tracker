"""Track five objects through noisy detections with misses and false positives; compare with raw detections."""
import numpy as np

from mot import SortTracker, clear_mot, scenario

dets, gts = scenario(n_objects=5, n_frames=120, miss_rate=0.15, fp_rate=1.5, seed=4)
tracker = SortTracker(iou_threshold=0.3, min_hits=3, max_age=5)
hyps = [tracker.update(d) for d in dets]
raw = [[(-(k + 1), b) for k, b in enumerate(d)] for d in dets]  # every detection is its own "track"

print("raw detections :", {k: round(v, 3) for k, v in clear_mot(gts, raw).items() if k != "id_switches"})
print("SORT tracker   :", {k: round(v, 3) if isinstance(v, float) else v for k, v in clear_mot(gts, hyps).items()})
