import numpy as np
import pytest

from mot import BoxKalman, SortTracker, clear_mot, iou_matrix, scenario


def test_iou_basic():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_kalman_learns_velocity():
    kf = BoxKalman(np.array([0, 0, 10, 10.0]))
    for k in range(1, 15):
        kf.predict(); kf.update(np.array([5 * k, 0, 5 * k + 10, 10.0]))
    assert kf.x[4] == pytest.approx(5, abs=0.5)
    assert kf.predict()[0] == pytest.approx(75, abs=2)


def test_tracker_keeps_identity_through_a_miss():
    tr = SortTracker(min_hits=1, max_age=3)
    ids = []
    for k in range(10):
        dets = [] if k == 5 else [[10 * k, 0, 10 * k + 30, 30]]
        out = tr.update(np.array(dets).reshape(-1, 4))
        ids += [i for i, _ in out]
    assert set(ids) == {0}


def test_tracker_improves_mota_over_raw_detections():
    dets, gts = scenario(n_frames=60, miss_rate=0.1, fp_rate=1.0, seed=1)
    tr = SortTracker()
    hyps = [tr.update(d) for d in dets]
    raw = [[(-(k + 1), b) for k, b in enumerate(d)] for d in dets]
    m_trk, m_raw = clear_mot(gts, hyps), clear_mot(gts, raw)
    assert m_trk["precision"] > m_raw["precision"]
    assert m_trk["MOTA"] > 0.6
