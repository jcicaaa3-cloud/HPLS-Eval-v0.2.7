import math
import numpy as np

from hpls_eval.metrics import dice_score, iou_score, boundary_recall, boundary_iou, small_lesion_recall


def test_dice_and_iou_perfect_empty_masks():
    pred = np.zeros((8, 8), dtype=bool)
    gt = np.zeros((8, 8), dtype=bool)
    assert dice_score(pred, gt) == 1.0
    assert iou_score(pred, gt) == 1.0


def test_dice_and_iou_partial_overlap():
    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    pred[0:2, 0:2] = True  # 4 px
    gt[1:3, 1:3] = True    # 4 px, intersection 1
    assert abs(dice_score(pred, gt) - 0.25) < 1e-6
    assert abs(iou_score(pred, gt) - (1 / 7)) < 1e-6


def test_small_lesion_recall_pixel_micro():
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    gt[1:3, 1:3] = True      # 4 px small component
    gt[6:10, 6:10] = True    # 16 px large component
    pred[1:2, 1:3] = True    # 2/4 small component recalled
    val = small_lesion_recall(pred, gt, area_threshold_px=4, connectivity=8, aggregation="pixel_micro")
    assert abs(val - 0.5) < 1e-6


def test_small_lesion_recall_nan_when_no_small_component():
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    gt[1:6, 1:6] = True
    val = small_lesion_recall(pred, gt, area_threshold_px=4)
    assert math.isnan(val)


def test_boundary_recall_runs_on_simple_mask():
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    pred[2:6, 2:6] = True
    gt[2:6, 2:6] = True
    assert boundary_recall(pred, gt, radius=1) == 1.0


def test_boundary_metrics_nan_when_gt_boundary_empty():
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    assert math.isnan(boundary_recall(pred, gt, radius=1))
    assert math.isnan(boundary_iou(pred, gt, radius=1))



def test_boundary_false_positive_empty_gt_is_recorded():
    from hpls_eval.metrics import compute_mask_metrics
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    pred[2:6, 2:6] = True
    vals = compute_mask_metrics(pred, gt, {"boundary": {"radius": 1}, "small_lesion": {"area_threshold_px": 4}})
    assert math.isnan(vals["boundary_iou"])
    assert vals["empty_gt_boundary"] == 1.0
    assert vals["boundary_fp_empty_gt"] == 1.0
    assert vals["pred_boundary_px"] > 0


def test_compute_mask_metrics_outputs_small_region_aliases():
    from hpls_eval.metrics import compute_mask_metrics
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    gt[1:3, 1:3] = True
    pred[1:3, 1:3] = True
    vals = compute_mask_metrics(pred, gt, {"small_region": {"area_threshold_px": 4}, "boundary": {"radius": 1}})
    assert vals["small_region_recall"] == 1.0
    assert vals["small_lesion_recall"] == 1.0
    assert vals["small_region_gt_components"] == 1.0
