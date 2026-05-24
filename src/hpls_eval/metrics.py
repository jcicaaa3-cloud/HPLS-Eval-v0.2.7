from __future__ import annotations

import numpy as np
from .masks import ensure_binary, boundary_band, connected_components


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = ensure_binary(pred)
    gt = ensure_binary(gt)
    intersection = np.logical_and(pred, gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float((2.0 * intersection + eps) / (denom + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = ensure_binary(pred)
    gt = ensure_binary(gt)
    union = np.logical_or(pred, gt).sum(dtype=np.float64)
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum(dtype=np.float64)
    return float((intersection + eps) / (union + eps))


def recall_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = ensure_binary(pred)
    gt = ensure_binary(gt)
    positives = gt.sum(dtype=np.float64)
    if positives == 0:
        return 1.0
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    return float((tp + eps) / (positives + eps))


def boundary_recall(pred: np.ndarray, gt: np.ndarray, radius: int = 2, eps: float = 1e-8) -> float:
    """Recall on GT boundary pixels.

    Samples with no GT boundary return NaN instead of 1.0. Otherwise datasets
    with many negative/empty masks can inflate boundary stability.
    """
    pred_b = boundary_band(pred, radius)
    gt_b = boundary_band(gt, radius)
    if gt_b.sum(dtype=np.float64) == 0:
        return float("nan")
    return recall_score(pred_b, gt_b, eps=eps)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, radius: int = 2, eps: float = 1e-8) -> float:
    """Boundary IoU excluding samples with empty GT boundary.

    Returning NaN is deliberate: blank GT masks are not treated as perfect
    boundary predictions. False boundary on blank GT is tracked separately by
    boundary_fp_empty_gt.
    """
    pred_b = boundary_band(pred, radius)
    gt_b = boundary_band(gt, radius)
    if gt_b.sum(dtype=np.float64) == 0:
        return float("nan")
    return iou_score(pred_b, gt_b, eps=eps)


def small_lesion_recall(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    area_threshold_px: int,
    connectivity: int = 8,
    aggregation: str = "pixel_micro",
    eps: float = 1e-8,
) -> float:
    """Recall over GT components whose area is <= area_threshold_px.

    Returns NaN when the sample contains no small region component. NaN is later
    ignored during aggregation, so absent small regions are not treated as
    perfect or failed recalls.
    """
    pred = ensure_binary(pred)
    gt = ensure_binary(gt)
    comps = [c for c in connected_components(gt, connectivity=connectivity) if int(c.sum()) <= area_threshold_px]
    if not comps:
        return float("nan")
    if aggregation == "pixel_micro":
        small_mask = np.logical_or.reduce(comps)
        return recall_score(pred, small_mask, eps=eps)
    if aggregation == "component_macro":
        vals = [recall_score(pred, comp, eps=eps) for comp in comps]
        return float(np.nanmean(vals))
    raise ValueError("small_region/small_lesion aggregation must be pixel_micro or component_macro")



def small_region_recall(*args, **kwargs) -> float:
    """Generic alias used for road-damage / visual-anomaly experiments.

    `small_lesion_*` names are retained as medical-domain aliases for future
    lesion segmentation experiments. The alias itself is not clinical evidence.
    """
    return small_lesion_recall(*args, **kwargs)

def compute_mask_metrics(pred: np.ndarray, gt: np.ndarray, cfg: dict) -> dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and GT mask shapes differ: pred={pred.shape}, gt={gt.shape}")
    eps = float(cfg.get("eps", 1e-8))
    boundary_cfg = cfg.get("boundary", {})
    small_cfg = cfg.get("small_region", cfg.get("small_lesion", {}))
    radius = int(boundary_cfg.get("radius", 2))

    pred_boundary = boundary_band(pred, radius)
    gt_boundary = boundary_band(gt, radius)
    pred_boundary_px = int(pred_boundary.sum(dtype=np.int64))
    gt_boundary_px = int(gt_boundary.sum(dtype=np.int64))
    empty_gt_boundary = gt_boundary_px == 0
    boundary_fp_empty_gt = bool(empty_gt_boundary and pred_boundary_px > 0)

    connectivity = int(small_cfg.get("connectivity", 8))
    small_components = [
        c for c in connected_components(gt, connectivity=connectivity)
        if int(c.sum()) <= int(small_cfg.get("area_threshold_px", 512))
    ]
    small_gt_px = int(sum(int(c.sum()) for c in small_components))

    return {
        "dice": dice_score(pred, gt, eps=eps),
        "iou": iou_score(pred, gt, eps=eps),
        "recall": recall_score(pred, gt, eps=eps),
        "boundary_recall": boundary_recall(pred, gt, radius=radius, eps=eps),
        "boundary_iou": boundary_iou(pred, gt, radius=radius, eps=eps),
        "gt_boundary_px": float(gt_boundary_px),
        "pred_boundary_px": float(pred_boundary_px),
        "empty_gt_boundary": float(1 if empty_gt_boundary else 0),
        "boundary_fp_empty_gt": float(1 if boundary_fp_empty_gt else 0),
        "small_region_recall": small_lesion_recall(
            pred,
            gt,
            area_threshold_px=int(small_cfg.get("area_threshold_px", 512)),
            connectivity=connectivity,
            aggregation=str(small_cfg.get("aggregation", "pixel_micro")),
            eps=eps,
        ),
        "has_small_region": float(1 if small_components else 0),
        "small_region_gt_components": float(len(small_components)),
        "small_region_gt_px": float(small_gt_px),
        # Medical-domain aliases retained for future lesion-segmentation runs.
        # They mirror small_region_* and must not be interpreted as evidence of
        # clinical validity.
        "small_lesion_recall": small_lesion_recall(
            pred,
            gt,
            area_threshold_px=int(small_cfg.get("area_threshold_px", 512)),
            connectivity=connectivity,
            aggregation=str(small_cfg.get("aggregation", "pixel_micro")),
            eps=eps,
        ),
        "has_small_lesion": float(1 if small_components else 0),
        "small_lesion_gt_components": float(len(small_components)),
        "small_lesion_gt_px": float(small_gt_px),
    }
