"""COCO evaluation and a fixed-threshold detection summary."""

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _coco_results(ground_truth: COCO, predictions: list[dict]) -> COCO:
    if predictions:
        return ground_truth.loadRes(predictions)
    empty = COCO()
    empty.dataset = {
        "info": ground_truth.dataset.get("info", {}),
        "images": ground_truth.dataset["images"],
        "categories": ground_truth.dataset["categories"],
        "annotations": [],
    }
    empty.createIndex()
    return empty


def evaluate(annotation_path, predictions: list[dict], detection_confidence: float, matching_iou: float, max_detections: int) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))
        detections = _coco_results(ground_truth, predictions)
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = sorted(ground_truth.getImgIds())
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    precision_array = evaluator.eval["precision"]
    per_class = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        values = precision_array[:, :, index, 0, 2]
        valid = values[values >= 0]
        gt_count = sum(1 for item in ground_truth.dataset["annotations"] if item["category_id"] == category_id and not item.get("iscrowd", 0))
        category = ground_truth.cats[category_id]
        per_class[category["name"]] = {
            "ground_truth_count": gt_count,
            "ap50_95": float(np.mean(valid)) if gt_count >= 5 and valid.size else None,
        }

    # Reuse COCOeval's crowd handling, ignore flags, and one-to-one matches.
    selected = [item for item in predictions if item["score"] >= detection_confidence]
    with contextlib.redirect_stdout(io.StringIO()):
        selected_coco = _coco_results(ground_truth, selected)
        matched = COCOeval(ground_truth, selected_coco, "bbox")
        matched.params.imgIds = sorted(ground_truth.getImgIds())
        matched.params.iouThrs = np.array([matching_iou])
        matched.params.maxDets = [1, 10, max_detections]
        matched.evaluate()
    true_positive = false_positive = false_negative = 0
    for result in matched.evalImgs:
        if result is None or result["aRng"] != matched.params.areaRng[0] or result["maxDet"] != max_detections:
            continue
        gt_ignore = np.asarray(result["gtIgnore"], dtype=bool)
        dt_ignore = np.asarray(result["dtIgnore"][0], dtype=bool)
        gt_matched = result["gtMatches"][0]
        dt_matched = result["dtMatches"][0]
        true_positive += int(np.sum((dt_matched > 0) & ~dt_ignore))
        false_positive += int(np.sum((dt_matched == 0) & ~dt_ignore))
        false_negative += int(np.sum((gt_matched == 0) & ~gt_ignore))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    return {
        "map50": float(stats[1]) if stats[1] >= 0 else None,
        "map50_95": float(stats[0]) if stats[0] >= 0 else None,
        "precision": precision,
        "recall": recall,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "per_class": per_class,
        "evaluation_settings": {
            "ap_iou_thresholds": "0.50:0.05:0.95",
            "ap_max_detections": max_detections,
            "count_confidence": detection_confidence,
            "count_iou": matching_iou,
            "count_matching": "COCOeval class-aware, one-to-one, crowd-aware",
        },
    }
