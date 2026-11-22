import json
import hashlib

import pytest

from visionstress.config import load_config
from visionstress.dataset import validate_dataset
from visionstress.evaluation import evaluate
from visionstress.model import MODEL_URLS, class_mapping, setup_model, verify_model


def test_config_rejects_inference_cutoff_above_count_cutoff(tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        "dataset: {root: data, image_count: 1}\n"
        "model: {path: model.pt}\n"
        "inference: {confidence: 0.5}\n"
        "evaluation: {detection_confidence: 0.25, matching_iou: 0.5}\n"
        "output_root: outputs\nseed: 42\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not exceed"):
        load_config(path)


def test_rejects_annotation_outside_image(tmp_path):
    data = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 10, "height": 10}],
        "categories": [{"id": value, "name": str(value)} for value in range(80)],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 4, 4]}],
    }
    (tmp_path / "instances_subset.json").write_text(json.dumps(data), encoding="utf-8")
    import cv2
    import numpy as np

    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "test.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="invalid bounding box"):
        validate_dataset(tmp_path, 1)


def test_class_mapping_checks_names_and_noncontiguous_coco_ids():
    assert class_mapping({0: "person", 1: "bicycle"}, [{"id": 3, "name": "bicycle"}, {"id": 1, "name": "person"}]) == {0: 1, 1: 3}
    with pytest.raises(ValueError, match="Class mismatch"):
        class_mapping({0: "car"}, [{"id": 1, "name": "person"}])


def test_checkpoint_allowlist_and_manifest(tmp_path):
    with pytest.raises(ValueError, match="only supports"):
        setup_model(tmp_path / "unknown.pt")
    checkpoint = tmp_path / "yolo11s.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    checkpoint.with_suffix(".manifest.json").write_text(json.dumps({
        "name": "yolo11s.pt",
        "source": MODEL_URLS["yolo11s.pt"],
        "sha256": hashlib.sha256(b"not a checkpoint").hexdigest(),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 verification"):
        verify_model(checkpoint)


def test_coco_evaluator_perfect_detection(tmp_path):
    annotations = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations), encoding="utf-8")
    result = evaluate(path, [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "score": 0.9}], 0.25, 0.5, 100)
    assert result["map50"] == pytest.approx(1)
    assert result["map50_95"] == pytest.approx(1)
    assert result["precision"] == pytest.approx(1)
    assert result["recall"] == pytest.approx(1)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (1, 0, 0)


def test_coco_evaluator_reports_false_detection(tmp_path):
    annotations = {
        "info": {},
        "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "object"}, {"id": 2, "name": "absent_class"}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations), encoding="utf-8")
    result = evaluate(path, [
        {"image_id": 1, "category_id": 1, "bbox": [70, 70, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [40, 40, 10, 10], "score": 0.9},
    ], 0.25, 0.5, 100)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (0, 2, 1)
    assert result["precision"] == 0
    assert result["recall"] == 0


def test_coco_evaluator_handles_zero_predictions(tmp_path):
    annotations = {
        "info": {},
        "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "object"}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations), encoding="utf-8")
    result = evaluate(path, [], 0.25, 0.5, 100)
    assert result["false_negatives"] == 1
    assert result["true_positives"] == 0
    assert result["map50"] == 0
