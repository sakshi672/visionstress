"""Run real baseline and stressed inference, then COCOeval both sets."""

from datetime import datetime, timezone
from importlib import metadata
import hashlib
import platform
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import ultralytics

from visionstress.dataset import sha256
from visionstress.evaluation import evaluate
from visionstress.model import load_model, verify_model
from .stress import transform
from .failures import rank_failures
from .tables import Dataset, Job, RegisteredModel
from .yolo_data import coco_annotations, validate_yolo_dataset


INFERENCE = {"image_size": 640, "confidence": 0.001, "nms_iou": 0.7, "max_detections": 300}
EVALUATOR = {"detection_confidence": 0.25, "matching_iou": 0.5, "max_detections": 100}


def metric_change(before: float | None, after: float | None) -> dict:
    return {"absolute": after - before if before is not None and after is not None else None, "relative_percent": 100 * (after - before) / before if before not in (None, 0) and after is not None else None}


class JobCancelled(Exception):
    pass


def _predict(model, image, image_id: int, mapping: dict[int, int], device) -> tuple[list[dict], float]:
    result = model.predict(source=image, imgsz=INFERENCE["image_size"], conf=INFERENCE["confidence"], iou=INFERENCE["nms_iou"], max_det=INFERENCE["max_detections"], device=device, verbose=False)[0]
    predictions = []
    for xyxy, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(float, xyxy)
        predictions.append({"image_id": image_id, "category_id": mapping[int(class_id)], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    return predictions, float(result.speed["inference"])


def _annotate(image, annotations: list[dict], predictions: list[dict]):
    image = image.copy()
    for item in annotations:
        x, y, w, h = map(round, item["bbox"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (70, 230, 120), 2)
    for item in predictions:
        if item["score"] < EVALUATOR["detection_confidence"]:
            continue
        x, y, w, h = map(round, item["bbox"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (70, 90, 255), 2)
        cv2.putText(image, f"{item['category_id']} {item['score']:.2f}", (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 90, 255), 1)
    return image


def execute_job(session_factory, store, job_id: int, timeout_seconds: int = 1800) -> dict:
    with session_factory() as db:
        job = db.get(Job, job_id)
        if not job or job.status != "running":
            raise ValueError("Job is not running")
        dataset = db.get(Dataset, job.dataset_id)
        registered = db.get(RegisteredModel, job.model_id)
        spec = dict(job.spec)
        dataset_root = Path(dataset.root)
        manifest = dataset.manifest
        model_path = Path(registered.path)
        model_digest = registered.digest
    actual = validate_yolo_dataset(dataset_root, dataset_root.parent)
    if actual["hash"] != manifest["hash"]:
        raise ValueError("Dataset changed since registration")
    trusted = verify_model(model_path)
    if trusted["sha256"] != model_digest:
        raise ValueError("Model changed since registration")
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(manifest["classes"])]
    model, mapping = load_model(model_path, categories)
    device = 0 if torch.cuda.is_available() else "cpu"
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    if "image_ids" in spec:
        selected = spec["image_ids"]
        if not selected or len(selected) > 500 or len(set(selected)) != len(selected) or not set(selected) <= set(manifest["split"]["validation"]):
            raise ValueError("Suite image IDs must be distinct validation images")
        if spec.get("inference") != INFERENCE or spec.get("evaluation") != EVALUATOR:
            raise ValueError("Suite evaluation settings differ from the current runner")
    else:
        selected = manifest["split"]["validation"][:spec["max_images"]]
    if not selected:
        raise ValueError("No validation images available")
    image_info = {item["id"]: item for item in manifest["images"]}
    ground_truth = coco_annotations(manifest, selected)
    annotation_path = store.write_json(job_id, "ground_truth.json", ground_truth)
    baseline, stressed, parameters, latencies = [], [], [], {"baseline": [], "stressed": []}
    started = time.monotonic()
    for index, image_id in enumerate(selected, 1):
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError("Evaluation exceeded configured timeout")
        with session_factory() as db:
            current = db.get(Job, job_id)
            if current.status == "cancel_requested":
                raise JobCancelled()
        item = image_info[image_id]
        path = dataset_root / "images" / item["file_name"]
        if sha256(path) != item["image_sha256"]:
            raise ValueError("Dataset image changed during evaluation")
        image = cv2.imread(str(path))
        stressed_image, params = transform(image, spec["stress"], spec["severity"], spec["seed"] + image_id)
        params["image_id"] = image_id
        parameters.append(params)
        if index <= 3:
            cv2.imwrite(str(store.job_dir(job_id) / f"original-{image_id}.jpg"), image)
            cv2.imwrite(str(store.job_dir(job_id) / f"stressed-{image_id}.jpg"), stressed_image)
        for name, source, predictions in (("baseline", image, baseline), ("stressed", stressed_image, stressed)):
            found, latency = _predict(model, source, image_id, mapping, device)
            predictions.extend(found)
            latencies[name].append(latency)
        with session_factory.begin() as db:
            current = db.get(Job, job_id)
            current.progress = round(index / len(selected) * 100)
    baseline_metrics = evaluate(annotation_path, baseline, **EVALUATOR)
    stressed_metrics = evaluate(annotation_path, stressed, **EVALUATOR)
    for name, metrics in (("baseline", baseline_metrics), ("stressed", stressed_metrics)):
        metrics["average_inference_ms"] = float(np.mean(latencies[name]))
        metrics["image_count"] = len(selected)
        metrics["prediction_count"] = len(baseline if name == "baseline" else stressed)
    degradation = {}
    for key in ("precision", "recall", "map50", "map50_95"):
        degradation[key] = metric_change(baseline_metrics[key], stressed_metrics[key])
    store.write_json(job_id, "baseline_predictions.json", baseline)
    store.write_json(job_id, "stressed_predictions.json", stressed)
    store.write_json(job_id, "transform_parameters.json", parameters)
    source_paths = [Path(__file__), Path(__file__).with_name("stress.py"), Path(__file__).with_name("failures.py"), Path(__file__).resolve().parents[1] / "evaluation.py"]
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_paths)).hexdigest()
    store.write_json(job_id, "environment.json", {"python": platform.python_version(), "visionstress": metadata.version("visionstress"), "code_sha256": source_hash, "torch": torch.__version__, "ultralytics": ultralytics.__version__, "opencv": cv2.__version__, "numpy": np.__version__, "pycocotools": metadata.version("pycocotools"), "device": str(device), "seed": spec["seed"], "model_sha256": model_digest, "dataset_sha256": manifest["hash"], "inference": INFERENCE, "evaluation": EVALUATOR, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    ranked = rank_failures(ground_truth["annotations"], baseline, stressed, selected)
    for example in ranked[:3]:
        image_id = example["image_id"]
        image = cv2.imread(str(dataset_root / "images" / image_info[image_id]["file_name"]))
        stressed_image, _ = transform(image, spec["stress"], spec["severity"], spec["seed"] + image_id)
        gt = [item for item in ground_truth["annotations"] if item["image_id"] == image_id]
        before = [item for item in baseline if item["image_id"] == image_id]
        after = [item for item in stressed if item["image_id"] == image_id]
        example["original_artifact"] = f"comparison-{image_id}-original.jpg"
        example["stressed_artifact"] = f"comparison-{image_id}-stressed.jpg"
        cv2.imwrite(str(store.job_dir(job_id) / example["original_artifact"]), _annotate(image, gt, before))
        cv2.imwrite(str(store.job_dir(job_id) / example["stressed_artifact"]), _annotate(stressed_image, gt, after))
    store.write_json(job_id, "failure_examples.json", ranked)
    warnings = []
    if len(selected) < 30:
        warnings.append("Fewer than 30 images: metrics are development diagnostics, not robust performance estimates")
    if spec["stress"] == "partial_occlusion":
        warnings.append("Occlusion may invalidate ground-truth labels; treat this run as exploratory")
    result = {"baseline": baseline_metrics, "stressed": stressed_metrics, "degradation": degradation, "stress": spec["stress"], "severity": spec["severity"], "image_count": len(selected), "label_policy": parameters[0]["label_policy"], "warnings": warnings, "failure_examples": ranked[:20], "artifacts": ["ground_truth.json", "baseline_predictions.json", "stressed_predictions.json", "transform_parameters.json", "environment.json", "failure_examples.json"]}
    store.write_json(job_id, "result.json", result)
    return result
