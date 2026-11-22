"""Milestone 1 command-line entry points."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import uuid

import cv2
import numpy as np

from .config import load_config
from .dataset import setup_dataset, validate_dataset
from .evaluation import evaluate
from .model import load_model, setup_model, verify_model


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def dataset_categories(root: Path) -> list[dict]:
    return json.loads((root / "instances_subset.json").read_text(encoding="utf-8"))["categories"]


def validate(config: dict, load_checkpoint: bool = True) -> tuple[dict, dict]:
    dataset_root = Path(config["dataset"]["root"])
    dataset = validate_dataset(dataset_root, config["dataset"]["image_count"])
    checkpoint = verify_model(Path(config["model"]["path"]))
    if load_checkpoint:
        _, mapping = load_model(Path(config["model"]["path"]), dataset_categories(dataset_root))
        print(f"Validated {dataset['image_count']} images, {dataset['annotation_count']} annotations, and {len(mapping)} mapped classes.")
    return dataset, checkpoint


def run_baseline(config: dict, config_path: Path) -> Path:
    import torch
    import ultralytics

    dataset_manifest, model_manifest = validate(config, load_checkpoint=False)
    root = Path(config["dataset"]["root"])
    annotation_path = root / "instances_subset.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    model, mapping = load_model(Path(config["model"]["path"]), data["categories"])
    device = config["inference"]["device"]
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(config["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.yaml")
    shutil.copy2(annotation_path, run_dir / "ground_truth.json")
    save_json(run_dir / "dataset_manifest.json", dataset_manifest)
    save_json(run_dir / "model_manifest.json", model_manifest)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    predictions = []
    latencies = []
    images = sorted(data["images"], key=lambda item: item["id"])
    for index, item in enumerate(images, 1):
        image_path = root / "images" / item["file_name"]
        image = cv2.imread(str(image_path))
        result = model.predict(
            source=image,
            imgsz=config["inference"]["image_size"],
            conf=config["inference"]["confidence"],
            iou=config["inference"]["nms_iou"],
            max_det=config["inference"]["max_detections"],
            device=device,
            verbose=False,
        )[0]
        latencies.append(float(result.speed["inference"]))
        boxes = result.boxes
        for xyxy, confidence, class_id in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(float, xyxy)
            predictions.append({
                "image_id": item["id"],
                "category_id": mapping[int(class_id)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(confidence),
            })
        if index % 16 == 0 or index == len(images):
            print(f"Baseline inference: {index}/{len(images)} images", flush=True)
    save_json(run_dir / "predictions.json", predictions)
    metrics = evaluate(
        annotation_path,
        predictions,
        config["evaluation"]["detection_confidence"],
        config["evaluation"]["matching_iou"],
        config["evaluation"]["max_detections"],
    )
    metrics["average_inference_ms"] = float(np.mean(latencies))
    metrics["image_count"] = len(images)
    metrics["prediction_count"] = len(predictions)
    save_json(run_dir / "baseline_metrics.json", metrics)
    save_json(run_dir / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pycocotools": metadata.version("pycocotools"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "seed": config["seed"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_settings": config["inference"],
    })
    print(f"Baseline results saved to {run_dir}")
    print(f"mAP@0.5: {metrics['map50']:.4f}; mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VisionStress object detection benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "validate", "baseline"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/benchmark.yaml")
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.command == "setup":
            setup_model(Path(config["model"]["path"]))
            summary = setup_dataset(Path(config["dataset"]["root"]), config["dataset"]["image_count"])
            print(f"Prepared {summary['image_count']} COCO val2017 images.")
        elif arguments.command == "validate":
            validate(config)
        elif arguments.command == "baseline":
            run_baseline(config, Path(arguments.config).resolve())
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"VisionStress: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
