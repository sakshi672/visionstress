"""Validate a bounded, local YOLO detection dataset and build COCO ground truth."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2

from visionstress.dataset import sha256

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGES = 500
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 30_000_000


def within(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def validate_yolo_dataset(root: Path, allowed_root: Path, test_fraction: float = 0.2) -> dict:
    root = root.resolve()
    allowed_root = allowed_root.resolve()
    if root == allowed_root or not root.is_relative_to(allowed_root):
        raise ValueError("Dataset must be a subdirectory of the configured data root")
    if not root.is_dir():
        raise ValueError("Dataset directory does not exist")
    classes_path = root / "classes.json"
    if not classes_path.is_file() or not within(classes_path, root):
        raise ValueError("Dataset requires classes.json")
    if classes_path.stat().st_size > 100_000:
        raise ValueError("classes.json exceeds size limit")
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not classes or len(classes) > 1000 or any(not isinstance(c, str) or not c.strip() for c in classes) or len(set(classes)) != len(classes):
        raise ValueError("classes.json must contain unique nonempty class names")
    image_dir, label_dir = root / "images", root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError("Dataset requires images/ and labels/ directories")
    images = sorted(p for p in image_dir.iterdir() if p.is_file())
    if not images or len(images) > MAX_IMAGES:
        raise ValueError(f"Dataset must contain 1–{MAX_IMAGES} images")
    if any(p.suffix.lower() not in EXTENSIONS or not within(p, root) for p in images):
        raise ValueError("Unsupported or unsafe image file")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Image stems must be unique")
    labels = sorted(p for p in label_dir.iterdir() if p.is_file())
    if any(p.suffix != ".txt" or not within(p, root) for p in labels) or {p.stem for p in labels} != {p.stem for p in images}:
        raise ValueError("Every image needs one matching .txt label file and no extras")
    entries, annotations, counts, content = [], [], Counter(), []
    for image_id, path in enumerate(images, 1):
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds size limit: {path.name}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path.name}")
        height, width = image.shape[:2]
        if width * height > MAX_PIXELS:
            raise ValueError(f"Image exceeds pixel limit: {path.name}")
        label_path = label_dir / f"{path.stem}.txt"
        if label_path.stat().st_size > 1024 * 1024:
            raise ValueError("Label file exceeds size limit")
        boxes = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid label at {label_path.name}:{line_number}")
            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError as error:
                raise ValueError(f"Invalid label at {label_path.name}:{line_number}") from error
            if class_id < 0 or class_id >= len(classes) or not all(0 <= v <= 1 for v in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0 or cx - bw / 2 < -1e-6 or cy - bh / 2 < -1e-6 or cx + bw / 2 > 1 + 1e-6 or cy + bh / 2 > 1 + 1e-6:
                raise ValueError(f"Out-of-range box at {label_path.name}:{line_number}")
            x, y, w, h = (cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height
            boxes.append({"category_id": class_id + 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            counts[classes[class_id]] += 1
        entries.append({"id": image_id, "file_name": path.name, "width": width, "height": height, "image_sha256": sha256(path), "label_sha256": sha256(label_path)})
        annotations.extend({"id": len(annotations) + index + 1, "image_id": image_id, **box} for index, box in enumerate(boxes))
        content.append(f"{path.name}:{sha256(path)}:{sha256(label_path)}")
    if len({item["image_sha256"] for item in entries}) != len(entries):
        raise ValueError("Duplicate image content would leak between splits")
    # Reassign annotation IDs after all images; the comprehension above is unique already.
    digest = hashlib.sha256((json.dumps(classes) + "\n" + "\n".join(content)).encode()).hexdigest()
    ordered = sorted(entries, key=lambda item: hashlib.sha256(f"{digest}:{item['image_sha256']}".encode()).hexdigest())
    test_count = max(1, round(len(ordered) * test_fraction)) if len(ordered) >= 2 else 0
    test_ids = {item["id"] for item in ordered[:test_count]}
    return {
        "format": "yolo-detection", "hash": digest, "classes": classes,
        "image_count": len(entries), "box_count": len(annotations),
        "class_distribution": dict(counts), "images": entries,
        "annotations": annotations,
        "split": {"test": sorted(test_ids), "validation": sorted({item['id'] for item in entries} - test_ids)},
        "split_method": "content-hash ordering, 20% held-out test; duplicate image hashes prohibited",
    }


def coco_annotations(manifest: dict, image_ids: list[int] | None = None) -> dict:
    selected = set(image_ids or [image["id"] for image in manifest["images"]])
    return {
        "info": {"description": "VisionStress YOLO conversion"},
        "images": [{k: item[k] for k in ("id", "file_name", "width", "height")} for item in manifest["images"] if item["id"] in selected],
        "annotations": [item for item in manifest["annotations"] if item["image_id"] in selected],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(manifest["classes"])],
    }
