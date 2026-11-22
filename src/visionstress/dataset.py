"""Official COCO val2017 subset download and validation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import urlopen

import cv2

# HTTPS path-style access to the official images.cocodataset.org S3 bucket.
# The official vanity hostname currently presents a mismatched TLS certificate.
COCO_BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"
# A small trusted mirror of COCO's instances_val2017.json, pinned by published SHA-256.
ANNOTATIONS_URL = "https://huggingface.co/datasets/pcuenq/coco2017-instances/resolve/main/instances_val2017.json"
ANNOTATIONS_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
IMAGE_URL = COCO_BUCKET + "/val2017/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    for attempt in range(4):
        try:
            with urlopen(url, timeout=45) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            temporary.replace(path)
            return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def setup_dataset(root: Path, image_count: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    annotations = root / "instances_subset.json"
    original_path = root / "instances_val2017.json"
    if not annotations.exists():
        if not original_path.exists():
            print("Downloading pinned COCO val2017 annotations (~20 MB)...", flush=True)
            download(ANNOTATIONS_URL, original_path)
        if sha256(original_path) != ANNOTATIONS_SHA256:
            raise ValueError("COCO annotation mirror failed SHA-256 verification")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        selected = sorted(original["images"], key=lambda item: item["id"])[:image_count]
        ids = {item["id"] for item in selected}
        subset = {
            "info": original.get("info", {}),
            "licenses": original.get("licenses", []),
            "images": selected,
            "annotations": [item for item in original["annotations"] if item["image_id"] in ids],
            "categories": original["categories"],
        }
        annotations.write_text(json.dumps(subset), encoding="utf-8")
        original_path.unlink()
    subset = json.loads(annotations.read_text(encoding="utf-8"))
    if len(subset["images"]) != image_count:
        raise ValueError(f"Dataset has {len(subset['images'])} images, expected {image_count}; remove it and rerun setup")
    image_dir = root / "images"
    image_dir.mkdir(exist_ok=True)
    missing = [item for item in subset["images"] if not (image_dir / item["file_name"]).is_file()]
    if missing:
        print(f"Downloading {len(missing)} official COCO val2017 images...", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(download, IMAGE_URL + item["file_name"], image_dir / item["file_name"]): item for item in missing}
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 16 == 0 or index == len(missing):
                    print(f"Images: {index}/{len(missing)}", flush=True)
    return validate_dataset(root, image_count)


def validate_dataset(root: Path, image_count: int) -> dict:
    annotations = root / "instances_subset.json"
    if not annotations.is_file():
        raise ValueError(f"Missing dataset annotations: {annotations}. Run 'visionstress setup'.")
    dataset = json.loads(annotations.read_text(encoding="utf-8"))
    images = dataset.get("images", [])
    if len(images) != image_count or len({item["id"] for item in images}) != len(images):
        raise ValueError(f"Expected {image_count} unique images")
    image_ids = {item["id"] for item in images}
    categories = dataset.get("categories", [])
    category_ids = {item["id"] for item in categories}
    if len(category_ids) != 80 or len(categories) != 80:
        raise ValueError("Expected 80 unique COCO categories")
    image_hashes = {}
    for item in images:
        path = root / "images" / item["file_name"]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing or unreadable image: {path}")
        height, width = image.shape[:2]
        if (height, width) != (item["height"], item["width"]):
            raise ValueError(f"Image dimensions disagree with annotations: {path}")
        image_hashes[item["file_name"]] = sha256(path)
    for item in dataset.get("annotations", []):
        if item["image_id"] not in image_ids or item["category_id"] not in category_ids:
            raise ValueError(f"Annotation {item['id']} has invalid image or category ID")
        image_info = next(image for image in images if image["id"] == item["image_id"])
        x, y, width, height = item["bbox"]
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_info["width"] + 1 or y + height > image_info["height"] + 1:
            raise ValueError(f"Annotation {item['id']} has invalid bounding box")
    return {
        "source": "COCO 2017 val images and instances_val2017 annotations (pinned mirror)",
        "annotation_url": ANNOTATIONS_URL,
        "source_annotations_sha256": ANNOTATIONS_SHA256,
        "image_url_prefix": IMAGE_URL,
        "selection": "first image_count images by sorted COCO image ID",
        "image_count": len(images),
        "annotation_count": len(dataset.get("annotations", [])),
        "annotations_sha256": sha256(annotations),
        "image_sha256": image_hashes,
    }
