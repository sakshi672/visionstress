"""Trusted YOLO11 checkpoint management."""

import json
from pathlib import Path

from .dataset import download, sha256

MODEL_URLS = {
    name: f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"
    for name in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")
}
EXPECTED_SHA256 = {
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo11s.pt": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    "yolo11m.pt": "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
}


def setup_model(path: Path) -> dict:
    if path.name not in MODEL_URLS:
        raise ValueError(f"Automatic setup only supports: {', '.join(MODEL_URLS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".manifest.json")
    if path.is_file() and not manifest_path.is_file():
        raise ValueError("Existing checkpoint has no trusted manifest; remove it before running setup")
    if not path.is_file():
        print(f"Downloading official Ultralytics {path.name} checkpoint...", flush=True)
        download(MODEL_URLS[path.name], path)
    digest = sha256(path)
    if digest != EXPECTED_SHA256[path.name]:
        raise ValueError("Checkpoint failed pinned SHA-256 verification")
    manifest = {"name": path.name, "source": MODEL_URLS[path.name], "sha256": digest}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Checkpoint differs from saved manifest; delete it and rerun trusted setup")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_model(path: Path) -> dict:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("Missing checkpoint or trusted manifest. Run 'visionstress setup'.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if path.name not in MODEL_URLS or manifest.get("source") != MODEL_URLS[path.name] or manifest.get("name") != path.name or manifest.get("sha256") != EXPECTED_SHA256[path.name] or sha256(path) != EXPECTED_SHA256[path.name]:
        raise ValueError("Checkpoint failed trusted source and SHA-256 verification")
    return manifest


def load_model(path: Path, categories: list[dict]):
    verify_model(path)  # Verify before torch deserializes the checkpoint.
    from ultralytics import YOLO

    model = YOLO(str(path))
    return model, class_mapping(model.names, categories)


def class_mapping(names: dict[int, str], categories: list[dict]) -> dict[int, int]:
    ordered = sorted(categories, key=lambda item: item["id"])
    if len(names) != len(ordered):
        raise ValueError("Model and dataset have different class counts")
    mapping = {}
    for model_id, category in enumerate(ordered):
        if names[model_id].strip().lower() != category["name"].strip().lower():
            raise ValueError(f"Class mismatch at model ID {model_id}: {names[model_id]} != {category['name']}")
        mapping[model_id] = category["id"]
    return mapping
