"""Convert a user-downloaded COCO development subset to YOLO format locally."""

import json
from pathlib import Path
import shutil

from visionstress.dataset import setup_dataset
from visionstress.model import setup_model
from .yolo_data import validate_yolo_dataset


def prepare_demo(repo: Path | None = None, count: int = 8) -> dict:
    repo = repo or Path(__file__).resolve().parents[3]
    source = repo / "data" / "coco_val2017_128"
    setup_dataset(source, 128)
    setup_model(repo / "models" / "yolo11n.pt")
    target = repo / "data" / f"demo_coco{count}"
    image_dir, label_dir = target / "images", target / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    coco = json.loads((source / "instances_subset.json").read_text(encoding="utf-8"))
    categories = sorted(coco["categories"], key=lambda category: category["id"])
    category_to_class = {category["id"]: index for index, category in enumerate(categories)}
    (target / "classes.json").write_text(json.dumps([category["name"] for category in categories], indent=2), encoding="utf-8")
    selected = sorted(coco["images"], key=lambda item: item["id"])[:count]
    for image in selected:
        name = image["file_name"]
        shutil.copy2(source / "images" / name, image_dir / name)
        lines = []
        for annotation in coco["annotations"]:
            if annotation["image_id"] != image["id"] or annotation.get("iscrowd", 0):
                continue
            x, y, w, h = annotation["bbox"]
            width, height = image["width"], image["height"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"{category_to_class[annotation['category_id']]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        (label_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return validate_yolo_dataset(target, repo / "data")


if __name__ == "__main__":
    summary = prepare_demo()
    print(f"Prepared YOLO demo: {summary['image_count']} images, {summary['box_count']} boxes, SHA-256 {summary['hash']}")
