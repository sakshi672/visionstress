"""Simple class-aware image-level failure summary at a fixed confidence/IoU."""

from collections import defaultdict


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def summarize_image(ground_truth: list[dict], predictions: list[dict], confidence: float = 0.25, threshold: float = 0.5) -> dict:
    detections = sorted((p for p in predictions if p["score"] >= confidence), key=lambda p: p["score"], reverse=True)
    matched = set()
    false_positive = localization = 0
    for prediction in detections:
        candidates = [(iou(prediction["bbox"], target["bbox"]), index) for index, target in enumerate(ground_truth) if target["category_id"] == prediction["category_id"] and index not in matched]
        best, index = max(candidates, default=(0, -1))
        if best >= threshold:
            matched.add(index)
        else:
            false_positive += 1
            if 0.1 <= best < threshold:
                localization += 1
    return {"true_positives": len(matched), "false_positives": false_positive, "false_negatives": len(ground_truth) - len(matched), "localization_failures": localization}


def rank_failures(ground_truth: list[dict], baseline: list[dict], stressed: list[dict], image_ids: list[int] | None = None) -> list[dict]:
    gt, before, after = defaultdict(list), defaultdict(list), defaultdict(list)
    for item in ground_truth:
        gt[item["image_id"]].append(item)
    for item in baseline:
        before[item["image_id"]].append(item)
    for item in stressed:
        after[item["image_id"]].append(item)
    ranked = []
    for image_id in sorted(set(image_ids or []) | set(gt) | set(before) | set(after)):
        b = summarize_image(gt[image_id], before[image_id])
        a = summarize_image(gt[image_id], after[image_id])
        score = 3 * max(0, a["false_negatives"] - b["false_negatives"]) + 2 * max(0, a["false_positives"] - b["false_positives"]) + max(0, a["localization_failures"] - b["localization_failures"])
        ranked.append({"image_id": image_id, "score": score, "baseline": b, "stressed": a})
    return sorted(ranked, key=lambda item: (-item["score"], item["image_id"]))
