"""Freeze failure-case evidence and compare a candidate on identical inputs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from visionstress.evaluation import evaluate

from .runner import EVALUATOR, INFERENCE

GATE_METRICS = ("map50_95", "map50", "precision", "recall")


def _artifact(store, job_id: int, name: str):
    return json.loads(store.get(job_id, name).read_text(encoding="utf-8"))


def build_snapshot(store, source_job, dataset, model, image_ids: list[int], metric: str, max_drop: float) -> dict:
    if source_job.status != "completed" or not source_job.result:
        raise ValueError("Source evaluation must be completed")
    if metric not in GATE_METRICS or not 0 <= max_drop <= 1:
        raise ValueError("Invalid gate metric or maximum drop")
    if not image_ids or len(image_ids) > 500 or len(set(image_ids)) != len(image_ids):
        raise ValueError("Select between 1 and 500 distinct image IDs")
    ranked = {item["image_id"] for item in source_job.result.get("failure_examples", [])}
    validation = set(dataset.manifest["split"]["validation"])
    if not set(image_ids) <= ranked & validation:
        raise ValueError("Selected IDs must be failure cases from this source run's validation split")

    ground_truth = _artifact(store, source_job.id, "ground_truth.json")
    baseline = _artifact(store, source_job.id, "baseline_predictions.json")
    stressed = _artifact(store, source_job.id, "stressed_predictions.json")
    environment = _artifact(store, source_job.id, "environment.json")
    if environment["dataset_sha256"] != dataset.digest or environment["model_sha256"] != model.digest:
        raise ValueError("Source evidence no longer matches the registered inputs")
    if environment["inference"] != INFERENCE or environment["evaluation"] != EVALUATOR:
        raise ValueError("Source evaluation settings differ from the current runner")
    chosen = set(image_ids)
    subset = dict(ground_truth)
    subset["images"] = [item for item in ground_truth["images"] if item["id"] in chosen]
    subset["annotations"] = [item for item in ground_truth["annotations"] if item["image_id"] in chosen]
    if {item["id"] for item in subset["images"]} != chosen:
        raise ValueError("Source ground truth is missing a selected image")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "ground_truth.json"
        path.write_text(json.dumps(subset), encoding="utf-8")
        reference_baseline = evaluate(path, [item for item in baseline if item["image_id"] in chosen], **EVALUATOR)
        reference_stressed = evaluate(path, [item for item in stressed if item["image_id"] in chosen], **EVALUATOR)
    return {
        "image_ids": image_ids,
        "dataset_sha256": dataset.digest,
        "source_model_id": model.id,
        "source_model_sha256": model.digest,
        "source_job_id": source_job.id,
        "stress": source_job.spec["stress"],
        "severity": source_job.spec["severity"],
        "seed": source_job.spec["seed"],
        "inference": environment["inference"],
        "evaluation": environment["evaluation"],
        "reference": {"baseline": reference_baseline, "stressed": reference_stressed},
        "gate": {"metric": metric, "max_absolute_drop": max_drop},
        "label_policy": source_job.result.get("label_policy"),
    }


def compare_suite(snapshot: dict, candidate_result: dict) -> dict:
    metric = snapshot["gate"]["metric"]
    reference = snapshot["reference"]["stressed"][metric]
    candidate = candidate_result["stressed"][metric]
    max_drop = snapshot["gate"]["max_absolute_drop"]
    delta = candidate - reference if reference is not None and candidate is not None else None
    if snapshot["stress"] == "partial_occlusion" or delta is None:
        status = "inconclusive"
    else:
        status = "passed" if delta >= -max_drop - 1e-12 else "failed"
    return {
        "status": status,
        "metric": metric,
        "reference": reference,
        "candidate": candidate,
        "absolute_change": delta,
        "max_absolute_drop": max_drop,
        "image_count": len(snapshot["image_ids"]),
        "warnings": candidate_result.get("warnings", []),
    }
