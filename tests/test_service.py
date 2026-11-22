import json
from datetime import timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionstress.service.api import create_app
from visionstress.service.db import Base, make_engine, make_session_factory
from visionstress.service.stress import transform
from visionstress.service.yolo_data import validate_yolo_dataset
from visionstress.service.report import render_report
from visionstress.service.failures import rank_failures, summarize_image
from visionstress.service.tables import Job, Project, Dataset, RegisteredModel, utcnow
from visionstress.service.storage import LocalArtifactStore
from visionstress.service.worker import process_next, recover_stale_jobs
from visionstress.service.runner import metric_change
from visionstress.service.runner import EVALUATOR, INFERENCE
from visionstress.service.suites import compare_suite
from visionstress.service.yolo_data import coco_annotations


def dataset_fixture(tmp_path):
    root = tmp_path / "data" / "tiny"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.json").write_text('["object"]', encoding="utf-8")
    for index in range(3):
        image = np.full((40, 50, 3), 40 + index * 20, dtype=np.uint8)
        cv2.imwrite(str(root / "images" / f"{index}.jpg"), image)
        (root / "labels" / f"{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    return root


def test_dataset_validation_and_split(tmp_path):
    root = dataset_fixture(tmp_path)
    manifest = validate_yolo_dataset(root, tmp_path / "data")
    assert manifest["image_count"] == 3
    assert manifest["box_count"] == 3
    assert set(manifest["split"]["validation"]).isdisjoint(manifest["split"]["test"])
    assert len(manifest["hash"]) == 64
    assert validate_yolo_dataset(root, tmp_path / "data")["hash"] == manifest["hash"]


@pytest.mark.parametrize("bad_label", ["0 1.1 0.5 0.2 0.2", "1 0.5 0.5 0.2 0.2", "0 0.1 0.1 0.5 0.5", "0 0.5 0.5"])
def test_dataset_rejects_bad_labels(tmp_path, bad_label):
    root = dataset_fixture(tmp_path)
    (root / "labels" / "0.txt").write_text(bad_label, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_yolo_dataset(root, tmp_path / "data")


def test_dataset_rejects_duplicate_images(tmp_path):
    root = dataset_fixture(tmp_path)
    (root / "images" / "1.jpg").write_bytes((root / "images" / "0.jpg").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_yolo_dataset(root, tmp_path / "data")


def test_stress_is_deterministic_and_geometry_preserving():
    image = np.arange(120 * 100 * 3, dtype=np.uint8).reshape(120, 100, 3)
    for name in ("brightness", "brightness_contrast", "gaussian_blur", "motion_blur", "gaussian_noise", "jpeg", "fog", "rain", "partial_occlusion", "color_shift", "defocus"):
        first, settings = transform(image, name, 3, 42)
        second, again = transform(image, name, 3, 42)
        assert first.shape == image.shape
        assert np.array_equal(first, second)
        assert settings == again
        if name == "partial_occlusion":
            assert "exploratory" in settings["label_policy"]
    with pytest.raises(ValueError):
        transform(image, "brightness", 6, 42)


def test_api_project_dataset_and_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSTRESS_TESTING", "1")
    root = dataset_fixture(tmp_path)
    engine = make_engine(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    Base.metadata.create_all(engine)
    client = TestClient(create_app(make_session_factory(engine), tmp_path / "data", tmp_path / "models", tmp_path / "outputs"))
    assert client.get("/api/health").json() == {"status": "ok"}
    project = client.post("/api/projects", json={"name": "Test", "description": ""})
    assert project.status_code == 201
    project_id = project.json()["id"]
    assert client.post("/api/projects", json={"name": "Test"}).status_code == 409
    dataset = client.post("/api/datasets", json={"project_id": project_id, "name": "Tiny", "directory": root.name})
    assert dataset.status_code == 201, dataset.text
    assert dataset.json()["manifest"]["image_count"] == 3
    assert client.get(f"/api/datasets/{dataset.json()['id']}/samples/1").status_code == 200
    assert client.get(f"/api/datasets/{dataset.json()['id']}/samples/99").status_code == 404
    assert client.post("/api/datasets", json={"project_id": project_id, "name": "Outside", "directory": "../outside"}).status_code == 422
    assert client.post("/api/models", json={"project_id": project_id, "checkpoint": "untrusted.pt"}).status_code == 422
    assert client.get("/api/jobs/999").status_code == 404


def test_report_requires_measured_result(tmp_path):
    output = tmp_path / "report.html"
    with pytest.raises(ValueError):
        render_report([{"status": "queued", "result": None}], output)
    run = {"status": "completed", "spec": {"stress": "brightness", "severity": 2}, "result": {"image_count": 3, "baseline": {"map50_95": 0.5}, "stressed": {"map50_95": 0.4, "average_inference_ms": 12}, "degradation": {"map50_95": {"absolute": -0.1}}}}
    render_report([run], output)
    assert "0.4000" in output.read_text(encoding="utf-8")
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))[0]["status"] == "completed"


def test_failure_ranking_detects_lost_detection():
    gt = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}]
    correct = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    missed = [{"image_id": 1, "category_id": 1, "bbox": [60, 60, 20, 20], "score": 0.9}]
    assert summarize_image(gt, correct)["true_positives"] == 1
    ranked = rank_failures(gt, correct, missed, [1])
    assert ranked[0]["score"] == 5
    assert ranked[0]["stressed"]["false_negatives"] == 1


def test_degradation_handles_zero_and_undefined_baselines():
    assert metric_change(0.5, 0.4)["relative_percent"] == pytest.approx(-20)
    assert metric_change(0, 0.2)["absolute"] == pytest.approx(0.2)
    assert metric_change(0, 0.2)["relative_percent"] is None
    assert metric_change(None, 0.2)["absolute"] is None


def test_alembic_initial_migration(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect
    monkeypatch.setenv("VISIONSTRESS_TESTING", "1")
    url = f"sqlite:///{tmp_path / 'migrated.sqlite3'}"
    monkeypatch.setenv("VISIONSTRESS_DATABASE_URL", url)
    config = Config(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "head")
    engine = make_engine(url)
    assert {"projects", "datasets", "registered_models", "jobs", "regression_suites", "suite_runs"}.issubset(set(inspect(engine).get_table_names()))
    engine.dispose()


def test_suite_uses_exact_cases_and_persists_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSTRESS_TESTING", "1")
    root = dataset_fixture(tmp_path)
    manifest = validate_yolo_dataset(root, tmp_path / "data")
    engine = make_engine(f"sqlite:///{tmp_path / 'suite.sqlite3'}")
    Base.metadata.create_all(engine)
    sessions = make_session_factory(engine)
    store = LocalArtifactStore(tmp_path / "outputs")
    image_id = manifest["split"]["validation"][0]
    with sessions.begin() as db:
        db.add(Project(id=1, name="Suite project", description=""))
        db.add(Dataset(id=1, project_id=1, name="Tiny", root=str(root), digest=manifest["hash"], manifest=manifest))
        db.add(RegisteredModel(id=1, project_id=1, name="Approved", path="unused", digest="modelhash", provenance="test", classes=manifest["classes"]))
        db.add(Job(id=1, project_id=1, dataset_id=1, model_id=1, status="completed", progress=100, attempt=1, spec={"stress": "brightness", "severity": 2, "seed": 42}, result={"failure_examples": [{"image_id": image_id, "score": 3}], "label_policy": "preserved"}))
    store.write_json(1, "ground_truth.json", coco_annotations(manifest, manifest["split"]["validation"]))
    store.write_json(1, "baseline_predictions.json", [])
    store.write_json(1, "stressed_predictions.json", [])
    store.write_json(1, "environment.json", {"dataset_sha256": manifest["hash"], "model_sha256": "modelhash", "inference": INFERENCE, "evaluation": EVALUATOR})
    client = TestClient(create_app(sessions, tmp_path / "data", tmp_path / "models", tmp_path / "outputs"))
    body = {"project_id": 1, "source_job_id": 1, "name": "Misses", "image_ids": [image_id], "metric": "map50_95", "max_absolute_drop": 0.02}
    assert client.post("/api/suites", json={**body, "image_ids": [999]}).status_code == 422
    assert client.post("/api/suites", json={**body, "image_ids": [image_id, image_id]}).status_code == 422
    created = client.post("/api/suites", json=body)
    assert created.status_code == 201, created.text
    suite = created.json()
    assert suite["snapshot"]["image_ids"] == [image_id]
    assert suite["snapshot"]["reference"]["stressed"]["map50_95"] == 0
    assert client.post("/api/suites", json=body).status_code == 409
    queued = client.post(f"/api/suites/{suite['id']}/runs", json={"model_id": 1})
    assert queued.status_code == 202
    job_id = queued.json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["spec"]["image_ids"] == [image_id]
    monkeypatch.setattr("visionstress.service.worker.execute_job", lambda session_factory, store, job_id: {"stressed": {"map50_95": 0.0}, "warnings": []})
    assert process_next(sessions, store) == job_id
    completed = client.get(f"/api/suites/{suite['id']}").json()["runs"][0]
    assert completed["status"] == "completed"
    assert completed["comparison"]["status"] == "passed"
    assert completed["comparison"]["image_count"] == 1
    engine.dispose()


def test_suite_gate_handles_failed_and_inconclusive_metrics():
    snapshot = {"stress": "brightness", "image_ids": [1, 2], "gate": {"metric": "map50_95", "max_absolute_drop": 0.02}, "reference": {"stressed": {"map50_95": 0.5}}}
    assert compare_suite(snapshot, {"stressed": {"map50_95": 0.48}})["status"] == "passed"
    assert compare_suite(snapshot, {"stressed": {"map50_95": 0.47}})["status"] == "failed"
    assert compare_suite(snapshot, {"stressed": {"map50_95": None}})["status"] == "inconclusive"
    assert compare_suite({**snapshot, "stress": "partial_occlusion"}, {"stressed": {"map50_95": 0.6}})["status"] == "inconclusive"


def test_persistent_job_transitions(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSTRESS_TESTING", "1")
    engine = make_engine(f"sqlite:///{tmp_path / 'queue.sqlite3'}")
    Base.metadata.create_all(engine)
    sessions = make_session_factory(engine)
    with sessions.begin() as db:
        db.add(Project(id=1, name="Queue", description=""))
        db.add(Dataset(id=1, project_id=1, name="D", root="unused", digest="x", manifest={}))
        db.add(RegisteredModel(id=1, project_id=1, name="M", path="unused", digest="x", provenance="test", classes=[]))
        db.add(Job(id=1, project_id=1, dataset_id=1, model_id=1, status="queued", progress=0, attempt=0, spec={}))
    monkeypatch.setattr("visionstress.service.worker.execute_job", lambda session_factory, store, job_id: {"measured": 1})
    assert process_next(sessions, LocalArtifactStore(tmp_path / "outputs")) == 1
    with sessions() as db:
        job = db.get(Job, 1)
        assert (job.status, job.progress, job.attempt) == ("completed", 100, 1)
        assert job.result == {"measured": 1}
    with sessions.begin() as db:
        db.add(Job(id=2, project_id=1, dataset_id=1, model_id=1, status="running", progress=50, attempt=1, spec={}, started_at=utcnow() - timedelta(hours=1)))
    assert recover_stale_jobs(sessions) == 1
    with sessions() as db:
        assert db.get(Job, 2).status == "failed"
    engine.dispose()
