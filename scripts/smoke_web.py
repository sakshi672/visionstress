"""CPU integration smoke test using SQLite only as an automated test database."""

import os
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient

os.environ["VISIONSTRESS_TESTING"] = "1"

from visionstress.service.api import create_app
from visionstress.service.db import Base, make_engine, make_session_factory
from visionstress.service.demo_data import prepare_demo
from visionstress.service.storage import LocalArtifactStore
from visionstress.service.worker import process_next
from visionstress.service.report import render_report


def main():
    repo = Path(__file__).resolve().parents[1]
    prepare_demo(repo)
    with tempfile.TemporaryDirectory() as temporary:
        engine = make_engine(f"sqlite:///{Path(temporary) / 'smoke.sqlite3'}")
        Base.metadata.create_all(engine)
        sessions = make_session_factory(engine)
        store = LocalArtifactStore(repo / "outputs" / "web-smoke")
        client = TestClient(create_app(sessions, repo / "data", repo / "models", store.root))
        project = client.post("/api/projects", json={"name": "CPU smoke"}).json()
        dataset = client.post("/api/datasets", json={"project_id": project["id"], "name": "COCO demo", "directory": "demo_coco8"})
        assert dataset.status_code == 201, dataset.text
        model = client.post("/api/models", json={"project_id": project["id"], "checkpoint": "yolo11n.pt"})
        assert model.status_code == 201, model.text
        runs = []
        for stress in ("brightness", "gaussian_blur", "jpeg"):
            job = client.post("/api/evaluations", json={"project_id": project["id"], "dataset_id": dataset.json()["id"], "model_id": model.json()["id"], "stress": stress, "severity": 2, "seed": 42, "max_images": 6})
            assert job.status_code == 202, job.text
            job_id = job.json()["id"]
            assert process_next(sessions, store) == job_id
            result = client.get(f"/api/jobs/{job_id}").json()
            assert result["status"] == "completed", result["error"]
            assert client.get(f"/api/jobs/{job_id}/artifacts/result.json").status_code == 200
            runs.append(result)
            print(f"Completed real CPU job {job_id} ({stress}): baseline mAP50-95={result['result']['baseline']['map50_95']:.4f}, stressed={result['result']['stressed']['map50_95']:.4f}")
        print(f"Report: {render_report(runs, store.root / 'report.html')}")
        engine.dispose()


if __name__ == "__main__":
    main()
