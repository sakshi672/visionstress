"""One-command local Docker/PostgreSQL demo with three measured stresses."""

import subprocess
import time
from pathlib import Path

import httpx

from visionstress.service.demo_data import prepare_demo
from visionstress.service.report import render_report


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def main():
    repo = Path(__file__).resolve().parents[1]
    prepared = prepare_demo(repo)
    print(f"Prepared {prepared['image_count']} local COCO images, {prepared['box_count']} boxes")
    try:
        healthy = httpx.get("http://localhost:8000/api/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        healthy = False
    if not healthy:
        subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=repo, check=True)
    with httpx.Client(base_url="http://localhost:8000", timeout=30) as client:
        for _ in range(90):
            try:
                if request(client, "GET", "/api/health")["status"] == "ok":
                    break
            except (httpx.HTTPError, KeyError):
                time.sleep(2)
        else:
            raise RuntimeError("API did not become healthy")
        project = request(client, "POST", "/api/projects", json={"name": f"Demo {int(time.time())}", "description": "Three-condition CPU demo"})
        dataset = request(client, "POST", "/api/datasets", json={"project_id": project["id"], "name": "COCO demo 8", "directory": "demo_coco8"})
        model = request(client, "POST", "/api/models", json={"project_id": project["id"], "checkpoint": "yolo11n.pt"})
        runs = []
        for stress in ("brightness", "gaussian_blur", "jpeg"):
            job = request(client, "POST", "/api/evaluations", json={"project_id": project["id"], "dataset_id": dataset["id"], "model_id": model["id"], "stress": stress, "severity": 3, "seed": 42, "max_images": 6})
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                current = request(client, "GET", f"/api/jobs/{job['id']}")
                if current["status"] == "completed":
                    runs.append(current)
                    print(f"{stress}: mAP50:95 {current['result']['stressed']['map50_95']:.4f}")
                    break
                if current["status"] in ("failed", "cancelled"):
                    raise RuntimeError(f"{stress} job {job['id']} {current['status']}: {current['error']}")
                time.sleep(2)
            else:
                raise TimeoutError(f"{stress} job timed out")
    report = render_report(runs, repo / "outputs" / "demo_report.html")
    print(f"Report: {report}")
    print("Dashboard: http://localhost:5173")


if __name__ == "__main__":
    main()
