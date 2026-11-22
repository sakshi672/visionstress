"""Exercise a real regression suite against the running local Compose stack."""

import argparse
import json
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", type=int, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    client = httpx.Client(base_url=args.api, timeout=30)
    source_response = client.get(f"/api/jobs/{args.source_job}")
    source_response.raise_for_status()
    source = source_response.json()
    if source["status"] != "completed":
        raise SystemExit("Source job must be completed")
    image_ids = [item["image_id"] for item in source["result"]["failure_examples"] if item["score"] > 0][:2]
    if not image_ids:
        raise SystemExit("Source job has no ranked failures")
    body = {"project_id": source["spec"]["project_id"], "source_job_id": source["id"], "name": f"Smoke suite {time.time_ns()}", "image_ids": image_ids, "metric": "map50_95", "max_absolute_drop": 0.02}
    response = client.post("/api/suites", json=body)
    response.raise_for_status()
    suite = response.json()
    response = client.post(f"/api/suites/{suite['id']}/runs", json={"model_id": source["spec"]["model_id"]})
    response.raise_for_status()
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(3)
    else:
        raise SystemExit("Suite run timed out")
    response = client.get(f"/api/suites/{suite['id']}")
    response.raise_for_status()
    run = response.json()["runs"][0]
    original_parameters = client.get(f"/api/jobs/{source['id']}/artifacts/transform_parameters.json")
    original_parameters.raise_for_status()
    candidate_parameters = client.get(f"/api/jobs/{job_id}/artifacts/transform_parameters.json")
    candidate_parameters.raise_for_status()
    expected = {item["image_id"]: item for item in original_parameters.json() if item["image_id"] in image_ids}
    observed = {item["image_id"]: item for item in candidate_parameters.json()}
    same_parameters = expected == observed and set(expected) == set(image_ids)
    comparison = run["comparison"]
    same_accuracy = comparison is not None and comparison["absolute_change"] is not None and abs(comparison["absolute_change"]) <= 1e-6
    print(json.dumps({"suite_id": suite["id"], "source_job_id": source["id"], "job_id": job_id, "image_ids": image_ids, "status": job["status"], "comparison": comparison, "same_transform_parameters": same_parameters, "same_accuracy_within_1e-6": same_accuracy, "error": job.get("error")}, indent=2))
    if job["status"] != "completed" or comparison is None or comparison["status"] != "passed" or not same_parameters or not same_accuracy:
        raise SystemExit("Smoke regression gate did not pass")


if __name__ == "__main__":
    main()
