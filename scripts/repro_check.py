"""Repeat the first saved PostgreSQL demo job and compare deterministic outputs."""

import json
from pathlib import Path
import time

import httpx


def main():
    repo = Path(__file__).resolve().parents[1]
    previous = json.loads((repo / "outputs" / "demo_report.json").read_text(encoding="utf-8"))[0]
    with httpx.Client(base_url="http://localhost:8000", timeout=30) as client:
        response = client.post("/api/evaluations", json=previous["spec"])
        response.raise_for_status()
        job_id = response.json()["id"]
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            response = client.get(f"/api/jobs/{job_id}")
            response.raise_for_status()
            current = response.json()
            if current["status"] == "completed":
                break
            if current["status"] in ("failed", "cancelled"):
                raise RuntimeError(f"Repeat job failed: {current['error']}")
            time.sleep(2)
        else:
            raise TimeoutError("Repeat job timed out")
        for scenario in ("baseline", "stressed"):
            for key in ("map50", "map50_95", "precision", "recall"):
                before, after = previous["result"][scenario][key], current["result"][scenario][key]
                if before is None or after is None:
                    assert before == after, (scenario, key, before, after)
                else:
                    assert abs(before - after) <= 1e-6, (scenario, key, before, after)
        for name in ("transform_parameters.json", "ground_truth.json"):
            before = client.get(f"/api/jobs/{previous['id']}/artifacts/{name}")
            after = client.get(f"/api/jobs/{job_id}/artifacts/{name}")
            before.raise_for_status()
            after.raise_for_status()
            assert before.content == after.content, name
        print(f"Repeat job {job_id} matched job {previous['id']} within 1e-6 for accuracy; ground truth and transform parameters matched byte-for-byte")


if __name__ == "__main__":
    main()
