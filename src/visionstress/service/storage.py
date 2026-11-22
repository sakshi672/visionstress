"""Local artifact store with a narrow interface for future S3 replacement."""

import json
from pathlib import Path


class LocalArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def job_dir(self, job_id: int) -> Path:
        path = self.root / f"job-{job_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, job_id: int, name: str, data) -> Path:
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Invalid artifact name")
        path = self.job_dir(job_id) / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def get(self, job_id: int, name: str) -> Path:
        if Path(name).name != name:
            raise ValueError("Invalid artifact name")
        path = self.job_dir(job_id) / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return path
