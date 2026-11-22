"""REST API. Evaluation executes only in the separate worker process."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from visionstress.model import verify_model
from visionstress.dataset import sha256
from .db import make_engine, make_session_factory
from .storage import LocalArtifactStore
from .tables import Dataset, Job, Project, RegisteredModel
from .yolo_data import validate_yolo_dataset


class ProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)


class DatasetInput(BaseModel):
    project_id: int
    name: str = Field(min_length=1, max_length=120)
    directory: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ModelInput(BaseModel):
    project_id: int
    checkpoint: str = Field(pattern=r"^yolo11[nsm]\.pt$")


class EvaluationInput(BaseModel):
    project_id: int
    dataset_id: int
    model_id: int
    stress: str = Field(default="brightness", pattern=r"^(brightness|brightness_contrast|gaussian_blur|motion_blur|gaussian_noise|jpeg|fog|rain|partial_occlusion|color_shift|defocus)$")
    severity: int = Field(default=2, ge=1, le=5)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_images: int = Field(default=12, ge=1, le=500)


def record(obj, *names):
    return {name: getattr(obj, name) for name in names}


def create_app(session_factory=None, data_root: Path | None = None, model_root: Path | None = None, output_root: Path | None = None) -> FastAPI:
    repo = Path(__file__).resolve().parents[3]
    data_root = (data_root or repo / "data").resolve()
    model_root = (model_root or repo / "models").resolve()
    store = LocalArtifactStore(output_root or repo / "outputs" / "web")
    session_factory = session_factory or make_session_factory(make_engine())
    app = FastAPI(title="VisionStress API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        with session_factory() as db:
            db.execute(select(Project.id).limit(1))
        return {"status": "ok"}

    @app.get("/api/projects")
    def projects():
        with session_factory() as db:
            return [record(p, "id", "name", "description", "created_at") for p in db.scalars(select(Project).order_by(Project.id))]

    @app.post("/api/projects", status_code=201)
    def create_project(body: ProjectInput):
        with session_factory.begin() as db:
            if db.scalar(select(Project.id).where(Project.name == body.name)):
                raise HTTPException(409, "Project name already exists")
            item = Project(**body.model_dump())
            db.add(item)
            db.flush()
            return record(item, "id", "name", "description")

    @app.patch("/api/projects/{project_id}")
    def edit_project(project_id: int, body: ProjectInput):
        with session_factory.begin() as db:
            item = db.get(Project, project_id)
            if not item:
                raise HTTPException(404, "Project not found")
            existing = db.scalar(select(Project.id).where(Project.name == body.name, Project.id != project_id))
            if existing:
                raise HTTPException(409, "Project name already exists")
            item.name, item.description = body.name, body.description
            return record(item, "id", "name", "description")

    @app.delete("/api/projects/{project_id}", status_code=204)
    def delete_project(project_id: int):
        with session_factory.begin() as db:
            item = db.get(Project, project_id)
            if not item:
                raise HTTPException(404, "Project not found")
            if db.scalar(select(Job.id).where(Job.project_id == project_id).limit(1)):
                raise HTTPException(409, "Delete project jobs first")
            for child in db.scalars(select(Dataset).where(Dataset.project_id == project_id)):
                db.delete(child)
            for child in db.scalars(select(RegisteredModel).where(RegisteredModel.project_id == project_id)):
                db.delete(child)
            db.delete(item)

    @app.get("/api/datasets")
    def datasets(project_id: int):
        with session_factory() as db:
            return [record(d, "id", "project_id", "name", "digest", "manifest") for d in db.scalars(select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.id))]

    @app.post("/api/datasets", status_code=201)
    def create_dataset(body: DatasetInput):
        with session_factory.begin() as db:
            if not db.get(Project, body.project_id):
                raise HTTPException(404, "Project not found")
            root = data_root / body.directory
            try:
                manifest = validate_yolo_dataset(root, data_root)
            except (ValueError, OSError) as error:
                raise HTTPException(422, str(error)) from error
            item = Dataset(project_id=body.project_id, name=body.name, root=str(root.resolve()), digest=manifest["hash"], manifest=manifest)
            db.add(item)
            db.flush()
            return record(item, "id", "project_id", "name", "digest", "manifest")

    @app.get("/api/datasets/{dataset_id}/samples/{image_id}")
    def dataset_sample(dataset_id: int, image_id: int):
        with session_factory() as db:
            item = db.get(Dataset, dataset_id)
            if not item:
                raise HTTPException(404, "Dataset not found")
            image = next((image for image in item.manifest["images"] if image["id"] == image_id), None)
            if not image:
                raise HTTPException(404, "Image not in dataset")
            path = Path(item.root) / "images" / image["file_name"]
            if not path.is_file() or sha256(path) != image["image_sha256"]:
                raise HTTPException(409, "Dataset image changed since registration")
            return FileResponse(path)

    @app.get("/api/models")
    def models(project_id: int):
        with session_factory() as db:
            return [record(m, "id", "project_id", "name", "digest", "provenance", "classes") for m in db.scalars(select(RegisteredModel).where(RegisteredModel.project_id == project_id).order_by(RegisteredModel.id))]

    @app.post("/api/models", status_code=201)
    def register_model(body: ModelInput):
        with session_factory.begin() as db:
            if not db.get(Project, body.project_id):
                raise HTTPException(404, "Project not found")
            path = model_root / body.checkpoint
            try:
                manifest = verify_model(path)
            except (ValueError, OSError) as error:
                raise HTTPException(422, str(error)) from error
            from ultralytics import YOLO
            names = YOLO(str(path)).names
            classes = [names[i] for i in range(len(names))]
            item = RegisteredModel(project_id=body.project_id, name=body.checkpoint, path=str(path), digest=manifest["sha256"], provenance=manifest["source"], classes=classes)
            db.add(item)
            db.flush()
            return record(item, "id", "project_id", "name", "digest", "provenance", "classes")

    @app.post("/api/evaluations", status_code=202)
    def queue_evaluation(body: EvaluationInput):
        with session_factory.begin() as db:
            dataset = db.get(Dataset, body.dataset_id)
            model = db.get(RegisteredModel, body.model_id)
            if not dataset or not model or dataset.project_id != body.project_id or model.project_id != body.project_id:
                raise HTTPException(404, "Dataset or model not found in project")
            if dataset.manifest["classes"] != model.classes:
                raise HTTPException(422, "Dataset class names/order must match model classes")
            item = Job(project_id=body.project_id, dataset_id=body.dataset_id, model_id=body.model_id, spec=body.model_dump(), status="queued", progress=0)
            db.add(item)
            db.flush()
            return record(item, "id", "status", "progress", "spec")

    @app.get("/api/jobs")
    def jobs(project_id: int):
        with session_factory() as db:
            return [record(j, "id", "status", "progress", "attempt", "error", "spec", "created_at", "finished_at") for j in db.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.id.desc()))]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: int):
        with session_factory() as db:
            item = db.get(Job, job_id)
            if not item:
                raise HTTPException(404, "Job not found")
            return record(item, "id", "status", "progress", "attempt", "error", "spec", "result", "created_at", "started_at", "finished_at")

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: int):
        with session_factory.begin() as db:
            item = db.get(Job, job_id)
            if not item:
                raise HTTPException(404, "Job not found")
            if item.status in ("queued", "running"):
                item.status = "cancel_requested" if item.status == "running" else "cancelled"
            return record(item, "id", "status")

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: int):
        with session_factory.begin() as db:
            item = db.get(Job, job_id)
            if not item:
                raise HTTPException(404, "Job not found")
            if item.status != "failed" or item.attempt >= 3:
                raise HTTPException(409, "Only failed jobs with fewer than 3 attempts may be retried")
            item.status, item.error, item.progress = "queued", None, 0
            return record(item, "id", "status")

    @app.get("/api/jobs/{job_id}/artifacts/{name}")
    def artifact(job_id: int, name: str):
        with session_factory() as db:
            if not db.get(Job, job_id):
                raise HTTPException(404, "Job not found")
        try:
            path = store.get(job_id, name)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, "Artifact not found") from error
        return FileResponse(path)

    return app


app = create_app()
