"""Single-concurrency persistent database queue worker."""

from datetime import timedelta
import logging
import time

from sqlalchemy import select

from .db import make_engine, make_session_factory
from .runner import JobCancelled, execute_job
from .storage import LocalArtifactStore
from .tables import Job, RegressionSuite, SuiteRun, utcnow
from .suites import compare_suite

log = logging.getLogger(__name__)


def process_next(session_factory, store: LocalArtifactStore) -> int | None:
    with session_factory.begin() as db:
        job = db.scalar(select(Job).where(Job.status == "queued").order_by(Job.id).with_for_update(skip_locked=True).limit(1))
        if job is None:
            return None
        job.status, job.started_at, job.attempt = "running", utcnow(), job.attempt + 1
        job_id = job.id
    try:
        result = execute_job(session_factory, store, job_id)
    except JobCancelled:
        with session_factory.begin() as db:
            item = db.get(Job, job_id)
            item.status, item.finished_at = "cancelled", utcnow()
    except Exception as error:
        log.exception("Job %s failed", job_id)
        with session_factory.begin() as db:
            item = db.get(Job, job_id)
            item.status, item.error, item.finished_at = "failed", str(error)[:2000], utcnow()
    else:
        with session_factory.begin() as db:
            item = db.get(Job, job_id)
            item.status, item.result, item.progress, item.finished_at = "completed", result, 100, utcnow()
            suite_run = db.scalar(select(SuiteRun).where(SuiteRun.job_id == job_id))
            if suite_run:
                suite = db.get(RegressionSuite, suite_run.suite_id)
                suite_run.comparison = compare_suite(suite.snapshot, result)
    return job_id


def recover_stale_jobs(session_factory, max_age_seconds: int = 1900) -> int:
    cutoff = utcnow() - timedelta(seconds=max_age_seconds)
    count = 0
    with session_factory.begin() as db:
        jobs = db.scalars(select(Job).where(Job.status.in_(["running", "cancel_requested"]), Job.started_at < cutoff).with_for_update(skip_locked=True)).all()
        for item in jobs:
            item.status = "cancelled" if item.status == "cancel_requested" else "failed"
            item.error = "Worker stopped or timed out before completion" if item.status == "failed" else None
            item.finished_at = utcnow()
            count += 1
    return count


def main():
    logging.basicConfig(level=logging.INFO)
    import os
    from pathlib import Path
    engine = make_engine()
    session_factory = make_session_factory(engine)
    store = LocalArtifactStore(Path(os.getenv("VISIONSTRESS_OUTPUT_ROOT", "outputs/web")))
    recovered = recover_stale_jobs(session_factory)
    if recovered:
        log.warning("Recovered %s stale jobs", recovered)
    while True:
        if process_next(session_factory, store) is None:
            time.sleep(2)


if __name__ == "__main__":
    main()
