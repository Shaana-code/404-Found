"""Task 2.3 - Background embedding worker.

The upload endpoint stores the file, creates a job, schedules run_embedding_job
via FastAPI BackgroundTasks and returns immediately (HTTP 202). Clients poll
get_job(job_id) for status. Swap the in-memory store for Redis/DB if you run
multiple workers.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from .vectorizer import generate_image_embedding, generate_text_embedding

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class EmbeddingJob:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    embedding: Optional[List[float]] = field(default=None, repr=False)
    error: Optional[str] = None


_jobs: Dict[str, EmbeddingJob] = {}
_jobs_lock = threading.Lock()

# Callback that persists the result (e.g. Module 1 DB insert): (job_id, vector) -> None
Sink = Callable[[str, List[float]], None]


def create_job() -> str:
    job = EmbeddingJob(job_id=str(uuid.uuid4()))
    with _jobs_lock:
        _jobs[job.job_id] = job
    return job.job_id


def get_job(job_id: str) -> Optional[EmbeddingJob]:
    return _jobs.get(job_id)


def run_embedding_job(
    job_id: str,
    *,
    image_bytes: Optional[bytes] = None,
    text: Optional[str] = None,
    sink: Optional[Sink] = None,
) -> None:
    """Sync function on purpose: BackgroundTasks runs it in a threadpool, so the
    CPU/GPU-heavy inference doesn't block the event loop."""
    job = _jobs[job_id]
    job.status = JobStatus.PROCESSING
    try:
        if image_bytes is not None:
            vec = generate_image_embedding(image_bytes)
        elif text is not None:
            vec = generate_text_embedding(text)
        else:
            raise ValueError("Provide image_bytes or text")
        job.embedding = vec
        if sink:
            sink(job_id, vec)
        job.status = JobStatus.DONE
    except Exception as exc:
        logger.exception("Embedding job %s failed", job_id)
        job.error = str(exc)
        job.status = JobStatus.FAILED
