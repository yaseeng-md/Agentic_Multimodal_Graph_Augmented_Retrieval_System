from __future__ import annotations

import asyncio
import shutil
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from qdrant_client import QdrantClient
from config import AppConfig, override_config
from ingest import ingest_documents, IngestManager, save_upload_sync, IngestJob
from utils import (
    load_model_and_processor,
    select_device,
    load_visual_config,
)

LOGGER = logging.getLogger("ingest_api")
UPLOAD_DIR = Path("./uploaded_pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    visual_cfg = load_visual_config()
    cfg = override_config(visual_cfg)

    # add these to your config if they are not there yet
    if not hasattr(cfg, "ingest_workers"):
        cfg.ingest_workers = 2
    if not hasattr(cfg, "ingest_queue_size"):
        cfg.ingest_queue_size = 100

    device, dtype = select_device()
    model, processor = load_model_and_processor(device, dtype)

    client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    manager = IngestManager(cfg, client, model, processor)

    app.state.cfg = cfg
    app.state.ingest_manager = manager

    await manager.start()
    try:
        yield
    finally:
        await manager.stop()


# Load FastAPI App
app = FastAPI(lifespan=lifespan)


@app.post("/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    entity_id: str | None = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")

    manager: IngestManager = app.state.ingest_manager
    jobs = []

    for upload in files:
        if not (upload.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{upload.filename} is not a PDF")

        saved_path = await asyncio.to_thread(save_upload_sync, upload)
        await upload.close()

        job_id = uuid4().hex
        await manager.enqueue(
            IngestJob(
                job_id=job_id,
                pdf_paths=[saved_path],   # one PDF = one job
                entity_id=entity_id,
            )
        )

        jobs.append(
            {
                "job_id": job_id,
                "filename": upload.filename,
                "status": "queued",
            }
        )

    return {
        "message": "Ingest job(s) accepted",
        "worker_count": manager.worker_count,
        "queued_jobs": jobs,
    }