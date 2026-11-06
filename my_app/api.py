"""
FastAPI app for fire & smoke detection on the VisionAI framework.

Run:  uvicorn my_app.api:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""

# Import torch first so it loads its own cuDNN (CUDA 12). If onnxruntime's preload runs first,
# its CUDA 13 cuDNN DLLs (same file names) get picked up by torch and fail with WinError 127.
import torch  # noqa: F401

# Then load onnxruntime's CUDA/cuDNN DLLs on Windows, before any framework or Ultralytics imports.
# Without it, inference falls back to the CPU without any error.
import onnxruntime as ort  # noqa: E402

ort.preload_dlls()

import logging  # noqa: E402
import tempfile  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402

from visionai_framework.config import ConfigLoader  # noqa: E402
from visionai_framework.ingestion import IngestionRequest  # noqa: E402
from visionai_framework.observability import get_metrics  # noqa: E402
from visionai_framework.plugins import PluginManager  # noqa: E402
from visionai_framework.runtime import FrameworkBootstrap, SyncRuntime  # noqa: E402

from my_app.detectors.fire_smoke import FireSmokePlugin  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("my_app.api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "vizai.yaml"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

# Map framework error codes to HTTP status codes
ERROR_STATUS = {
    "pipeline_not_found": 404,          # no routing rule for this site
    "concurrency_limit_exceeded": 429,  # too many requests running at once
    "execution_timeout": 504,           # exceeded sync.request.timeout_ms
}

runtime: SyncRuntime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the framework once at startup and shut it down on exit."""
    global runtime
    config = ConfigLoader.load(CONFIG_PATH)

    pm = PluginManager()
    pm.register_plugin(FireSmokePlugin())

    bootstrap = FrameworkBootstrap.initialize(config=config, plugin_manager=pm)
    runtime = bootstrap.create_sync_runtime()
    logger.info("VisionAI ready. pipelines=%s", bootstrap.registry.list_pipelines())
    logger.info("ONNX Runtime providers: %s", ort.get_available_providers())
    yield
    runtime.close()
    runtime = None


app = FastAPI(title="Fire & Smoke Detection", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "runtime_ready": runtime is not None}


@app.get("/metrics")
def metrics() -> dict[str, int]:
    """In-process framework counters (jobs total / success / failure, timings)."""
    return get_metrics().snapshot()


@app.post("/analyze/{site}/{camera}")
async def analyze(site: str, camera: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Run detection on one uploaded image.

    - site   → context_id (selects pipelines via routing.rules)
    - camera → source_id
    """
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not ready")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # DecodeImageStage reads from a file path, so save the upload to a temp file
    suffix = Path(file.filename or "frame.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    now = datetime.now(timezone.utc).isoformat()
    request = IngestionRequest(
        source_uri=str(tmp_path),
        source_type="file",
        modality="image",
        context_id=site,
        source_id=camera,
        start_timestamp=now,
        end_timestamp=now,
        content_type=file.content_type or "",
        attributes={"filename": file.filename},
    )

    try:
        # The runtime blocks while it works, so run it off the event loop
        result = await run_in_threadpool(runtime.submit, request)
    finally:
        tmp_path.unlink(missing_ok=True)

    if result is None:  # only when sync.output.mode == "emit"
        return {"status": "accepted"}

    body = asdict(result)
    if result.status == "failure" and result.error is not None:
        status_code = ERROR_STATUS.get(result.error.code, 500)
        logger.warning("analysis failed code=%s message=%s", result.error.code, result.error.message)
        raise HTTPException(status_code=status_code, detail=body)

    return body
