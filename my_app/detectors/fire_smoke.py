"""
Fire & smoke detection plugin for the VisionAI framework.

Pipeline: decode image → YOLO (.onnx) inference → filter → format
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from visionai_framework.core.executor import PipelineExecutor
from visionai_framework.core.job import AnalysisJob
from visionai_framework.core.pipeline import BasePipeline, PipelineResult
from visionai_framework.core.stage import Stage, StageContext
from visionai_framework.plugins.manager import ConfigDrivenPlugin
from visionai_framework.stages import DecodeImageStage, FilterDetectionsStage, FormatDetectionsStage
from visionai_framework.vision.types import Detection
from visionai_framework.vision.yolo import YoloModel

# must match plugins.enabled / pipelines.definitions[].plugin
PLUGIN_NAME = "myapp.fire_smoke"

# Project root (demo_vizai_framework/), so relative model paths work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pipelines are created once per job, so keep model wrappers here to load each model once
_models: dict[str, YoloModel] = {}
_models_lock = threading.Lock()
# One Ultralytics model object is not safe for parallel predict() calls, so run them one at a time
_predict_lock = threading.Lock()


def _resolve_model_path(raw: str) -> str:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return str(path)


def _get_model(model_path: str, confidence: float) -> YoloModel:
    with _models_lock:
        model = _models.get(model_path)
        if model is None:
            model = YoloModel(model_path=model_path,
                              confidence_threshold=confidence)
            if not model.load():
                raise RuntimeError(
                    model.load_error or f"Failed to load model: {model_path}")
            _models[model_path] = model
        return model


class FireSmokeInferStage(Stage[Any, list[Detection]]):
    """Run the fire/smoke YOLO model on a decoded BGR image."""

    def __init__(self, params: dict[str, Any]) -> None:
        self._model_path = _resolve_model_path(
            str(params.get("model_path", "models/best_e2e.onnx")))
        self._default_conf = float(params.get("confidence_threshold", 0.25))

    @property
    def stage_id(self) -> str:
        return "fire_smoke_infer"

    def execute(self, input_data: Any, context: StageContext) -> list[Detection]:
        conf = float(context.params.get(
            "confidence_threshold", self._default_conf))
        model = _get_model(self._model_path, conf)
        with _predict_lock:
            detections = model.predict(input_data, conf=conf)
        context.metadata["raw_detection_count"] = len(detections)
        return detections


class FireSmokePipeline(BasePipeline):
    """decode → infer → filter → format."""

    def __init__(self, pipeline_id: str, params: dict[str, Any]) -> None:
        self._pipeline_id = pipeline_id
        self._params = params
        self._stages: list[Stage[Any, Any]] = [
            DecodeImageStage(),
            FireSmokeInferStage(params),
            FilterDetectionsStage(),   # reads confidence_threshold / allowed_labels from params
            FormatDetectionsStage(),   # always returns a PipelineResult, even with 0 detections
        ]

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    def run(self, job: AnalysisJob) -> PipelineResult:
        return PipelineExecutor.execute(
            self._stages,
            job,
            pipeline_id=self._pipeline_id,
            params=self._params,
        )


class FireSmokePlugin(ConfigDrivenPlugin):
    """Creates one FireSmokePipeline per config entry that names this plugin."""

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    def create_pipeline(self, pipeline_id: str, params: dict[str, Any]) -> BasePipeline:
        return FireSmokePipeline(pipeline_id, params)
