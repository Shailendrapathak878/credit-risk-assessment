"""FastAPI ML inference prototype for credit-risk assessment."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import ApplicantRequest, ExplanationResponse, PredictionResponse
from app.config import METADATA_PATH, MODEL_PATH, MODEL_VERSION, STATIC_PATH, THRESHOLD
from app.services.explanation_service import ExplanationService
from app.services.model_service import ModelService, ModelUnavailableError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service = ModelService(
        MODEL_PATH,
        METADATA_PATH,
    )
    try:
        service.load()
        app.state.model_service = service
        app.state.explanation_service = ExplanationService(service)
        LOGGER.info("Credit Risk Assessment API started")
    except ModelUnavailableError:
        app.state.model_service = service
        app.state.explanation_service = None
        LOGGER.error("API started without a usable model")
    yield


app = FastAPI(
    title="Credit Risk Assessment API",
    version=MODEL_VERSION,
    description=(
        "Development/demo ML inference prototype. Probabilities are uncalibrated, "
        "the evaluation threshold is 0.50, and SHAP values use raw-margin/log-odds space. "
        "Feature availability requires business validation."
    ),
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


@app.middleware("http")
async def request_tracking(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_PATH / "index.html")


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "model_service", None)
    loaded = bool(service and service.loaded)
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "model_name": "tuned_xgboost_all_valid",
        "model_version": MODEL_VERSION,
    }


def model_info(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "model_service", None)
    metadata = service.metadata if service else {}
    return {
        "model_name": metadata.get("model_name", "Tuned XGBoost"),
        "model_version": MODEL_VERSION,
        "feature_set": metadata.get("feature_set", "all_valid_features"),
        "threshold": THRESHOLD,
        "calibration_method": metadata.get("calibration_method", "uncalibrated"),
        "explainability": "SHAP TreeExplainer",
        "explanation_space": "raw_margin_log_odds",
        "model_loaded": bool(service and service.loaded),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ApplicantRequest, http_request: Request) -> PredictionResponse:
    service = getattr(http_request.app.state, "model_service", None)
    if service is None or not service.loaded:
        raise HTTPException(status_code=503, detail="Model service unavailable")
    try:
        result = service.predict(request, THRESHOLD)
        LOGGER.info("Prediction request completed")
        return PredictionResponse(**result)
    except Exception as exc:
        LOGGER.exception("Prediction request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: ApplicantRequest, http_request: Request) -> ExplanationResponse:
    service = getattr(http_request.app.state, "explanation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Explanation service unavailable")
    try:
        result = service.explain(request, THRESHOLD)
        LOGGER.info("Explanation request completed")
        return ExplanationResponse(**result)
    except Exception as exc:
        LOGGER.exception("Explanation request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Explanation failed") from exc


app.add_api_route("/api/v1/health", health, methods=["GET"], response_model=dict)
app.add_api_route("/api/v1/model-info", model_info, methods=["GET"])
app.add_api_route("/api/v1/predict", predict, methods=["POST"], response_model=PredictionResponse)
app.add_api_route("/api/v1/explain", explain, methods=["POST"], response_model=ExplanationResponse)
