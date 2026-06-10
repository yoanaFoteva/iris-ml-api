"""
ML REST API - Iris Flower Classifier
Provides endpoints for prediction, model info, batch prediction, and health checks.
"""

import pickle
import os
import time
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
METADATA_PATH = os.getenv("METADATA_PATH", "model_metadata.pkl")

# Global state
ml_model = {}


def load_artifacts():
    """Load model and metadata from disk. Raises RuntimeError if not found."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at '{MODEL_PATH}'. Run train_model.py first.")
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError(f"Metadata file not found at '{METADATA_PATH}'. Run train_model.py first.")

    with open(MODEL_PATH, "rb") as f:
        ml_model["model"] = pickle.load(f)
    with open(METADATA_PATH, "rb") as f:
        ml_model["metadata"] = pickle.load(f)

    logger.info("Model loaded. Accuracy on test set: %.4f", ml_model["metadata"]["accuracy"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    try:
        load_artifacts()
        logger.info("Startup complete — model is ready.")
    except RuntimeError as e:
        logger.error("Startup failed: %s", e)
        raise
    yield
    ml_model.clear()
    logger.info("Shutdown — model unloaded.")


# App
app = FastAPI(
    title="Iris Classifier API",
    description=(
        "REST API for the Iris flower classification model.\n\n"
        "The model predicts one of three species — **setosa**, **versicolor**, or **virginica** — "
        "from four numeric measurements."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Schemas
class IrisFeatures(BaseModel):
    """Four measurements (in cm) of an Iris flower."""

    sepal_length: float = Field(..., gt=0, le=30, description="Sepal length in cm", json_schema_extra={"example": 5.1})
    sepal_width:  float = Field(..., gt=0, le=30, description="Sepal width in cm",  json_schema_extra={"example": 3.5})
    petal_length: float = Field(..., gt=0, le=30, description="Petal length in cm", json_schema_extra={"example": 1.4})
    petal_width:  float = Field(..., gt=0, le=30, description="Petal width in cm",  json_schema_extra={"example": 0.2})

    @model_validator(mode="after")
    def petals_not_bigger_than_sepals(self):
        if self.petal_length > self.sepal_length * 2:
            raise ValueError(
                "petal_length is unrealistically large relative to sepal_length"
            )
        return self


class BatchIrisFeatures(BaseModel):
    """A batch of Iris flower measurements (1–100 samples)."""

    samples: List[IrisFeatures] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
    probabilities: dict


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_accuracy: Optional[float] = None


class ModelInfoResponse(BaseModel):
    feature_names: List[str]
    target_names: List[str]
    n_features: int
    model_accuracy: float


# Helpers
def _predict_single(features: IrisFeatures) -> PredictionResponse:
    model    = ml_model["model"]
    metadata = ml_model["metadata"]

    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    class_id   = int(model.predict(X)[0])
    class_name = metadata["target_names"][class_id]
    proba      = model.predict_proba(X)[0]
    proba_dict = {
        metadata["target_names"][i]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    return PredictionResponse(
        predicted_class_id=class_id,
        predicted_class_name=class_name,
        probabilities=proba_dict,
    )


# Exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check the server logs."},
    )


# Routes
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns the health status and whether the model is loaded."""
    loaded = "model" in ml_model
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_accuracy=ml_model["metadata"]["accuracy"] if loaded else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Returns metadata about the loaded model (features, classes, accuracy)."""
    if "metadata" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    m = ml_model["metadata"]
    return ModelInfoResponse(
        feature_names=m["feature_names"],
        target_names=m["target_names"],
        n_features=m["n_features"],
        model_accuracy=m["accuracy"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: IrisFeatures):
    """
    Predict the Iris species for a **single** sample.

    Provide four measurements in centimetres. Returns the predicted class
    name and the probability for each class.
    """
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    try:
        result = _predict_single(features)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    logger.info("Predicted: %s", result.predicted_class_name)
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchIrisFeatures):
    """
    Predict Iris species for a **batch** of samples (1–100).

    Returns a list of predictions in the same order as the input.
    """
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        predictions = [_predict_single(s) for s in batch.samples]
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    logger.info("Batch predicted %d samples", len(predictions))
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.get("/", tags=["System"])
def root():
    """Redirect hint — visit /docs for the interactive API documentation."""
    return {
        "message": "Iris Classifier API is running.",
        "docs": "/docs",
        "health": "/health",
    }
