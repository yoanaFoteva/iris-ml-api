"""
pytest test suite for the Iris Classifier API.

Run inside Docker:
    docker exec -it iris-api pytest tests/ -v
"""

import os
import sys
import pytest

# Make sure the app module is importable when tests run from /app
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient

# ensure model artefacts exist
if not (os.path.exists("model.pkl") and os.path.exists("model_metadata.pkl")):
    import train_model
    train_model.train_and_save()

from app import app, ml_model, load_artifacts  # noqa: E402

# Manually load the model - lifespan not supported in this Starlette version
load_artifacts()

client = TestClient(app)


# Fixtures
@pytest.fixture
def valid_setosa():
    return {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


@pytest.fixture
def valid_virginica():
    return {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}


# System endpoints
class TestRoot:
    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_docs_hint(self):
        body = client.get("/").json()
        assert "docs" in body


class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_loaded(self):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"

    def test_health_has_accuracy(self):
        data = client.get("/health").json()
        assert isinstance(data["model_accuracy"], float)
        assert 0.0 <= data["model_accuracy"] <= 1.0


class TestModelInfo:
    def test_info_200(self):
        assert client.get("/model/info").status_code == 200

    def test_info_feature_names(self):
        data = client.get("/model/info").json()
        assert len(data["feature_names"]) == 4

    def test_info_target_names(self):
        data = client.get("/model/info").json()
        assert set(data["target_names"]) == {"setosa", "versicolor", "virginica"}

    def test_info_n_features(self):
        data = client.get("/model/info").json()
        assert data["n_features"] == 4


# Single prediction
class TestPredict:
    def test_predict_200(self, valid_setosa):
        r = client.post("/predict", json=valid_setosa)
        assert r.status_code == 200

    def test_predict_returns_class_name(self, valid_setosa):
        data = client.post("/predict", json=valid_setosa).json()
        assert data["predicted_class_name"] in {"setosa", "versicolor", "virginica"}

    def test_predict_returns_class_id(self, valid_setosa):
        data = client.post("/predict", json=valid_setosa).json()
        assert data["predicted_class_id"] in {0, 1, 2}

    def test_predict_probabilities_sum_to_one(self, valid_setosa):
        data = client.post("/predict", json=valid_setosa).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_predict_setosa_sample(self, valid_setosa):
        data = client.post("/predict", json=valid_setosa).json()
        assert data["predicted_class_name"] == "setosa"

    def test_predict_virginica_sample(self, valid_virginica):
        data = client.post("/predict", json=valid_virginica).json()
        assert data["predicted_class_name"] == "virginica"

    # Validation errors
    def test_missing_field_returns_422(self):
        r = client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5})
        assert r.status_code == 422

    def test_negative_value_returns_422(self):
        r = client.post("/predict", json={
            "sepal_length": -1.0, "sepal_width": 3.5,
            "petal_length": 1.4,  "petal_width": 0.2,
        })
        assert r.status_code == 422

    def test_zero_value_returns_422(self):
        r = client.post("/predict", json={
            "sepal_length": 0.0, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        assert r.status_code == 422

    def test_value_too_large_returns_422(self):
        r = client.post("/predict", json={
            "sepal_length": 999.0, "sepal_width": 3.5,
            "petal_length": 1.4,   "petal_width": 0.2,
        })
        assert r.status_code == 422

    def test_string_value_returns_422(self):
        r = client.post("/predict", json={
            "sepal_length": "big", "sepal_width": 3.5,
            "petal_length": 1.4,   "petal_width": 0.2,
        })
        assert r.status_code == 422

    def test_empty_body_returns_422(self):
        r = client.post("/predict", json={})
        assert r.status_code == 422


# Batch prediction
class TestPredictBatch:
    def test_batch_single_sample(self, valid_setosa):
        r = client.post("/predict/batch", json={"samples": [valid_setosa]})
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_batch_multiple_samples(self, valid_setosa, valid_virginica):
        r = client.post("/predict/batch", json={"samples": [valid_setosa, valid_virginica]})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_order_preserved(self, valid_setosa, valid_virginica):
        data = client.post("/predict/batch", json={
            "samples": [valid_setosa, valid_virginica]
        }).json()
        assert data["predictions"][0]["predicted_class_name"] == "setosa"
        assert data["predictions"][1]["predicted_class_name"] == "virginica"

    def test_batch_empty_list_returns_422(self):
        r = client.post("/predict/batch", json={"samples": []})
        assert r.status_code == 422

    def test_batch_missing_samples_key_returns_422(self):
        r = client.post("/predict/batch", json={})
        assert r.status_code == 422

    def test_batch_invalid_sample_returns_422(self, valid_setosa):
        bad = {"sepal_length": -5.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        r = client.post("/predict/batch", json={"samples": [valid_setosa, bad]})
        assert r.status_code == 422