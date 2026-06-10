# Iris Classifier — ML REST API

A REST API that uses a trained machine learning model to classify Iris flowers, packaged and run entirely through Docker.

---

## What the model does

The model is a **Random Forest Classifier** trained on the Iris dataset - 150 real flower measurements collected by botanist Ronald Fisher in 1936.

You give it four measurements of an Iris flower (in centimetres):
- sepal length
- sepal width
- petal length
- petal width

It tells you which of three species the flower belongs to: **setosa**, **versicolor**, or **virginica**, along with a confidence probability for each. The model reaches 100% accuracy on unseen data.

During `docker build`, the model is trained automatically and saved inside the image. When the container starts, the model is loaded into memory and ready to serve predictions.

---

## Project structure

```
    app.py               # REST API (all endpoints and validation)
    train_model.py       # Trains and saves the model
    requirements.txt     # Python dependencies
    Dockerfile           # Builds and packages everything
    pytest.ini           # Test config
    tests/
      __init__.py
      test_api.py      # 27 automated tests
```

---

## Requirements

- [Docker Desktop](https://docs.docker.com/get-docker/)

---

## Step 1 - Build the image

```bash
docker build -t iris-classifier .
```

Installs all dependencies and trains the model inside the image.

## Step 2 - Run the container

```bash
docker run -d --name iris-api -p 8000:8000 iris-classifier
```

---

## Trying the API

Open **http://localhost:8000/docs** in your browser. This is the interactive Swagger UI - you can call every endpoint directly from the browser without any extra tools.

To make a prediction, click **POST /predict → Try it out**, paste this into the request body and hit Execute:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Expected response:

```json
{
  "predicted_class_id": 0,
  "predicted_class_name": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

Second example - it should return virginica:

```json
{
  "sepal_length": 6.7,
  "sepal_width": 3.0,
  "petal_length": 5.2,
  "petal_width": 2.3
}
```

### Available endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Is the model loaded and ready? |
| GET | `/model/info` | Feature names, class names, accuracy |
| POST | `/predict` | Predict species for one flower |
| POST | `/predict/batch` | Predict species for up to 100 flowers |

---

## Running the tests

```bash
docker exec -it iris-api pytest tests/ -v
```

Expected result: **27 passed**.

The tests cover every endpoint, verify correct predictions for known samples, and check that invalid inputs are rejected with proper error codes.

---

## Stopping the container

```bash
docker stop iris-api && docker rm iris-api
```