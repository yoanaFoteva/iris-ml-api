"""
Script to train and save the ML model.
Uses the Iris dataset with a RandomForestClassifier.
"""

import pickle
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

MODEL_PATH = "model.pkl"
METADATA_PATH = "model_metadata.pkl"


def train_and_save():
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

    # Save metadata
    metadata = {
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
        "n_features": X.shape[1],
        "feature_mins": X.min(axis=0).tolist(),
        "feature_maxs": X.max(axis=0).tolist(),
        "accuracy": float(acc),
    }
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {METADATA_PATH}")
    print("Done! Metadata:", metadata)


if __name__ == "__main__":
    train_and_save()
