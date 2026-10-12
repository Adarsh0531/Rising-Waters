"""Flask interface for the trained flood-probability model."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts"


def create_app(artifact_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    model_dir = artifact_dir or DEFAULT_ARTIFACT_DIR
    model_path = model_dir / "flood_model.joblib"
    metadata_path = model_dir / "metadata.json"
    model = None
    metadata: dict[str, Any] | None = None

    if model_path.is_file() and metadata_path.is_file():
        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            ready=model is not None and metadata is not None,
            metadata=metadata,
            prediction=None,
            error=None,
            values={},
        )

    @app.post("/predict")
    def predict() -> tuple[str, int] | str:
        if model is None or metadata is None:
            return (
                render_template(
                    "index.html",
                    ready=False,
                    metadata=None,
                    prediction=None,
                    error="Train the model before requesting a prediction.",
                    values=request.form,
                ),
                503,
            )

        values: dict[str, Any] = {}
        try:
            for feature in metadata["features"]:
                raw_value = request.form.get(feature, "").strip()
                if not raw_value:
                    raise ValueError(f"Enter a value for {feature}.")
                if metadata["feature_types"][feature] == "number":
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise ValueError(f"{feature} must be a finite number.")
                else:
                    value = raw_value
                    allowed_values = metadata["feature_choices"].get(feature)
                    if allowed_values and value not in allowed_values:
                        raise ValueError(f"Choose a valid value for {feature}.")
                values[feature] = value
        except (TypeError, ValueError) as exc:
            return render_template(
                "index.html",
                ready=True,
                metadata=metadata,
                prediction=None,
                error=str(exc),
                values=request.form,
            )

        row = pd.DataFrame([values], columns=metadata["features"])
        if metadata["task_type"] == "classification":
            encoded_prediction = int(model.predict(row)[0])
            predicted_class = metadata["target_classes"][encoded_prediction]
            probabilities = model.predict_proba(row)[0]
            positive_index = metadata["positive_class_index"]
            positive_probability = float(probabilities[positive_index])
            prediction = {
                "kind": "classification",
                "label": predicted_class,
                "positive_class": metadata["target_classes"][positive_index],
                "probability": positive_probability,
                "risk": "Flood indicated"
                if encoded_prediction == positive_index
                else "No flood indicated",
            }
        else:
            probability = float(model.predict(row)[0])
            probability = min(1.0, max(0.0, probability))
            risk = (
                "High"
                if probability >= 0.6
                else "Elevated"
                if probability >= 0.4
                else "Low"
            )
            prediction = {
                "kind": "regression",
                "probability": probability,
                "risk": risk,
            }
        return render_template(
            "index.html",
            ready=True,
            metadata=metadata,
            prediction=prediction,
            error=None,
            values=values,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
