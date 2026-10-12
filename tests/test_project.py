import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from app import create_app
from train import build_models, load_dataset, train


class FixedModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def fit(self, rows, targets):
        return self

    def predict(self, rows):
        return [self.prediction] * len(rows)

    def predict_proba(self, rows):
        return [[0.25, 0.75]] * len(rows)


class ProjectTests(unittest.TestCase):
    def test_load_dataset_drops_id_and_filters_invalid_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flood.csv"
            pd.DataFrame(
                {
                    "id": list(range(22)),
                    "MonsoonIntensity": list(range(22)),
                    "FloodProbability": [0.4, 1.2] + [0.7] * 20,
                }
            ).to_csv(path, index=False)

            features, target, task_type = load_dataset(path)

        self.assertEqual(list(features.columns), ["MonsoonIntensity"])
        self.assertEqual(len(target), 21)
        self.assertEqual(target.iloc[0], 0.4)
        self.assertEqual(target.iloc[1], 0.7)
        self.assertEqual(task_type, "regression")

    def test_load_dataset_supports_kerala_flood_labels_and_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kerala.csv"
            pd.DataFrame(
                {
                    " subdivision ": ["KERALA ", "TAMIL", "KERALA", "TAMIL"] * 6,
                    "annual rainfall": list(range(24)),
                    " floods ": ["YES", "NO"] * 12,
                }
            ).to_csv(path, index=False)

            features, target, task_type = load_dataset(path)

        self.assertEqual(task_type, "classification")
        self.assertEqual(list(features.columns), ["subdivision", "annual rainfall"])
        self.assertEqual(target.iloc[0], "YES")
        self.assertEqual(target.iloc[1], "NO")

    def test_load_dataset_requires_supported_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unlabelled.csv"
            pd.DataFrame({"rainfall": [1, 2]}).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "FLOODS"):
                load_dataset(path)

    def test_requested_regressors_fit_numeric_data(self):
        features = pd.DataFrame(
            {
                "MonsoonIntensity": list(range(30)),
                "RiverManagement": [value % 7 for value in range(30)],
            }
        )
        target = pd.Series([0.2 + value / 100 for value in range(30)])

        for name, model in build_models("regression", features).items():
            with self.subTest(model=name):
                model.fit(features, target)
                self.assertEqual(len(model.predict(features.iloc[:2])), 2)

    def test_requested_classifiers_fit_numeric_and_categorical_data(self):
        features = pd.DataFrame(
            {
                "rainfall": list(range(30)),
                "season": ["wet", "dry"] * 15,
            }
        )
        target = pd.Series([0, 1] * 15)

        for name, model in build_models("classification", features).items():
            with self.subTest(model=name):
                model.fit(features, target)
                self.assertEqual(len(model.predict(features.iloc[:2])), 2)

    def test_prediction_page_uses_classes_and_displays_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            joblib.dump(FixedModel(1), artifact_dir / "flood_model.joblib")
            (artifact_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "model_name": "Test model",
                        "task_type": "classification",
                        "target_classes": ["NO", "YES"],
                        "positive_class_index": 1,
                        "features": ["rainfall", "season"],
                        "feature_defaults": {
                            "rainfall": 5,
                            "season": "wet",
                        },
                        "feature_types": {"rainfall": "number", "season": "category"},
                        "feature_choices": {"season": ["dry", "wet"]},
                        "holdout_metrics": {
                            "accuracy": 0.9,
                            "balanced_accuracy": 0.9,
                            "f1": 0.9,
                        },
                    }
                ),
                encoding="utf-8",
            )
            client = create_app(artifact_dir).test_client()
            response = client.post(
                "/predict",
                data={"rainfall": "8", "season": "wet"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"75.0", response.data)
        self.assertIn(b"Flood indicated", response.data)
        self.assertIn(b"YES", response.data)

    def test_prediction_rejects_non_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            joblib.dump(FixedModel(0.25), artifact_dir / "flood_model.joblib")
            (artifact_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "model_name": "Test model",
                        "task_type": "regression",
                        "features": ["rainfall"],
                        "feature_defaults": {"rainfall": 5},
                        "feature_types": {"rainfall": "number"},
                        "feature_choices": {},
                        "holdout_metrics": {"r2": 0.9, "mae": 0.02, "rmse": 0.03},
                    }
                ),
                encoding="utf-8",
            )
            client = create_app(artifact_dir).test_client()
            response = client.post("/predict", data={"rainfall": "nan"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"must be a finite number", response.data)

    def test_training_saves_selected_classifier_and_holdout_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "kerala.csv"
            pd.DataFrame(
                {
                    "YEAR": list(range(60)),
                    "rainfall": list(range(60)),
                    "season": ["wet", "dry"] * 30,
                    "FLOODS": ["YES", "NO"] * 30,
                }
            ).to_csv(data_path, index=False)
            candidates = {
                "Decision Tree": FixedModel(1),
                "Random Forest": FixedModel(1),
                "K-Nearest Neighbours": FixedModel(1),
                "XGBoost": FixedModel(1),
            }
            with patch("train.build_models", return_value=candidates):
                metadata = train(data_path, root / "artifacts")

            self.assertTrue((root / "artifacts" / "flood_model.joblib").is_file())
            self.assertEqual(metadata["model_name"], "Decision Tree")
            self.assertEqual(metadata["task_type"], "classification")
            self.assertEqual(metadata["split_strategy"], "chronological by YEAR")
            self.assertEqual(metadata["holdout_rows"], 12)
            self.assertIn("accuracy", metadata["holdout_metrics"])

    def test_untrained_app_explains_setup_and_returns_service_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(Path(directory)).test_client()
            page = client.get("/")
            response = client.post("/predict", data={})

        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Train your first model", page.data)
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
