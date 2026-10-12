"""Train and compare flood classification or probability regression models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


PROBABILITY_TARGET = "FloodProbability"
CLASS_TARGET = "FLOODS"
RANDOM_STATE = 42


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Read a labelled CSV and infer classification or probability regression."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Put the labelled CSV at that path."
        )

    frame = pd.read_csv(data_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    target_columns = {str(column).casefold(): column for column in frame.columns}
    if CLASS_TARGET.casefold() in target_columns:
        target_name = target_columns[CLASS_TARGET.casefold()]
        task_type = "classification"
        target = frame[target_name].astype("string").str.strip()
        valid_rows = target.notna() & target.ne("")
    elif PROBABILITY_TARGET.casefold() in target_columns:
        target_name = target_columns[PROBABILITY_TARGET.casefold()]
        task_type = "regression"
        target = pd.to_numeric(frame[target_name], errors="coerce")
        valid_rows = target.notna() & target.between(0, 1)
    else:
        raise ValueError(
            f"Dataset must contain a '{CLASS_TARGET}' or "
            f"'{PROBABILITY_TARGET}' target column."
        )

    feature_frame = frame.drop(
        columns=[
            column
            for column in frame.columns
            if column == target_name or column.lower() in {"id", "index"}
        ]
    ).copy()
    for column in feature_frame.select_dtypes(include=["object", "string"]).columns:
        feature_frame[column] = feature_frame[column].astype("string").str.strip()
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.loc[valid_rows].dropna(axis="columns", how="all")
    target = target.loc[valid_rows]
    feature_frame = feature_frame.loc[:, feature_frame.nunique(dropna=True) > 1]

    if feature_frame.shape[1] == 0:
        raise ValueError("The dataset has no usable non-constant predictor columns.")
    if len(feature_frame) < 20:
        raise ValueError("At least 20 labelled rows are required for a reliable split.")
    if task_type == "classification" and target.nunique() < 2:
        raise ValueError("A classification dataset must have at least two target classes.")
    if task_type == "classification" and target.value_counts().min() < 3:
        raise ValueError("Each target class needs at least three rows for validation.")

    return feature_frame, target, task_type


def make_pipeline(estimator: Any, features: pd.DataFrame) -> Pipeline:
    """Build preprocessing and estimator steps from the input feature types."""
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers)),
            ("estimator", estimator),
        ]
    )


def build_models(
    task_type: str, features: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Create Decision Tree, Random Forest, KNN, and XGBoost model pipelines."""
    if task_type == "classification":
        estimators = {
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=2, random_state=RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "XGBoost": XGBClassifier(
                objective="binary:logistic",
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
        }
    elif task_type == "regression":
        estimators = {
            "Decision Tree": DecisionTreeRegressor(
                max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "K-Nearest Neighbours": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
            "XGBoost": XGBRegressor(
                objective="reg:squarederror",
                n_estimators=250,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if features is None:
        return estimators
    return {
        name: make_pipeline(estimator, features)
        for name, estimator in estimators.items()
    }


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def feature_form_metadata(features: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    defaults: dict[str, Any] = {}
    feature_types: dict[str, str] = {}
    choices: dict[str, list[str]] = {}
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            defaults[column] = float(features[column].median())
            feature_types[column] = "number"
        else:
            observed = features[column].dropna().astype(str)
            if observed.empty:
                continue
            defaults[column] = str(observed.mode().iloc[0])
            feature_types[column] = "category"
            unique_values = sorted(observed.unique().tolist())
            if len(unique_values) <= 50:
                choices[column] = unique_values
    return (
        {
            "feature_defaults": defaults,
            "feature_choices": choices,
            "feature_types": feature_types,
        },
        feature_types,
    )


def train(data_path: Path, artifact_dir: Path) -> dict[str, Any]:
    features, target, task_type = load_dataset(data_path)
    label_encoder: LabelEncoder | None = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        encoded_target = pd.Series(label_encoder.fit_transform(target.astype(str)))
        target_values = encoded_target
        labels = label_encoder.classes_.tolist()
        if len(labels) != 2:
            raise ValueError("Currently the classification task requires exactly two classes.")
        positive_index = next(
            (
                index
                for index, label in enumerate(labels)
                if str(label).strip().casefold()
                in {"yes", "flood", "flooded", "true", "1"}
            ),
            1,
        )
        stratify = target_values
    else:
        target_values = target
        labels = []
        positive_index = None
        stratify = None

    year_column = next(
        (column for column in features.columns if column.casefold() == "year"), None
    )
    if year_column is not None:
        order = np.argsort(
            pd.to_numeric(features[year_column], errors="coerce").to_numpy(),
            kind="stable",
        )
        features = features.iloc[order].reset_index(drop=True)
        target_values = target_values.iloc[order].reset_index(drop=True)
        train_end = int(len(features) * 0.8)
        validation_start = int(train_end * 0.8125)
        x_fit = features.iloc[:validation_start]
        y_fit = target_values.iloc[:validation_start]
        x_validation = features.iloc[validation_start:train_end]
        y_validation = target_values.iloc[validation_start:train_end]
        x_holdout = features.iloc[train_end:]
        y_holdout = target_values.iloc[train_end:]
        split_strategy = f"chronological by {year_column}"
    else:
        x_train, x_holdout, y_train, y_holdout = train_test_split(
            features,
            target_values,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
        x_fit, x_validation, y_fit, y_validation = train_test_split(
            x_train,
            y_train,
            test_size=0.1875,
            random_state=RANDOM_STATE,
            stratify=y_train if task_type == "classification" else None,
        )
        split_strategy = "random holdout with fixed seed"

    if task_type == "classification":
        for partition_name, partition in (
            ("training", y_fit),
            ("validation", y_validation),
            ("holdout", y_holdout),
        ):
            if partition.nunique() != len(labels):
                raise ValueError(
                    f"The chronological {partition_name} partition does not contain "
                    "every class; use more labelled years or a non-temporal dataset."
                )

    validation_results: dict[str, dict[str, float]] = {}
    best_name: str | None = None
    best_score = float("-inf") if task_type == "classification" else float("inf")
    models = build_models(task_type, features)

    for name, model in models.items():
        model.fit(x_fit, y_fit)
        predicted = model.predict(x_validation)
        if task_type == "classification":
            metrics = classification_metrics(y_validation.to_numpy(), predicted)
            selection_score = metrics["accuracy"]
            print(
                f"{name}: validation accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}"
            )
        else:
            metrics = regression_metrics(y_validation, predicted)
            selection_score = metrics["rmse"]
            print(
                f"{name}: validation RMSE={metrics['rmse']:.5f}, "
                f"R²={metrics['r2']:.4f}"
            )
        validation_results[name] = metrics
        if (
            selection_score > best_score
            if task_type == "classification"
            else selection_score < best_score
        ):
            best_name, best_score = name, selection_score

    if best_name is None:
        raise RuntimeError("No candidate model was successfully evaluated.")

    selected_model = models[best_name]
    refit_features = pd.concat([x_fit, x_validation])
    refit_target = pd.concat([y_fit, y_validation])
    selected_model.fit(refit_features, refit_target)
    holdout_predictions = selected_model.predict(x_holdout)
    if task_type == "classification":
        holdout_metrics = classification_metrics(
            y_holdout.to_numpy(), holdout_predictions
        )
        print(
            f"\nSelected {best_name}; holdout accuracy="
            f"{holdout_metrics['accuracy']:.4f}, F1={holdout_metrics['f1']:.4f}"
        )
    else:
        holdout_metrics = regression_metrics(y_holdout, holdout_predictions)
        print(
            f"\nSelected {best_name}; holdout RMSE={holdout_metrics['rmse']:.5f}, "
            f"R²={holdout_metrics['r2']:.4f}"
        )

    form_metadata, _ = feature_form_metadata(features)
    metadata: dict[str, Any] = {
        "target": CLASS_TARGET if task_type == "classification" else PROBABILITY_TARGET,
        "task_type": task_type,
        "model_name": best_name,
        "split_strategy": split_strategy,
        "features": features.columns.tolist(),
        "target_classes": labels,
        "positive_class_index": positive_index,
        **form_metadata,
        "validation_metrics": validation_results,
        "holdout_metrics": holdout_metrics,
        "holdout_rows": int(len(y_holdout)),
        "training_rows": int(len(refit_target)),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, artifact_dir / "flood_model.joblib")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved model artifacts to {artifact_dir.resolve()}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/kerala.csv"),
        help="Labelled CSV with FLOODS or FloodProbability (default: data/kerala.csv)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for the trained model and metadata (default: artifacts)",
    )
    args = parser.parse_args()
    train(args.data, args.artifact_dir)


if __name__ == "__main__":
    main()
