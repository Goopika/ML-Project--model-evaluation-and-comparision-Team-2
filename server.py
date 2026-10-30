import io
import os
import threading
import urllib.request
import webbrowser
import zipfile
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# --------------------------------------------------
# APPLICATION SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CHART_DIR = STATIC_DIR / "charts"
UPLOAD_DIR = BASE_DIR / "uploads"

for folder in (CHART_DIR, UPLOAD_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

uploaded_datasets = {}
_student_cache = None

BUILT_IN_DATASETS = [
    "Iris Flower",
    "Wine Classification",
    "Credit Card Fraud",
    "Student Performance",
]


# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

def load_iris_data():
    dataset = load_iris(as_frame=True)
    df = dataset.data.copy()
    df["target"] = dataset.target
    return df, "target"


def load_wine_data():
    dataset = load_wine(as_frame=True)
    df = dataset.data.copy()
    df["target"] = dataset.target
    return df, "target"


def load_credit_card_data():
    csv_path = BASE_DIR / "creditcard.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found. Place it in the same "
            "folder as server.py."
        )

    df = pd.read_csv(csv_path)

    if "Class" not in df.columns:
        raise ValueError(
            "creditcard.csv must contain a 'Class' column."
        )

    # Retain all fraud examples and sample non-fraud examples
    # to make training faster on this large dataset.
    if len(df) > 15000:
        positives = df[df["Class"] == 1]
        negatives = df[df["Class"] == 0]

        negative_limit = max(1000, 15000 - len(positives))

        if len(negatives) > negative_limit:
            negatives = negatives.sample(
                n=negative_limit,
                random_state=42,
            )

        df = pd.concat(
            [negatives, positives],
            axis=0,
        ).sample(frac=1, random_state=42)

    return df, "Class"


def load_student_data():
    global _student_cache

    if _student_cache is not None:
        target = "G3" if "G3" in _student_cache.columns else _student_cache.columns[-1]
        return _student_cache.copy(), target

    # A local CSV avoids downloading the dataset repeatedly.
    for filename in ["student-mat.csv", "student.csv"]:
        local_path = BASE_DIR / filename

        if local_path.exists():
            df = pd.read_csv(local_path, sep=";")
            _student_cache = df
            target = "G3" if "G3" in df.columns else df.columns[-1]
            return df.copy(), target

    url = (
        "https://archive.ics.uci.edu/static/public/320/"
        "student+performance.zip"
    )

    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            archive_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer_zip:
            names = outer_zip.namelist()

            csv_name = next(
                (
                    name for name in names
                    if Path(name).name.lower() == "student-mat.csv"
                ),
                None,
            )

            if csv_name:
                with outer_zip.open(csv_name) as csv_file:
                    df = pd.read_csv(csv_file, sep=";")

            else:
                nested_name = next(
                    (
                        name for name in names
                        if name.lower().endswith(".zip")
                    ),
                    None,
                )

                if not nested_name:
                    raise FileNotFoundError(
                        "No CSV file or nested ZIP was found."
                    )

                with outer_zip.open(nested_name) as nested_file:
                    nested_bytes = nested_file.read()

                with zipfile.ZipFile(
                    io.BytesIO(nested_bytes)
                ) as nested_zip:
                    csv_name = next(
                        (
                            name for name in nested_zip.namelist()
                            if Path(name).name.lower()
                            == "student-mat.csv"
                        ),
                        None,
                    )

                    if not csv_name:
                        raise FileNotFoundError(
                            "student-mat.csv was not found "
                            "inside the nested ZIP."
                        )

                    with nested_zip.open(csv_name) as csv_file:
                        df = pd.read_csv(csv_file, sep=";")

        _student_cache = df

        target = "G3" if "G3" in df.columns else df.columns[-1]
        return df.copy(), target

    except Exception as exc:
        raise RuntimeError(
            "Could not load Student Performance. Check your internet "
            "connection, or place student-mat.csv beside server.py. "
            f"Details: {exc}"
        ) from exc


def get_builtin_dataset(name):
    if name == "Iris Flower":
        return load_iris_data()

    if name == "Wine Classification":
        return load_wine_data()

    if name == "Credit Card Fraud":
        return load_credit_card_data()

    if name == "Student Performance":
        return load_student_data()

    raise ValueError("Please select a valid dataset.")


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def make_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError(
            "No usable feature columns were found."
        )

    # Dense output also supports Gradient Boosting estimators.
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


# --------------------------------------------------
# MACHINE LEARNING MODELS
# --------------------------------------------------

def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),

        "Support Vector Machine": SVC(),

        "K-Nearest Neighbors": KNeighborsClassifier(),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
    }


def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42
        ),
    }


# --------------------------------------------------
# CHART UTILITIES
# --------------------------------------------------

def chart_url(filename):
    path = CHART_DIR / filename
    version = int(path.stat().st_mtime)
    return f"/static/charts/{filename}?v={version}"


def save_classification_charts(
    results,
    y_test,
    best_predictions,
    label_encoder,
    target_name,
    X,
):
    charts = {
        "comparison_chart": None,
        "classification_metrics_chart": None,
        "confusion_matrix": None,
        "target_distribution_chart": None,
        "feature_distribution_chart": None,
    }

    valid = {
        name: metrics
        for name, metrics in results.items()
        if "Error" not in metrics
    }

    names = list(valid.keys())

    # Graph 1: Accuracy comparison
    if valid:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.bar(
            names,
            [valid[name]["Accuracy"] for name in names],
        )

        ax.set_title("Model Comparison - Accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        path = CHART_DIR / "model_comparison.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)

        charts["comparison_chart"] = chart_url(path.name)

    # Graph 2: Precision, Recall and F1 Score
    if valid:
        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            x - width,
            [valid[name]["Precision"] for name in names],
            width,
            label="Precision",
        )

        ax.bar(
            x,
            [valid[name]["Recall"] for name in names],
            width,
            label="Recall",
        )

        ax.bar(
            x + width,
            [valid[name]["F1 Score"] for name in names],
            width,
            label="F1 Score",
        )

        ax.set_title(
            "Precision, Recall and F1-Score Comparison"
        )
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        path = CHART_DIR / "classification_metrics.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)

        charts["classification_metrics_chart"] = chart_url(
            path.name
        )

    # Graph 3: Confusion Matrix for the best model
    if best_predictions is not None:
        matrix = confusion_matrix(y_test, best_predictions)

        fig, ax = plt.subplots(figsize=(6, 5))

        image = ax.imshow(matrix, interpolation="nearest")

        ax.set_title("Confusion Matrix - Best Model")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        fig.colorbar(image, ax=ax)

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                )

        fig.tight_layout()

        path = CHART_DIR / "confusion_matrix.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)

        charts["confusion_matrix"] = chart_url(path.name)

    # Graph 4: Target class distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))

    decoded_targets = label_encoder.inverse_transform(
        np.asarray(y_test, dtype=int)
    )

    pd.Series(decoded_targets).value_counts().plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(f"Target Distribution: {target_name}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    path = CHART_DIR / "target_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    charts["target_distribution_chart"] = chart_url(
        path.name
    )

    # Graph 5: First feature distribution
    numeric = X.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if not numeric.empty:
        column = numeric.columns[0]

        numeric[column].dropna().plot(
            kind="hist",
            bins=20,
            ax=ax,
        )

        ax.set_title(f"Feature Distribution: {column}")
        ax.set_xlabel(str(column))

    else:
        column = X.columns[0]

        X[column].astype(str).value_counts().head(12).plot(
            kind="bar",
            ax=ax,
        )

        ax.set_title(f"Feature Distribution: {column}")
        ax.set_xlabel(str(column))
        ax.tick_params(axis="x", rotation=25)

    ax.set_ylabel("Count")
    fig.tight_layout()

    path = CHART_DIR / "feature_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    charts["feature_distribution_chart"] = chart_url(
        path.name
    )

    return charts


def save_regression_charts(
    results,
    y_test,
    best_predictions,
    target_name,
    X,
):
    charts = {
        "comparison_chart": None,
        "regression_metrics_chart": None,
        "actual_vs_predicted_chart": None,
        "target_distribution_chart": None,
        "feature_distribution_chart": None,
    }

    valid = {
        name: metrics
        for name, metrics in results.items()
        if "Error" not in metrics
    }

    names = list(valid.keys())

    # Graph 1: R² Score comparison
    if valid:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.bar(
            names,
            [valid[name]["R² Score"] for name in names],
        )

        ax.set_title("Model Comparison - R² Score")
        ax.set_ylabel("R² Score")
        ax.axhline(0, linewidth=1)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        path = CHART_DIR / "model_comparison.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)

        charts["comparison_chart"] = chart_url(path.name)

    # Graph 2: MAE and RMSE comparison
    if valid:
        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            x - width / 2,
            [valid[name]["MAE"] for name in names],
            width,
            label="MAE",
        )

        ax.bar(
            x + width / 2,
            [valid[name]["RMSE"] for name in names],
            width,
            label="RMSE",
        )

        ax.set_title(
            "Regression Error Comparison (Lower is Better)"
        )
        ax.set_ylabel("Error")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        path = CHART_DIR / "regression_metrics.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)

        charts["regression_metrics_chart"] = chart_url(
            path.name
        )

    # Graph 3: Actual vs Predicted
    if best_predictions is not None:
        actual = np.asarray(y_test, dtype=float)
        predicted = np.asarray(best_predictions, dtype=float)

        fig, ax = plt.subplots(figsize=(6.5, 5))

        ax.scatter(actual, predicted, alpha=0.7)

        low = min(float(actual.min()), float(predicted.min()))
        high = max(float(actual.max()), float(predicted.max()))

        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
        )

        ax.set_title("Actual vs Predicted Values")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")

        fig.tight_layout()

        path = CHART_DIR / "actual_vs_predicted.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)

        charts["actual_vs_predicted_chart"] = chart_url(
            path.name
        )

    # Graph 4: Target distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))

    pd.Series(y_test).plot(
        kind="hist",
        bins=15,
        ax=ax,
    )

    ax.set_title(f"Target Distribution: {target_name}")
    ax.set_xlabel(str(target_name))
    ax.set_ylabel("Count")

    fig.tight_layout()

    path = CHART_DIR / "target_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    charts["target_distribution_chart"] = chart_url(
        path.name
    )

    # Graph 5: Feature distribution
    numeric = X.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if not numeric.empty:
        column = numeric.columns[0]

        numeric[column].dropna().plot(
            kind="hist",
            bins=20,
            ax=ax,
        )

        ax.set_title(f"Feature Distribution: {column}")
        ax.set_xlabel(str(column))

    else:
        column = X.columns[0]

        X[column].astype(str).value_counts().head(12).plot(
            kind="bar",
            ax=ax,
        )

        ax.set_title(f"Feature Distribution: {column}")
        ax.set_xlabel(str(column))
        ax.tick_params(axis="x", rotation=25)

    ax.set_ylabel("Count")
    fig.tight_layout()

    path = CHART_DIR / "feature_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    charts["feature_distribution_chart"] = chart_url(
        path.name
    )

    return charts


# --------------------------------------------------
# WEBPAGE
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    return jsonify({"datasets": BUILT_IN_DATASETS})


# --------------------------------------------------
# UPLOAD DATASET
# --------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    try:
        file = request.files.get("file")

        if file is None or not file.filename:
            return jsonify({
                "error": "Please choose a dataset file."
            }), 400

        extension = Path(file.filename).suffix.lower()

        if extension not in {".csv", ".xlsx", ".xls"}:
            return jsonify({
                "error": "Upload a CSV, XLSX or XLS file."
            }), 400

        safe_name = Path(file.filename).name

        if extension == ".csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        if df.empty or df.shape[0] < 10 or df.shape[1] < 2:
            return jsonify({
                "error": (
                    "The dataset must have at least "
                    "10 rows and 2 columns."
                )
            }), 400

        df.columns = [str(column) for column in df.columns]

        upload_id = str(uuid4())

        uploaded_datasets[upload_id] = {
            "data": df,
            "filename": safe_name,
        }

        return jsonify({
            "upload_id": upload_id,
            "filename": safe_name,
            "rows": int(len(df)),
            "columns": [str(column) for column in df.columns],
            "message": "Dataset uploaded successfully.",
        })

    except Exception as exc:
        return jsonify({
            "error": f"Could not read uploaded file: {exc}"
        }), 400


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    try:
        payload = request.get_json(silent=True) or {}

        task = payload.get("task")
        upload_id = payload.get("upload_id")
        target_requested = payload.get("target_column")

        if task not in {"Classification", "Regression"}:
            return jsonify({
                "error": "Choose Classification or Regression."
            }), 400

        if upload_id:
            if upload_id not in uploaded_datasets:
                return jsonify({
                    "error": (
                        "Uploaded dataset session expired. "
                        "Please upload the file again."
                    )
                }), 400

            upload = uploaded_datasets[upload_id]
            df = upload["data"].copy()
            dataset_name = "Uploaded: " + upload["filename"]
            default_target = str(df.columns[-1])

        else:
            dataset_name = payload.get("dataset")

            if dataset_name not in BUILT_IN_DATASETS:
                return jsonify({
                    "error": "Please select an available dataset."
                }), 400

            if (
                task == "Regression"
                and dataset_name != "Student Performance"
            ):
                return jsonify({
                    "error": (
                        "Regression is available only for Student "
                        "Performance among the built-in datasets."
                    )
                }), 400

            df, default_target = get_builtin_dataset(dataset_name)

        target = target_requested or default_target

        if target not in df.columns:
            return jsonify({
                "error": f"Target column '{target}' was not found."
            }), 400

        df = df.dropna(subset=[target]).copy()

        if len(df) < 10:
            return jsonify({
                "error": (
                    "Not enough rows remain after removing "
                    "missing target values."
                )
            }), 400

        y = df[target]
        X = df.drop(columns=[target]).dropna(axis=1, how="all")

        if X.empty:
            return jsonify({
                "error": (
                    "The dataset needs at least one feature "
                    "column besides the target."
                )
            }), 400

        # Prepare the target
        if task == "Classification":
            y = y.astype(str)

            if y.nunique() < 2:
                return jsonify({
                    "error": (
                        "Classification requires at least "
                        "two target classes."
                    )
                }), 400

            if y.nunique() > 50:
                return jsonify({
                    "error": (
                        "The classification target has more than "
                        "50 unique values. Select a categorical target."
                    )
                }), 400

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        else:
            y = pd.to_numeric(y, errors="coerce")
            valid_rows = y.notna()
            X = X.loc[valid_rows]
            y = y.loc[valid_rows]
            label_encoder = None

            if len(y) < 10 or y.nunique() < 2:
                return jsonify({
                    "error": (
                        "Regression requires at least 10 valid rows "
                        "and a numeric target with two or more values."
                    )
                }), 400

        # Test size comes from the webpage.
        test_size = payload.get("test_size", 0.2)

        try:
            test_size = float(test_size)

            if test_size >= 1:
                test_size = test_size / 100

            if not 0.1 <= test_size <= 0.4:
                test_size = 0.2

        except (TypeError, ValueError):
            test_size = 0.2

        stratify_values = None

        if task == "Classification":
            class_counts = pd.Series(y).value_counts()

            if (
                class_counts.min() >= 2
                and int(np.ceil(len(y) * test_size))
                >= len(class_counts)
            ):
                stratify_values = y

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify_values,
            )

        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
            )

        models = (
            get_classification_models()
            if task == "Classification"
            else get_regression_models()
        )

        results = {}
        best_model = None
        best_score = None
        best_predictions = None

        # Train and evaluate every model.
        for model_name, model in models.items():
            try:
                pipeline = Pipeline([
                    ("preprocessor", make_preprocessor(X)),
                    ("model", model),
                ])

                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                if task == "Classification":
                    model_metrics = {
                        "Accuracy": round(
                            float(accuracy_score(y_test, predictions)), 4
                        ),
                        "Precision": round(
                            float(precision_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            )), 4
                        ),
                        "Recall": round(
                            float(recall_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            )), 4
                        ),
                        "F1 Score": round(
                            float(f1_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            )), 4
                        ),
                    }

                    score = model_metrics["Accuracy"]

                else:
                    mse = mean_squared_error(y_test, predictions)

                    model_metrics = {
                        "MAE": round(
                            float(mean_absolute_error(
                                y_test, predictions
                            )), 4
                        ),
                        "MSE": round(float(mse), 4),
                        "RMSE": round(float(np.sqrt(mse)), 4),
                        "R² Score": round(
                            float(r2_score(y_test, predictions)), 4
                        ),
                    }

                    score = model_metrics["R² Score"]

                results[model_name] = model_metrics

                if best_model is None or score > best_score:
                    best_model = model_name
                    best_score = score
                    best_predictions = predictions

            except Exception as exc:
                results[model_name] = {
                    "Error": str(exc)
                }

        valid_results = {
            name: metrics
            for name, metrics in results.items()
            if "Error" not in metrics
        }

        if not valid_results or best_model is None:
            details = "; ".join(
                f"{name}: {metrics.get('Error', 'failed')}"
                for name, metrics in results.items()
            )

            return jsonify({
                "error": "All models failed. " + details
            }), 400

        # Generate graphs for the selected task.
        if task == "Classification":
            charts = save_classification_charts(
                results,
                y_test,
                best_predictions,
                label_encoder,
                target,
                X,
            )

            metric_names = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
            ]

            summary = (
                f"Evaluation completed. Best model: {best_model} "
                f"(Accuracy {best_score:.4f})."
            )

        else:
            charts = save_regression_charts(
                results,
                y_test,
                best_predictions,
                target,
                X,
            )

            metric_names = [
                "MAE",
                "MSE",
                "RMSE",
                "R² Score",
            ]

            summary = (
                f"Evaluation completed. Best model by R² Score: "
                f"{best_model} ({best_score:.4f})."
            )

        # Format the result table for index.html.
        result_rows = []

        for name, metrics in results.items():
            result_rows.append({
                "model": name,
                **metrics,
            })

        return jsonify({
            "dataset": dataset_name,
            "task": task,
            "target_column": str(target),
            "total_rows": int(len(X)),
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test)),
            "metrics": metric_names,
            "results": result_rows,
            "models": results,
            "summary": summary,
            "best_model": best_model,
            "best_score": float(best_score),
            "comparison_chart": charts.get("comparison_chart"),
            "confusion_matrix": charts.get("confusion_matrix"),
            "classification_metrics_chart": charts.get(
                "classification_metrics_chart"
            ),
            "regression_metrics_chart": charts.get(
                "regression_metrics_chart"
            ),
            "actual_vs_predicted_chart": charts.get(
                "actual_vs_predicted_chart"
            ),
            "target_distribution_chart": charts.get(
                "target_distribution_chart"
            ),
            "feature_distribution_chart": charts.get(
                "feature_distribution_chart"
            ),
            "chart_note": (
                "Graphs are generated using the selected dataset "
                "and held-out test data."
            ),
        })

    except Exception as exc:
        app.logger.exception("Evaluation failed")

        return jsonify({
            "error": str(exc)
        }), 400


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "File is too large. Maximum file size is 20 MB."
    }), 413


if __name__ == "__main__":
    if os.environ.get("OPEN_BROWSER", "1") != "0":
        threading.Timer(
            1.0,
            lambda: webbrowser.open("http://127.0.0.1:5000"),
        ).start()

    app.run(debug=True, use_reloader=False)