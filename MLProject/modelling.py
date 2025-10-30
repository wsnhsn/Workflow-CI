"""
Pipeline training untuk MLflow Project.

Skrip ini membaca dataset hasil preprocessing, melatih RandomForestClassifier,
dan mencatat metrik serta model ke MLflow. Dirancang agar dapat dieksekusi
melalui `mlflow run`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from contextlib import contextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


TARGET_COL = "burns_calories_bin"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "namadataset_preprocessing/final_data_preprocessed.csv"


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Memuat dataset siap latih lalu memisahkan fitur dan target."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


@contextmanager
def _mlflow_run_scope():
    """Pastikan ada run MLflow aktif; gunakan nested run jika parent sudah ada."""
    if mlflow.active_run() is None:
        with mlflow.start_run():
            yield
    else:
        with mlflow.start_run(nested=True):
            yield


def train(X: pd.DataFrame, y: pd.Series, experiment: str, tracking_uri: str | None) -> None:
    """Melatih RandomForest dan mencatat artefak ke MLflow."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if mlflow.active_run() is None:
        mlflow.set_experiment(experiment)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    with _mlflow_run_scope():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_macro_f1", macro_f1)

        mlflow.sklearn.log_model(model, artifact_path="model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latih RandomForest melalui MLflow Project.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Lokasi CSV hasil preprocessing.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="mlproject_random_forest",
        help="Nama eksperimen MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="URI tracking MLflow (opsional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data_path if args.data_path.is_absolute() else BASE_DIR / args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path.resolve()}")

    X, y = load_dataset(data_path)
    train(X, y, experiment=args.experiment_name, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
