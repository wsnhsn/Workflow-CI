"""
Utility untuk membangun image Docker dari run MLflow terbaru.

Skrip ini diasumsikan dijalankan di dalam folder MLProject sehingga
`mlruns` lokal dapat diakses langsung.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow


def resolve_latest_run(experiment_name: str) -> str:
    """Mengambil run_id terbaru dari eksperimen MLflow yang diberikan."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Eksperimen MLflow dengan nama '{experiment_name}' tidak ditemukan."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(
            f"Tidak ada run yang ditemukan pada eksperimen '{experiment_name}'."
        )

    return str(runs.loc[0, "run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bangun image Docker dari model MLflow terbaru."
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Nama eksperimen MLflow yang ingin diambil run terbarunya.",
    )
    parser.add_argument(
        "--image-name",
        required=True,
        help="Nama/tag image Docker, contoh: username/repo:tag.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Tracking URI MLflow. Default ke mlruns lokal bila tidak diisi.",
    )
    parser.add_argument(
        "--model-artifact",
        default="model",
        help="Lokasi artefak model relatif terhadap run (default: 'model').",
    )
    args = parser.parse_args()

    tracking_uri = args.tracking_uri or f"file:{Path.cwd() / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)

    run_id = resolve_latest_run(args.experiment_name)
    model_uri = f"runs:/{run_id}/{args.model_artifact}"

    print(
        f"Membangun image Docker '{args.image_name}' dari run '{run_id}' "
        f"dengan artefak '{args.model_artifact}'."
    )

    mlflow.models.build_docker(model_uri=model_uri, name=args.image_name)

    print("Image Docker selesai dibangun.")


if __name__ == "__main__":
    main()

