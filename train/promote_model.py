"""Compares the latest MLflow training run against the model version currently
aliased "production" and promotes it if it does not regress the eval F1 score.

Latent by design: this only does anything once MLFLOW_TRACKING_URI points at a
reachable MLflow server. The local docker-compose instance is not reachable from
a GitHub Actions runner, so until that secret is added, the workflow that calls
this script is a no-op - see .github/workflows/model-promotion.yml.
"""
import os

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "intent-classifier-phase2")
REGISTERED_MODEL_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "intent-classifier-customer-support")
METRIC_NAME = "eval_f1_macro"
PRODUCTION_ALIAS = "production"
# A new run within this tolerance of the current production score is still
# promoted - retraining on refreshed data naturally has run-to-run noise, and a
# flat "must be strictly better" gate would leave the deployed model stale forever.
PROMOTION_TOLERANCE = float(os.environ.get("PROMOTION_TOLERANCE", "0.003"))


def _latest_run(client: MlflowClient):
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"no runs logged in experiment '{EXPERIMENT_NAME}'")
    return runs[0]


def _current_production_metric(client: MlflowClient) -> float | None:
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None
    run = client.get_run(version.run_id)
    return run.data.metrics.get(METRIC_NAME)


def main() -> None:
    client = MlflowClient()
    latest_run = _latest_run(client)
    new_metric = latest_run.data.metrics.get(METRIC_NAME)
    if new_metric is None:
        raise SystemExit(f"latest run {latest_run.info.run_id} has no '{METRIC_NAME}' metric")

    current_metric = _current_production_metric(client)

    if current_metric is not None and new_metric < current_metric - PROMOTION_TOLERANCE:
        print(
            f"REJECTED: new run {METRIC_NAME}={new_metric:.4f} is worse than "
            f"production {METRIC_NAME}={current_metric:.4f} (tolerance={PROMOTION_TOLERANCE})"
        )
        return

    model_uri = f"runs:/{latest_run.info.run_id}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, registered.version)

    baseline = "none (first production version)" if current_metric is None else f"{current_metric:.4f}"
    print(
        f"PROMOTED: run {latest_run.info.run_id} -> {REGISTERED_MODEL_NAME} v{registered.version} "
        f"({METRIC_NAME}={new_metric:.4f}, previous production={baseline})"
    )


if __name__ == "__main__":
    main()
