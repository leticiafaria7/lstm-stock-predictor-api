from prometheus_client import Counter, Gauge, Histogram, Summary

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_ERRORS = Counter(
    "http_request_errors_total",
    "HTTP errors",
    ["endpoint"]
)

HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["endpoint"]
)

HTTP_SUMMARY = Summary(
    "http_request_summary_seconds",
    "HTTP request summary",
    ["endpoint"]
)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Prediction requests",
    ["ticker"]
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Prediction errors",
    ["ticker"]
)

PREDICTION_DURATION = Histogram(
    "prediction_latency_seconds",
    "Prediction latency",
    ["ticker"]
)

DATABASE_UP = Gauge(
    "database_up",
    "Database status"
)

PREDICTIONS_ROWS = Gauge(
    "predictions_rows",
    "Rows in predictions table"
)

SUPPORTED_MODELS = Gauge(
    "supported_models",
    "Number of supported models"
)

MODEL_MAE = Gauge(
    "model_mae",
    "Model MAE",
    ["ticker"]
)

MODEL_RMSE = Gauge(
    "model_rmse",
    "Model RMSE",
    ["ticker"]
)

MODEL_MAPE = Gauge(
    "model_mape",
    "Model MAPE",
    ["ticker"]
)

WORKFLOW_DURATION = Gauge(
    "workflow_duration_seconds",
    "Workflow duration"
)

WORKFLOW_LAST_SUCCESS = Gauge(
    "workflow_last_success_timestamp",
    "Last successful workflow"
)

WORKFLOW_FAILURES = Gauge(
    "workflow_failures_total",
    "Workflow failures"
)