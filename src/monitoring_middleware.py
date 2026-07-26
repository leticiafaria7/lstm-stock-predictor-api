import time

from flask import request

from src.monitoring import (
    HTTP_REQUESTS,
    HTTP_ERRORS,
    HTTP_DURATION,
    HTTP_SUMMARY
)

def register_monitoring(app):

    @app.before_request
    def before():

        request.start_time = time.perf_counter()

    @app.after_request
    def after(response):

        duration = time.perf_counter() - request.start_time

        endpoint = request.path

        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()

        HTTP_DURATION.labels(
            endpoint=endpoint
        ).observe(duration)

        HTTP_SUMMARY.labels(
            endpoint=endpoint
        ).observe(duration)

        if response.status_code >= 400:

            HTTP_ERRORS.labels(
                endpoint=endpoint
            ).inc()

        return response