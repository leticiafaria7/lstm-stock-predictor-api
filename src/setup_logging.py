import time
from src.instances import supabase

class WorkflowLogger:

    def __init__(self, workflow):

        self.workflow = workflow
        self.start = time.perf_counter()

    def save(
        self,
        status,
        message,
        rows_historical_tickers=0,
        rows_historical_assets=0,
        rows_historical_indexes=0,
        rows_predictions=0
    ):

        duration = round(time.perf_counter() - self.start, 2)

        supabase.table("workflow_logs").insert({

            "workflow": self.workflow,
            "status": status,
            "message": message,
            "duration_seconds": duration,
            "rows_historical_tickers": rows_historical_tickers,
            "rows_historical_assets": rows_historical_assets,
            "rows_historical_indexes": rows_historical_indexes,
            "rows_predictions": rows_predictions

        }).execute()
