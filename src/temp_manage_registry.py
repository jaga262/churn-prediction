import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

def inspect_and_manage_registry():
    db_path = os.path.abspath("mlflow.db").replace("\\", "/")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_name = "Telco_Churn_Production_Model"
    print(f"[INFO] Inspecting MLflow Model Registry for '{model_name}'...")
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print("[INFO] No model versions currently registered.")
            return

        print(f"{'Version':<10} {'Stage':<15} {'Run ID':<35} {'Status':<15}")
        print("-" * 75)
        for v in versions:
            print(f"{v.version:<10} {v.current_stage:<15} {v.run_id:<35} {v.status:<15}")
    except Exception as e:
        print(f"[ERROR] Failed to query model registry: {e}")

if __name__ == "__main__":
    inspect_and_manage_registry()
