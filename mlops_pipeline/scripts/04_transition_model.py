import sys
from mlflow.tracking import MlflowClient

def transition_model_alias(model_name, alias):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"No versions found for model '{model_name}'."); return
    latest = max(versions, key=lambda mv: int(mv.version))
    client.set_registered_model_alias(name=model_name, alias=alias, version=latest.version)
    print(f"Alias '{alias}' -> model '{model_name}' v{latest.version}")

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("Usage: python mlops_pipeline/scripts/04_transition_model.py <model_name> <alias>"); sys.exit(1)
    transition_model_alias(sys.argv[1], sys.argv[2])
