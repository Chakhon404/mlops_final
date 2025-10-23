import sys, mlflow

def transition_model_alias(model_name: str, alias: str):
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(model_name)
    if not latest:
        raise RuntimeError("No model versions found.")
    v = max(latest, key=lambda m: int(m.version))
    client.set_registered_model_alias(model_name, alias, v.version)
    print(f"[ALIAS] {model_name} -> alias '{alias}' -> v{v.version}")

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("Usage: python scripts/04_transition_model.py <model_name> <alias>")
        sys.exit(1)
    transition_model_alias(sys.argv[1], sys.argv[2])
