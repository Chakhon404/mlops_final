# mlops_pipeline/scripts/04_load_and_predict.py
import os
import sys
import json
import argparse
import mlflow

LABELS = ["Anxiety","Bipolar","Depression","Normal","Personality Disorder","Stress","Suicidal"]

def resolve_model_uri(args) -> str:
    """
    ลำดับความสำคัญ:
    1) --uri (เช่น runs:/<RUN_ID>/mh_text_pipeline)
    2) ENV: MODEL_URI
    3) Model Registry: models:/<MODEL_NAME>/<MODEL_STAGE>
       โดยค่าเริ่มต้น MODEL_NAME=MH-classifier-prod, MODEL_STAGE=Staging
    """
    if args.uri:
        return args.uri
    if os.getenv("MODEL_URI"):
        return os.getenv("MODEL_URI")
    name = os.getenv("MODEL_NAME", "MH-classifier-prod")
    stage = os.getenv("MODEL_STAGE", "Staging")
    return f"models:/{name}/{stage}"

def main():
    parser = argparse.ArgumentParser(description="Load MH model from MLflow and predict texts.")
    parser.add_argument("--uri", help="Explicit model URI (e.g. runs:/<RUN_ID>/mh_text_pipeline)")
    parser.add_argument("--texts", nargs="*", help="Texts to classify")
    args = parser.parse_args()

    model_uri = resolve_model_uri(args)
    print(f"[INFO] Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    texts = args.texts or [
        "I feel anxious and can't sleep at all lately.",
        "Everything is stable and I feel fine these days.",
        "I'm overwhelmed with stress from work and family."
    ]
    preds = model.predict(texts)

    results = []
    for t, p in zip(texts, preds):
        try:
            i = int(p)
            label = LABELS[i] if 0 <= i < len(LABELS) else str(p)
        except Exception:
            label = str(p)
        results.append({"text": t, "label": label, "raw": str(p)})

    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
