import mlflow
import pandas as pd

def load_and_predict():
    """
    โหลดโมเดลจาก Model Registry (Staging)
    แล้วทำนายข้อความตัวอย่างใหม่
    """
    MODEL_NAME = "mental-health-classifier"
    MODEL_STAGE = "Staging"

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        print("✅ Model loaded successfully.")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version is in the '{MODEL_STAGE}' stage in MLflow UI.")
        return

    # ตัวอย่างข้อความ
    samples = [
        "I feel extremely anxious about my future.",
        "Life is beautiful and I am calm today.",
        "Lately I've been depressed and can't sleep well.",
        "Sometimes I think about ending it all."
    ]

    # สร้าง DataFrame
    X_in = pd.DataFrame({"text": samples})

    # ทำนาย
    preds = model.predict(X_in)

    print("-" * 40)
    for s, p in zip(samples, preds):
        print(f"Text: {s}\nPrediction: {p}\n")
    print("-" * 40)

if __name__ == "__main__":
    load_and_predict()
