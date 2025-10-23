# 06_load_and_predict.py
import mlflow
import pandas as pd

def load_and_predict():
    """
    ตัวอย่างจำลองการใช้งานโมเดลใน production:
    โหลดโมเดลจาก Model Registry ตาม stage แล้วทำนายข้อความใหม่
    """
    MODEL_NAME = "mental-health-classifier"   # เปลี่ยนจาก wine-classifier-prod
    MODEL_STAGE = "Staging"                   # เปลี่ยน stage ตามต้องการ

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")

    try:
        # โหลดโมเดลจาก Registry
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        print("Model loaded successfully.")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version is in the '{MODEL_STAGE}' stage in the MLflow UI.")
        return

    # ตัวอย่างข้อความ (สามารถเปลี่ยนได้ตามต้องการ)
    samples = [
        "I feel extremely anxious about my future.",
        "Life is beautiful and I am calm today.",
        "Lately I've been depressed and can't sleep well.",
        "Sometimes I think about ending it all."
    ]

    # สร้าง DataFrame ให้ตรงกับ signature ที่โมเดลคาด (คอลัมน์ 'text')
    X_in = pd.DataFrame({"text": samples})

    # ใช้โมเดลทำนาย
    preds = model.predict(X_in)

    print("-" * 40)
    for s, p in zip(samples, preds):
        print(f"Text: {s}\nPrediction: {p}\n")
    print("-" * 40)

if __name__ == "__main__":
    load_and_predict()
