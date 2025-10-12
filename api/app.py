import os, mlflow
from fastapi import FastAPI
from pydantic import BaseModel

LABELS = ["Anxiety","Bipolar","Depression","Normal","Personality Disorder","Stress","Suicidal"]
MODEL_NAME = os.getenv("MODEL_NAME","MH-classifier-prod")
MODEL_STAGE = os.getenv("MODEL_STAGE","Staging")

app = FastAPI(title="Mental Health Classifier API", version="1.0.0")

class Item(BaseModel):
    text: str

def _load_model():
    uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.pyfunc.load_model(uri)

@app.get("/health")
def health():
    return {"status":"ok","model":MODEL_NAME,"stage":MODEL_STAGE}

@app.post("/predict")
def predict(item: Item):
    model = _load_model()
    pred_id = int(model.predict([item.text])[0])
    return {"label_id": pred_id, "label": LABELS[pred_id]}
