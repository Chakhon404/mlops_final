import os
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

# ตั้งค่า MLflow Tracking URI (dynamic)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_PATH = os.path.join(BASE_DIR,"..", "mlruns")

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", f"file:///{MLRUNS_PATH.replace(os.sep, '/')}")
)

# โหลดโมเดลจาก Model Registry (Staging)
MODEL_URI = "models:/mental-health-classifier/Staging"
try:
    model = mlflow.pyfunc.load_model(model_uri=MODEL_URI)
    print(f"Model loaded successfully from {MODEL_URI}")
except Exception as e:
    print(f"Failed to load model from {MODEL_URI}: {e}")
    model = None

app = FastAPI(title="Mental Health Classifier", version="2.0")

# ตั้งค่า templates (โฟลเดอร์เดียวกับไฟล์นี้)
templates = Jinja2Templates(directory="templates")

# โครงสร้าง Input ของ API
class TextInputs(BaseModel):
    text: list[str]  # รองรับหลายข้อความ

# API Endpoint: /predict
@app.post("/predict")
def predict(input_data: TextInputs):
    if model is None:
        return {"error": "Model not loaded"}
    df = pd.DataFrame({"text": input_data.text})
    preds = model.predict(df)
    results = [{"text": t, "prediction": p} for t, p in zip(input_data.text, preds)]
    return {"results": results}

# หน้าเว็บหลัก (GET)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# หน้าเว็บ: รับ input แล้วทำนาย (POST)
@app.post("/", response_class=HTMLResponse)
def web_predict(request: Request, user_text: str = Form(...)):
    if model is None:
        pred = "Model not loaded"
    else:
        df = pd.DataFrame({"text": [user_text]})
        pred = model.predict(df)[0]

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": pred, "user_text": user_text}
    )
