import os
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

# ===== กำหนด path ของโมเดลโดยตรง =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# ===== โหลดโมเดล =====
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# ===== สร้าง FastAPI app =====
appml = FastAPI(title="Mental Health Classifier", version="2.0")

# ===== ตั้งค่า templates =====
templates = Jinja2Templates(directory="templates")

# ===== โครงสร้าง input =====
class TextInputs(BaseModel):
    text: list[str]  # รองรับหลายข้อความ

# ===== API Endpoint: /predict =====
@appml.post("/predict")
def predict(input_data: TextInputs):
    if model is None:
        return {"error": "Model not loaded"}
    df = pd.DataFrame({"text": input_data.text})
    preds = model.predict(df["text"])
    results = [{"text": t, "prediction": p} for t, p in zip(input_data.text, preds)]
    return {"results": results}

# ===== หน้าเว็บหลัก (GET) =====
@appml.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# ===== หน้าเว็บ: รับ input แล้วทำนาย (POST) =====
@appml.post("/", response_class=HTMLResponse)
def web_predict(request: Request, user_text: str = Form(...)):
    if model is None:
        pred = "Model not loaded"
    else:
        df = pd.DataFrame({"text": [user_text]})
        pred = model.predict(df["text"])[0]
    return templates.TemplateResponse("index.html", {"request": request, "result": pred, "user_text": user_text})
