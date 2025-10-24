## Model Serving

### MLflow Model Server

รันเซิร์ฟเวอร์สำหรับทำนายผลโดยตรงจาก **MLflow Registry**

mlflow models serve -m "models:/mental-health-classifier/Staging" -p 5001 --env-manager local

**ทดสอบด้วย cURL (Single Line):**

curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d "{\"dataframe_records\":[{\"text\":\"I feel extremely anxious about my future.\"},{\"text\":\"Life is beautiful and I am calm today.\"}]}"

### FastAPI Web Application

อีกทางเลือกหนึ่งคือการ Deploy โมเดลเป็น **Web Application**
เพื่อใช้ในการสาธิตและทดสอบผ่านเบราว์เซอร์

**Run FastAPI App:**

uvicorn templates.app:app --host 127.0.0.1 --port 5002 --reload

uvicorn templates.appml:appml --host 127.0.0.1 --port 5002 --reload

**ทดสอบ API โดยตรง (Single Line):**

curl -X POST http://127.0.0.1:5002/predict -H "Content-Type: application/json" -d "{\"text\":[\"I feel anxious today\",\"Life is good and calm\"]}"

**หรือเข้าผ่านเว็บเบราว์เซอร์:**

[http://127.0.0.1:5002](http://127.0.0.1:5002)
