# Mental-Health Sentiment MLOps (01–04 + Extras)

โครงโปรเจกต์ตัวอย่างสำหรับชุดข้อมูล Kaggle: *Sentiment Analysis for Mental Health* (multi-class 7 labels)
พร้อมสคริปต์ 01–04 ตาม rubric รายวิชา + ตัวอย่าง CI/CD และวิธีเสิร์ฟโมเดลด้วย MLflow

## โครงสร้าง
```
mental-health-mlops/
├─ data/                         # วางไฟล์ CSV จาก Kaggle (เช่น sentiment_mental_health.csv)
├─ mlops_pipeline/
│  ├─ requirements.txt
│  ├─ scripts/
│  │  ├─ common_text.py
│  │  ├─ 01_data_validation.py
│  │  ├─ 02_data_preprocessing.py
│  │  ├─ 03_train_evaluate_register.py
│  │  ├─ 04_transition_model.py
│  │  ├─ 06_client_test.py
└─ .github/workflows/main.yml    # GitHub Actions
```

## เตรียมสภาพแวดล้อม
```bash
pip install -r mlops_pipeline/requirements.txt
```

## วางข้อมูล
- ดาวน์โหลด CSV จาก Kaggle dataset แล้ววางไว้ที่ `data/`
- ตั้งตัวแปรแวดล้อม `DATA_CSV` ให้ชี้ไปยังไฟล์ เช่น
  - Windows (CMD): `set DATA_CSV=data\sentiment_mental_health.csv`

> สคริปต์รองรับคอลัมน์ชื่อ `text`/`label` หรือ `statement`/`status` โดยจะ auto-rename ให้เป็นมาตรฐาน

## รันทีละสเต็ป (โลคอล)
```bash
python mlops_pipeline/scripts/01_data_validation.py
python mlops_pipeline/scripts/02_data_preprocessing.py
# สังเกต run_id จากขั้นตอนก่อนหน้า (ดูใน MLflow UI ก็ได้)
python mlops_pipeline/scripts/03_train_evaluate_register.py <RUN_ID_PREPROCESS> 1.0
python mlops_pipeline/scripts/04_transition_model.py "mental-health-classifier" "Staging"
```

### เสิร์ฟโมเดลและทดสอบ (API)
```bash
mlflow models serve -m "models:/mental-health-classifier/Staging" -p 5001 --env-manager local
python mlops_pipeline/scripts/06_client_test.py
```
### WEB api 
'''
mlflow models serve -m "models:/mental-health-classifier/Staging" -p 5001 --env-manager local



curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d "{\"dataframe_records\":[{\"text\":\"I feel extremely anxious about my future.\"},{\"text\":\"Life is beautiful and I am calm today.\"}]}"
'''


## CI/CT/CD (GitHub Actions)
- เติมค่า `secrets` ให้ repo:
  - `MLFLOW_TRACKING_URI`
  - `MLFLOW_TRACKING_USERNAME`
  - `MLFLOW_TRACKING_PASSWORD`
- workflow จะรัน 01→02→03→04 เมื่อ push ไป `main`

## หมายเหตุ
- โมเดลตัวอย่างใช้ `TF-IDF + LogisticRegression` (balanced class_weight) สำหรับโจทย์ 7 คลาส
- เกณฑ์คัดเลือกเบื้องต้น: Accuracy ≥ 0.70 และ Macro-F1 ≥ 0.70 (ปรับได้ตามจริง)
- ไฟล์ผลการประเมิน (`classification_report.json`, `confusion_matrix.png`) จะถูกเก็บเป็น MLflow artifacts
