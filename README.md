pip install fastapi uvicorn pandas joblib

uvicorn app:app --reload --port 5001

curl -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" -d "{\"text\":\"I feel extremely anxious about my future.\"}"

mlflow models serve -m "models:/mental-health-classifier/Staging" -p 5001 --env-manager local
