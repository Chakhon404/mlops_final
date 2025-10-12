import os, re, sys, subprocess
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

MLOPS_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MLOPS_DIR, "data")
RAW_CSV  = os.path.join(DATA_DIR, "data.csv")
PROC_DIR = os.path.join(MLOPS_DIR, "processed_data")
TRAIN_OUT = os.path.join(PROC_DIR, "train.csv")
TEST_OUT  = os.path.join(PROC_DIR, "test.csv")

LABELS = ["Anxiety","Bipolar","Depression","Normal","Personality Disorder","Stress","Suicidal"]
LABEL_MAP = {k.lower(): i for i,k in enumerate(LABELS)}

def clean_text(s:str)->str:
    s = s or ""
    s = s.replace("\u200d"," ").replace("\u2060"," ")
    s = re.sub(r"https?://\S+|www\.\S+"," ", s)
    s = re.sub(r"<[^>]+>"," ", s)
    s = re.sub(r"[^\w\s\.,!?'\-]"," ", s)
    s = re.sub(r"\s+"," ", s).strip().lower()
    return s

def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV):
        return
    try:
        cmd = ["kaggle","datasets","download","-d","suchintikasarkar/sentiment-analysis-for-mental-health","-p", DATA_DIR]
        subprocess.run(cmd, check=True)
        import glob, zipfile
        zips = sorted(glob.glob(os.path.join(DATA_DIR,"*.zip")), key=os.path.getmtime, reverse=True)
        with zipfile.ZipFile(zips[0], "r") as zf:
            zf.extractall(DATA_DIR)
    except Exception as e:
        print("[WARN] Kaggle API failed:", e, file=sys.stderr)
        if not os.path.exists(RAW_CSV):
            raise FileNotFoundError("กรุณาวาง data.csv ไว้ที่ mlops_pipeline/data/ หรือเตรียม Kaggle API ให้พร้อม")

def main():
    mlflow.set_experiment("MH - Data Preprocessing")
    with mlflow.start_run(run_name="clean_split_save") as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step","data_preprocessing")
        ensure_data()

        df = pd.read_csv(RAW_CSV)
        if not {"statement","status"}.issubset(df.columns):
            raise KeyError("ต้องมีคอลัมน์ 'statement' และ 'status' ใน data.csv")
        df = df[["statement","status"]].dropna()
        df["status_norm"] = df["status"].astype(str).str.lower().str.strip()
        df = df[df["status_norm"].isin(LABEL_MAP.keys())]
        df["text"] = df["statement"].astype(str).map(clean_text)
        df = df[df["text"].str.split().str.len() >= 3].drop_duplicates(subset=["text","status_norm"])
        df["label"] = df["status_norm"].map(LABEL_MAP)

        train, test = train_test_split(df[["text","label"]], test_size=0.2, random_state=42, stratify=df["label"])

        os.makedirs(PROC_DIR, exist_ok=True)
        train.to_csv(TRAIN_OUT, index=False)
        test.to_csv(TEST_OUT, index=False)

        mlflow.log_metric("training_set_rows", len(train))
        mlflow.log_metric("test_set_rows", len(test))
        mlflow.log_param("num_classes", len(LABELS))
        mlflow.log_artifacts(PROC_DIR, artifact_path="processed_data")
        print(f"Preprocessing Run ID: {run_id}")
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)

if __name__ == "__main__":
    main()
