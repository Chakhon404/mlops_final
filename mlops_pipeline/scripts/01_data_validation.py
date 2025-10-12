# mlops_pipeline/scripts/01_data_validation.py
import os, sys, re, json
import mlflow
import pandas as pd
from collections import Counter

# default path (ถ้าไม่ส่งอาร์กิวเมนต์)
MLOPS_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DATA_CSV = os.path.join(MLOPS_DIR, "data", "data.csv")

REQUIRED_COLS = ["statement", "status"]
VALID_LABELS = {
    "anxiety","bipolar","depression","normal",
    "personality disorder","stress","suicidal"
}

def validate_csv(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {csv_path}")
    df = pd.read_csv(csv_path)

    report, problems = {}, []

    # 1) columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        problems.append({"missing_columns": missing})

    # ทำงานต่อได้แม้ขาดคอลัมน์ เพื่อออกรายงานให้มากที่สุด
    safe = df.copy()
    for c in REQUIRED_COLS:
        if c not in safe.columns:
            safe[c] = None

    # 2) NA
    na_counts = safe[REQUIRED_COLS].isna().sum().to_dict()

    # 3) labels
    labels_found = set(safe["status"].dropna().astype(str).str.lower().str.strip())
    unknown = sorted(list(labels_found - VALID_LABELS))
    if unknown:
        problems.append({"unknown_labels": unknown})

    # 4) text length
    lens = safe["statement"].astype(str).str.split().map(len)
    short_ratio = float((lens < 3).mean())

    # 5) duplicates
    dup_count = int(safe.duplicated(subset=["statement"]).sum())

    # 6) simple noise signals
    url_pat = re.compile(r"https?://\S+|www\.\S+")
    url_ratio = float(safe["statement"].astype(str).map(lambda s: bool(url_pat.search(s))).mean())
    ctrl_ratio = float(safe["statement"].astype(str).map(lambda s: ("\u200d" in s or "\u2060" in s)).mean())

    # 7) label distribution
    dist = dict(Counter(safe["status"].dropna().astype(str).str.lower().str.strip()))

    # สถานะรวม
    status = "Success"
    if missing: status = "Check"
    if unknown: status = "Check"
    if len(labels_found) < 7:  # ขาดคลาส
        status = "Failed: labels<7"

    report.update({
        "csv_path": csv_path,
        "rows": int(len(safe)),
        "na_counts": na_counts,
        "labels_found": sorted(list(labels_found)),
        "label_distribution": dist,
        "short_text_ratio_lt3": short_ratio,
        "duplicate_texts": dup_count,
        "url_ratio": url_ratio,
        "control_char_ratio": ctrl_ratio,
        "problems": problems,
        "status": status,
        "expected_labels": sorted(list(VALID_LABELS)),
        "required_columns": REQUIRED_COLS,
    })
    return report

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_CSV

    mlflow.set_experiment("MH - Data Validation")
    with mlflow.start_run(run_name=f"validate_raw_csv:{os.path.basename(csv_path)}"):
        mlflow.set_tag("ml.step", "data_validation")
        rep = validate_csv(csv_path)

        # log metrics
        mlflow.log_metric("rows", rep["rows"])
        mlflow.log_metric("na_statement", rep["na_counts"].get("statement", 0))
        mlflow.log_metric("na_status", rep["na_counts"].get("status", 0))
        mlflow.log_metric("unique_labels", len(rep["labels_found"]))
        mlflow.log_metric("short_text_ratio_lt3", rep["short_text_ratio_lt3"])
        mlflow.log_metric("duplicate_texts", rep["duplicate_texts"])
        mlflow.log_metric("url_ratio", rep["url_ratio"])
        mlflow.log_metric("control_char_ratio", rep["control_char_ratio"])

        # log params
        mlflow.log_param("expected_labels", ",".join(rep["expected_labels"]))
        mlflow.log_param("required_columns", ",".join(REQUIRED_COLS))
        mlflow.log_param("labels_found", ",".join(rep["labels_found"]))
        mlflow.log_param("validation_status", rep["status"])

        # save + log report artifact
        os.makedirs("validation_artifacts", exist_ok=True)
        out = os.path.join("validation_artifacts", "validation_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(out, artifact_path="validation")

        # console summary
        print(json.dumps(rep, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Validation error:", e)
        sys.exit(1)
