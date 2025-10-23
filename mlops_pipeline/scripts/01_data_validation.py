import os, json, re
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from typing import Dict, Any

MLFLOW_EXPERIMENT = "MentalHealth - Data Validation"

# ใช้ค่าเริ่มต้นได้ และอนุญาต override ผ่าน ENV (เช่น ALLOWED_LABELS='["A","B"]')
DEFAULT_ALLOWED_LABELS = {"Anxiety","Bipolar","Depression","Normal","Personality disorder","Stress","Suicidal"}

# เกณฑ์ความยาวข้อความ (หน่วย = จำนวน token แบบเว้นวรรค)
MIN_TOKENS = int(os.getenv("MIN_TOKENS", 2))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 200))

def _load_allowed_labels() -> set:
    raw = os.getenv("ALLOWED_LABELS", "").strip()
    if not raw:
        return DEFAULT_ALLOWED_LABELS
    try:
        parsed = json.loads(raw)
        return set(parsed)
    except Exception:
        # รองรับคั่นด้วยจุลภาคแบบง่าย
        return set(x.strip() for x in raw.split(",") if x.strip())

def _safe_ratio(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0

def main():
    base_dir = os.getcwd()
    tracking_path = os.path.join(base_dir, "mlruns_temp")
    os.makedirs(tracking_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_path}")
    os.makedirs("validation", exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")

        data_csv = os.getenv("DATA_CSV", "data/mental_health.csv")
        df = pd.read_csv(data_csv)
        mlflow.log_param("data_csv", data_csv)

        # -----------------------------
        # Normalize column names → text, label
        # -----------------------------
        cols_lower = {c.lower(): c for c in df.columns}
        text_col = cols_lower.get("text") or cols_lower.get("statement") or list(df.columns)[0]
        label_col = cols_lower.get("label") or cols_lower.get("status") or list(df.columns)[1]
        df = df.rename(columns={text_col: "text", label_col: "label"})
        mlflow.log_param("mapped_text_col", text_col)
        mlflow.log_param("mapped_label_col", label_col)

        # -----------------------------
        # Basic stats
        # -----------------------------
        num_rows, num_cols = df.shape
        num_missing_text = df["text"].isna().sum()
        num_missing_label = df["label"].isna().sum()
        empty_after_strip = (df["text"].astype(str).str.strip() == "").sum()
        nunique_labels = df["label"].nunique(dropna=True)

        mlflow.log_metric("num_rows", int(num_rows))
        mlflow.log_metric("num_cols", int(num_cols))
        mlflow.log_metric("missing_text", int(num_missing_text))
        mlflow.log_metric("missing_label", int(num_missing_label))
        mlflow.log_metric("empty_after_strip", int(empty_after_strip))
        mlflow.log_param("nunique_labels", int(nunique_labels))

        # -----------------------------
        # Text length (tokens) stats
        # -----------------------------
        # แปลง NaN เป็นว่างเฉพาะเพื่อคำนวณความยาว
        text_series = df["text"].fillna("").astype(str)
        token_len = text_series.str.split().apply(len)

        length_stats: Dict[str, Any] = {
            "min": int(token_len.min() if len(token_len) else 0),
            "p25": float(token_len.quantile(0.25)) if len(token_len) else 0.0,
            "median": float(token_len.median()) if len(token_len) else 0.0,
            "p75": float(token_len.quantile(0.75)) if len(token_len) else 0.0,
            "max": int(token_len.max() if len(token_len) else 0),
            "mean": float(token_len.mean() if len(token_len) else 0.0),
            "std": float(token_len.std(ddof=0) if len(token_len) else 0.0),
        }
        with open("text_length_stats.json", "w", encoding="utf-8") as f:
            json.dump(length_stats, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact("text_length_stats.json", artifact_path="validation")

        # Histogram ความยาว
        plt.figure(figsize=(8,4))
        plt.hist(token_len, bins=40)
        plt.title("Text Length (tokens) Histogram")
        plt.xlabel("tokens")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig("text_length_hist.png", dpi=150)
        mlflow.log_artifact("text_length_hist.png", artifact_path="validation")

        # -----------------------------
        # Label distribution & imbalance
        # -----------------------------
        label_counts = df["label"].value_counts(dropna=True).sort_values(ascending=False)
        label_counts.to_csv("label_counts.csv")
        mlflow.log_artifact("label_counts.csv", artifact_path="validation")

        plt.figure(figsize=(8,4))
        label_counts.plot(kind="bar")
        plt.title("Label Distribution")
        plt.tight_layout()
        plt.savefig("label_distribution.png", dpi=150)
        mlflow.log_artifact("label_distribution.png", artifact_path="validation")

        majority_prop = _safe_ratio(int(label_counts.max() if len(label_counts) else 0), int(label_counts.sum() if len(label_counts) else 0))
        mlflow.log_metric("majority_class_proportion", majority_prop)

        # -----------------------------
        # Anomalies detection
        # -----------------------------
        anomalies = []

        # 1) missing / empty
        if num_missing_text > 0 or empty_after_strip > 0:
            anomalies.append("empty_text")
        if num_missing_label > 0:
            anomalies.append("missing_label")

        # 2) too few classes
        if nunique_labels < 5:
            anomalies.append("too_few_classes")

        # 3) unknown labels (นอกเหนือ allowed)
        allowed = _load_allowed_labels()
        unknown_labels = sorted(list(set(df["label"].dropna().unique()) - set(allowed)))
        if unknown_labels:
            anomalies.append("unknown_labels")
            with open("unknown_labels.json", "w", encoding="utf-8") as f:
                json.dump({"unknown_labels": unknown_labels}, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact("unknown_labels.json", artifact_path="validation")
        mlflow.log_param("allowed_labels_used", ",".join(sorted(list(allowed))))

        # 4) duplicates
        dup_text = df["text"].duplicated(keep=False)
        dup_pair = df[["text","label"]].duplicated(keep=False)
        num_dup_text = int(dup_text.sum())
        num_dup_pair = int(dup_pair.sum())
        mlflow.log_metric("duplicate_text_rows", num_dup_text)
        mlflow.log_metric("duplicate_text_label_rows", num_dup_pair)
        if num_dup_text > 0:
            anomalies.append("duplicate_text")
        if num_dup_pair > 0:
            anomalies.append("duplicate_text_label")

        # 5) text length out-of-range
        too_short = int((token_len < MIN_TOKENS).sum())
        too_long  = int((token_len > MAX_TOKENS).sum())
        mlflow.log_metric("too_short_text_rows", too_short)
        mlflow.log_metric("too_long_text_rows", too_long)
        if too_short > 0:
            anomalies.append("too_short_text")
        if too_long > 0:
            anomalies.append("too_long_text")

        # 6) strong imbalance (เช่น majority > 60%)
        if majority_prop >= 0.60 and nunique_labels >= 2:
            anomalies.append("class_imbalance_gt_60pct")

        # -----------------------------
        # ออกรายงาน anomalies + ตัวอย่างแถวผิดปกติ
        # -----------------------------
        bad_mask = (
            df["text"].isna() |
            (df["text"].astype(str).str.strip() == "") |
            df["label"].isna() |
            (token_len < MIN_TOKENS) |
            (token_len > MAX_TOKENS) |
            dup_text
        )
        bad_df = df.loc[bad_mask].copy()
        # ไม่ให้ไฟล์ใหญ่เกินไป
        sample_rows = min(2000, len(bad_df))
        bad_df.head(sample_rows).to_csv("anomaly_rows_sample.csv", index=False)
        mlflow.log_artifact("anomaly_rows_sample.csv", artifact_path="validation")

        summary = {
            "rows": int(num_rows),
            "cols": int(num_cols),
            "nunique_labels": int(nunique_labels),
            "missing_text": int(num_missing_text),
            "missing_label": int(num_missing_label),
            "empty_after_strip": int(empty_after_strip),
            "duplicate_text_rows": num_dup_text,
            "duplicate_text_label_rows": num_dup_pair,
            "too_short_text_rows": too_short,
            "too_long_text_rows": too_long,
            "majority_class_proportion": majority_prop,
            "anomalies": anomalies
        }
        with open("anomalies.json","w",encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact("anomalies.json", artifact_path="validation")

        # schema (ชนิดข้อมูล + allowed labels)
        schema = {
            "columns": {c: str(df[c].dtype) for c in df.columns},
            "required": ["text","label"],
            "allowed_labels": sorted(list(allowed)),
            "length_rules": {"min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS}
        }
        with open("schema.json","w",encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact("schema.json", artifact_path="validation")

        mlflow.log_param("validation_status", "Failed" if anomalies else "Success")
        print(f"[VALIDATION] rows={num_rows}, labels={nunique_labels}, anomalies={anomalies}")

if __name__ == "__main__":
    main()
