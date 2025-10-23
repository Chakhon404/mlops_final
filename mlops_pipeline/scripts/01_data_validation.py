import os, json
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from typing import Dict, Any

# =========================
# CONFIG
# =========================
MLFLOW_EXPERIMENT = "MentalHealth - Data Validation"
ARTIFACT_DIR = "artifacts/validation"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DEFAULT_ALLOWED_LABELS = {
    "Anxiety", "Bipolar", "Depression",
    "Normal", "Personality disorder", "Stress", "Suicidal"
}

MIN_TOKENS = int(os.getenv("MIN_TOKENS", 2))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 200))


# =========================
# Helper Functions
# =========================
def _load_allowed_labels() -> set:
    raw = os.getenv("ALLOWED_LABELS", "").strip()
    if not raw:
        return DEFAULT_ALLOWED_LABELS
    try:
        parsed = json.loads(raw)
        return set(parsed)
    except Exception:
        return set(x.strip() for x in raw.split(",") if x.strip())


def _safe_ratio(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


# =========================
# Main
# =========================
def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")

        data_csv = os.getenv("DATA_CSV", "data/mental_health.csv")
        df = pd.read_csv(data_csv)
        mlflow.log_param("data_csv", data_csv)

        # Normalize column names → text, label
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

        mlflow.log_metrics({
            "num_rows": num_rows,
            "num_cols": num_cols,
            "missing_text": num_missing_text,
            "missing_label": num_missing_label,
            "empty_after_strip": empty_after_strip,
        })
        mlflow.log_param("nunique_labels", int(nunique_labels))

        # -----------------------------
        # Text length (tokens) stats
        # -----------------------------
        text_series = df["text"].fillna("").astype(str)
        token_len = text_series.str.split().apply(len)

        length_stats: Dict[str, Any] = {
            "min": int(token_len.min()),
            "p25": float(token_len.quantile(0.25)),
            "median": float(token_len.median()),
            "p75": float(token_len.quantile(0.75)),
            "max": int(token_len.max()),
            "mean": float(token_len.mean()),
            "std": float(token_len.std(ddof=0)),
        }
        path_stats = os.path.join(ARTIFACT_DIR, "text_length_stats.json")
        with open(path_stats, "w", encoding="utf-8") as f:
            json.dump(length_stats, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(path_stats, artifact_path="validation")

        # Histogram
        plt.figure(figsize=(8, 4))
        plt.hist(token_len, bins=40)
        plt.title("Text Length (tokens) Histogram")
        plt.xlabel("tokens")
        plt.ylabel("count")
        plt.tight_layout()
        path_hist = os.path.join(ARTIFACT_DIR, "text_length_hist.png")
        plt.savefig(path_hist, dpi=150)
        mlflow.log_artifact(path_hist, artifact_path="validation")

        # -----------------------------
        # Label distribution
        # -----------------------------
        label_counts = df["label"].value_counts(dropna=True).sort_values(ascending=False)
        path_label_csv = os.path.join(ARTIFACT_DIR, "label_counts.csv")
        label_counts.to_csv(path_label_csv)
        mlflow.log_artifact(path_label_csv, artifact_path="validation")

        plt.figure(figsize=(8, 4))
        label_counts.plot(kind="bar")
        plt.title("Label Distribution")
        plt.tight_layout()
        path_label_plot = os.path.join(ARTIFACT_DIR, "label_distribution.png")
        plt.savefig(path_label_plot, dpi=150)
        mlflow.log_artifact(path_label_plot, artifact_path="validation")

        majority_prop = _safe_ratio(label_counts.max(), label_counts.sum())
        mlflow.log_metric("majority_class_proportion", majority_prop)

        # -----------------------------
        # Anomalies detection
        # -----------------------------
        anomalies = []
        allowed = _load_allowed_labels()

        # Missing / Empty
        if num_missing_text > 0 or empty_after_strip > 0:
            anomalies.append("empty_text")
        if num_missing_label > 0:
            anomalies.append("missing_label")

        # Too few classes
        if nunique_labels < 5:
            anomalies.append("too_few_classes")

        # Unknown labels
        unknown_labels = sorted(list(set(df["label"].dropna().unique()) - set(allowed)))
        if unknown_labels:
            anomalies.append("unknown_labels")
            path_unknown = os.path.join(ARTIFACT_DIR, "unknown_labels.json")
            with open(path_unknown, "w", encoding="utf-8") as f:
                json.dump({"unknown_labels": unknown_labels}, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(path_unknown, artifact_path="validation")

        mlflow.log_param("allowed_labels_used", ",".join(sorted(list(allowed))))

        # Duplicates
        dup_text = df["text"].duplicated(keep=False)
        dup_pair = df[["text", "label"]].duplicated(keep=False)
        num_dup_text, num_dup_pair = int(dup_text.sum()), int(dup_pair.sum())
        mlflow.log_metrics({
            "duplicate_text_rows": num_dup_text,
            "duplicate_text_label_rows": num_dup_pair,
        })
        if num_dup_text > 0:
            anomalies.append("duplicate_text")
        if num_dup_pair > 0:
            anomalies.append("duplicate_text_label")

        # Text length out-of-range
        too_short = int((token_len < MIN_TOKENS).sum())
        too_long = int((token_len > MAX_TOKENS).sum())
        mlflow.log_metrics({
            "too_short_text_rows": too_short,
            "too_long_text_rows": too_long,
        })
        if too_short > 0:
            anomalies.append("too_short_text")
        if too_long > 0:
            anomalies.append("too_long_text")

        # Imbalance
        if majority_prop >= 0.60 and nunique_labels >= 2:
            anomalies.append("class_imbalance_gt_60pct")

        # -----------------------------
        # Export anomalies & schema
        # -----------------------------
        bad_mask = (
            df["text"].isna()
            | (df["text"].astype(str).str.strip() == "")
            | df["label"].isna()
            | (token_len < MIN_TOKENS)
            | (token_len > MAX_TOKENS)
            | dup_text
        )
        bad_df = df.loc[bad_mask].copy()
        bad_sample_path = os.path.join(ARTIFACT_DIR, "anomaly_rows_sample.csv")
        bad_df.head(2000).to_csv(bad_sample_path, index=False)
        mlflow.log_artifact(bad_sample_path, artifact_path="validation")

        summary = {
            "rows": num_rows,
            "cols": num_cols,
            "nunique_labels": nunique_labels,
            "anomalies": anomalies,
            "majority_class_proportion": majority_prop,
        }
        path_anomalies = os.path.join(ARTIFACT_DIR, "anomalies.json")
        with open(path_anomalies, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(path_anomalies, artifact_path="validation")

        schema = {
            "columns": {c: str(df[c].dtype) for c in df.columns},
            "required": ["text", "label"],
            "allowed_labels": sorted(list(allowed)),
            "length_rules": {"min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS},
        }
        path_schema = os.path.join(ARTIFACT_DIR, "schema.json")
        with open(path_schema, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(path_schema, artifact_path="validation")

        mlflow.log_param("validation_status", "Failed" if anomalies else "Success")
        print(f"[VALIDATION] rows={num_rows}, labels={nunique_labels}, anomalies={anomalies}")


if __name__ == "__main__":
    main()
