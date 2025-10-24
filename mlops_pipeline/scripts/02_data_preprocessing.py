import os, json, hashlib, io
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

MLFLOW_EXPERIMENT = "MentalHealth - Data Preprocessing"
MODEL_NAME = "mental-health-classifier"

# --- Fallback clean_text ---
try:
    from common_text import clean_text as _external_clean_text
    def clean_text(s: str) -> str:
        return _external_clean_text(s)
except Exception:
    import re
    def clean_text(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = s.replace("\u200b", "")  # zero-width
        s = re.sub(r"\s+", " ", s).strip()
        return s

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main(test_size=0.2, val_size=0.1, random_state=42):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")

        data_csv = os.getenv("DATA_CSV", "data/mental_health.csv")
        data_csv_path = Path(data_csv)
        if not data_csv_path.exists():
            alt = Path("/mnt/data/mental_health.csv")
            if alt.exists():
                data_csv_path = alt
        df = pd.read_csv(data_csv_path)

        # hash ของไฟล์
        try:
            data_hash = sha256_of_file(data_csv_path)
            mlflow.log_param("data_version_sha256", data_hash)
        except Exception:
            pass

        # normalize column names
        cols_lower = {c.lower(): c for c in df.columns}
        text_col = cols_lower.get("text") or cols_lower.get("statement") or list(df.columns)[0]
        label_col = cols_lower.get("label") or cols_lower.get("status") or list(df.columns)[1]
        df = df.rename(columns={text_col: "text", label_col: "label"})

        # basic cleaning
        raw_rows = len(df)
        df["text"] = df["text"].fillna("").astype(str).apply(clean_text)

        # drop obviously bad tokens
        bad_tokens = {"", "nan", "none", "null"}
        mask_bad = df["text"].str.strip().str.lower().isin(bad_tokens)
        n_bad = int(mask_bad.sum())
        df = df[~mask_bad]

        # drop NaN labels
        n_label_na = int(df["label"].isna().sum())
        df = df[df["label"].notna()].copy()

        # drop duplicate texts (optional แต่ดีต่อคุณภาพ)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["text"])
        n_dupes = before_dedup - len(df)

        # label mapping
        labels = sorted(df["label"].unique().tolist())
        label2id = {c: i for i, c in enumerate(labels)}
        id2label = {i: c for c, i in label2id.items()}
        df["label_id"] = df["label"].map(label2id)

        # log distributions BEFORE split
        label_counts = df["label"].value_counts().to_dict()
        mlflow.log_dict(label_counts, "pre_split_label_counts.json")
        mlflow.log_param("n_classes", len(labels))

        # --- stratify safety check ---
        too_small = {k: v for k, v in label_counts.items() if v < 2}
        if len(too_small) > 0:
            drop_idxs = df["label"].isin(list(too_small.keys()))
            mlflow.log_dict(too_small, "dropped_tiny_classes.json")
            df = df[~drop_idxs].copy()

        # split
        train_df, test_df = train_test_split(
            df[["text", "label_id"]],
            test_size=test_size,
            random_state=random_state,
            stratify=df["label_id"]
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=random_state,
            stratify=train_df["label_id"]
        )

        # out dir
        outdir = Path("processed_data")
        outdir.mkdir(exist_ok=True, parents=True)
        train_df.to_csv(outdir / "train.csv", index=False)
        val_df.to_csv(outdir / "val.csv", index=False)
        test_df.to_csv(outdir / "test.csv", index=False)

        with open(outdir / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

        # save a small sample for human inspection
        sample_lines = (
            train_df.head(50)
            .assign(label=train_df["label_id"].map(id2label))
            [["text", "label", "label_id"]]
            .to_dict(orient="records")
        )
        with open(outdir / "sample.jsonl", "w", encoding="utf-8") as f:
            for rec in sample_lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # log split distributions
        def counts(df_):
            return df_["label_id"].value_counts().sort_index().to_dict()

        mlflow.log_dict(counts(train_df), "split_counts_train.json")
        mlflow.log_dict(counts(val_df),   "split_counts_val.json")
        mlflow.log_dict(counts(test_df),  "split_counts_test.json")

        # params/metrics
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("raw_rows", raw_rows)
        mlflow.log_metric("rows_after_clean", len(df))
        mlflow.log_metric("dropped_bad_text", n_bad)
        mlflow.log_metric("dropped_label_na", n_label_na)
        mlflow.log_metric("dropped_duplicates", n_dupes)
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("val_rows", len(val_df))
        mlflow.log_metric("test_rows", len(test_df))

        # artifacts
        mlflow.log_artifacts(str(outdir), artifact_path="processed_data")

        print(f"[PREPROCESS] run_id={run_id}")
        print(f"[PREPROCESS] labels={labels}")
        print("[PREPROCESS] processed_data/: train.csv, val.csv, test.csv, label_mapping.json, sample.jsonl")

if __name__ == "__main__":
    main()
