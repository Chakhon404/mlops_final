# 03_train_evaluate_register.py
import sys, os, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment import SentimentIntensityAnalyzer

# ===== Config =====
EXPERIMENT = "MentalHealth - Model Training"
MODEL_NAME = "mental-health-classifier"
ACCURACY_THRESHOLD = 0.70
MACROF1_THRESHOLD = 0.70
RANDOM_STATE = 42

# -----------------------------
# Helper: แปลง DataFrame → list ของข้อความ
# -----------------------------
def coerce_to_1d_text(X):
    import pandas as pd, numpy as np
    if isinstance(X, pd.DataFrame):
        if "text" in X.columns:
            return X["text"].astype(str).values
        elif X.shape[1] == 1:
            return X.iloc[:, 0].astype(str).values
        else:
            raise ValueError(f"Expected 1 text column, got columns: {list(X.columns)}")
    if isinstance(X, pd.Series):
        return X.astype(str).values
    if isinstance(X, (list, np.ndarray)):
        return np.asarray(X, dtype=str)
    return np.asarray(X, dtype=str)

# -----------------------------
# Feature Extractor: Sentiment Score
# -----------------------------
class SentimentScore(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1.0):
        self.weight = weight
        self.analyzer = SentimentIntensityAnalyzer()
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame): texts = X["text"].astype(str)
        else: texts = pd.Series(X).astype(str)
        scores = texts.apply(lambda t: self.analyzer.polarity_scores(t)["compound"])
        return (scores * self.weight).to_numpy().reshape(-1, 1)

# -----------------------------
# Helper: วาด Confusion Matrix แล้ว log เข้า MLflow
# -----------------------------
def _plot_and_log_cm(cm, labels_sorted, name_png, title, artifact_dir="evaluation"):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=45, ha="right")
    plt.yticks(range(len(labels_sorted)), labels_sorted)
    plt.tight_layout()
    plt.savefig(name_png, dpi=150)
    plt.close()
    mlflow.log_artifact(name_png, artifact_path=artifact_dir)

# -----------------------------
# Main Function
# -----------------------------
def train_evaluate_register(preprocessing_run_id: str,
                            C: float = 1.0,
                            analyzer: str = "word",
                            ngram_range: tuple = (1, 2),
                            max_features: int = 60000,
                            sentiment_weight: float = 1.0):
    """
    Train TF-IDF + SentimentScore + LinearSVC pipeline
    พร้อมรองรับการปรับ Hyperparameters ผ่าน CLI
    """
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"tfidf_sentiment_C_{C}"):

        print(f"=== Start training with params ===")
        print(f"C={C}, analyzer={analyzer}, ngram_range={ngram_range}, "
              f"max_features={max_features}, sentiment_weight={sentiment_weight}")
        mlflow.set_tag("ml.step", "model_training_evaluation")

        # log hyperparams
        mlflow.log_param("C", C)
        mlflow.log_param("analyzer", analyzer)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("sentiment_weight", sentiment_weight)
        mlflow.log_param("clf", "LinearSVC_balanced")

        # -----------------------------
        # 1) โหลดข้อมูลจาก preprocessing artifacts
        # -----------------------------
        local_artifact_path = download_artifacts(
            run_id=preprocessing_run_id,
            artifact_path="processed_data"
        )
        print(f"Artifacts downloaded to: {local_artifact_path}")

        train_df = pd.read_csv(os.path.join(local_artifact_path, "train.csv"), keep_default_na=False)
        val_df   = pd.read_csv(os.path.join(local_artifact_path, "val.csv"), keep_default_na=False)
        test_df  = pd.read_csv(os.path.join(local_artifact_path, "test.csv"), keep_default_na=False)

        with open(os.path.join(local_artifact_path, "label_mapping.json"), "r", encoding="utf-8") as f:
            mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}

        X_train, y_train_id = train_df["text"], train_df["label_id"]
        X_val, y_val_id     = val_df["text"], val_df["label_id"]
        X_test, y_test_id   = test_df["text"], test_df["label_id"]

        y_train = y_train_id.map(id2label)
        y_val   = y_val_id.map(id2label)
        y_test  = y_test_id.map(id2label)
        X_trainval = pd.concat([X_train, X_val], ignore_index=True)
        y_trainval = pd.concat([y_train, y_val], ignore_index=True)

        # -----------------------------
        # 2) Pipeline: TF-IDF + SentimentScore + LinearSVC
        # -----------------------------
        pipe = Pipeline([
            ("coerce_text", FunctionTransformer(coerce_to_1d_text, validate=False)),
            ("features", FeatureUnion([
                ("tfidf", TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=True,
                    smooth_idf=True,
                    dtype=np.float32,
                )),
                ("sentiment", Pipeline([
                    ("senti", SentimentScore(weight=sentiment_weight)),
                    ("scaler", StandardScaler())
                ]))
            ])),
            ("clf", LinearSVC(C=C, class_weight='balanced', random_state=RANDOM_STATE))
        ])

        pipe.fit(X_trainval, y_trainval)

        # -----------------------------
        # 3) Evaluate
        # -----------------------------
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_macro_f1", f1m)
        print(f"[EVAL] test_accuracy={acc:.4f}, test_macro_f1={f1m:.4f}")

        # -----------------------------
        # 4) Log reports + artifacts
        # -----------------------------
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        with open("classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact("classification_report.json", artifact_path="evaluation")

        labels_sorted = sorted(set(list(y_test) + list(y_pred)))
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        _plot_and_log_cm(cm, labels_sorted, "confusion_matrix.png", "Confusion Matrix")
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        _plot_and_log_cm(cm_norm, labels_sorted, "confusion_matrix_normalized.png", "Confusion Matrix (normalized)")

        # -----------------------------
        # 5) Log model + register
        # -----------------------------
        input_example = pd.DataFrame({"text": ["I feel stressed and anxious about exams"]})
        mlflow.sklearn.log_model(pipe, artifact_path="model",
                                 input_example=input_example,
                                 registered_model_name=None)

        shutil.copy(os.path.join(local_artifact_path, "label_mapping.json"), "label_mapping.json")
        mlflow.log_artifact("label_mapping.json", artifact_path="preprocessing")

        if (acc >= ACCURACY_THRESHOLD) and (f1m >= MACROF1_THRESHOLD):
            print("[REGISTER] Passed threshold -> registering model...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            registered_model = mlflow.register_model(model_uri, MODEL_NAME)
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
        else:
            print("[REGISTER] Below threshold -> not registering.")

        print("Training run finished.")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [C] [analyzer] [ngram_min,ngram_max] [max_features] [sentiment_weight]")
        sys.exit(1)

    run_id = sys.argv[1]
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    analyzer = sys.argv[3] if len(sys.argv) > 3 else "word"

    # ✅ แปลง "(1,4)" -> (1, 4) อย่างปลอดภัย
    if len(sys.argv) > 4:
        ng_min, ng_max = sys.argv[4].strip("()").split(",")
        ngram_range = (int(ng_min), int(ng_max))
    else:
        ngram_range = (1, 2)

    max_features = int(sys.argv[5]) if len(sys.argv) > 5 else 60000
    sentiment_weight = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

    train_evaluate_register(run_id, C, analyzer, ngram_range, max_features, sentiment_weight)
