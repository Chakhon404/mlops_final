import sys, os
import mlflow, mlflow.sklearn
from mlflow.artifacts import download_artifacts
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

def train_evaluate_register(preprocessing_run_id, C=2.0, max_features=20000):
    ACCURACY_THRESHOLD = 0.75
    mlflow.set_experiment("MH - Model Training")
    with mlflow.start_run(run_name=f"svm_tfidf_C_{C}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        local_artifact_path = download_artifacts(run_id=preprocessing_run_id, artifact_path="processed_data")
        train_df = pd.read_csv(os.path.join(local_artifact_path, "train.csv"))
        test_df  = pd.read_csv(os.path.join(local_artifact_path, "test.csv"))

        Xtr, ytr = train_df["text"].astype(str), train_df["label"].astype(int)
        Xte, yte = test_df["text"].astype(str),  test_df["label"].astype(int)

        pipe = Pipeline([
            ("feat", FeatureUnion([
                ("w", TfidfVectorizer(
                    ngram_range=(1,2), max_features=int(max_features), min_df=2, max_df=0.95,
                    sublinear_tf=True, strip_accents="unicode", lowercase=True
                )),
                ("c", TfidfVectorizer(
                    analyzer="char", ngram_range=(3,5), min_df=2, sublinear_tf=True
                ))
            ])),
            ("clf", LinearSVC(C=float(C), random_state=42))
        ])

        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        acc = accuracy_score(yte, yhat)
        f1m = f1_score(yte, yhat, average="macro")
        print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")

        mlflow.log_param("C", C)
        mlflow.log_param("max_features", max_features)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1m)
        mlflow.sklearn.log_model(pipe, "mh_text_pipeline")

        if acc >= ACCURACY_THRESHOLD:
            uri = f"runs:/{mlflow.active_run().info.run_id}/mh_text_pipeline"
            rm = mlflow.register_model(uri, "MH-classifier-prod")
            print(f"Registered as {rm.name} v{rm.version}")
        else:
            print(f"Below threshold ({ACCURACY_THRESHOLD}). Not registering.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mlops_pipeline/scripts/03_train_evaluate_register.py <preprocessing_run_id> [C] [max_features]")
        sys.exit(1)
    rid = sys.argv[1]
    C = float(sys.argv[2]) if len(sys.argv)>2 else 2.0
    MF = int(sys.argv[3]) if len(sys.argv)>3 else 200000
    train_evaluate_register(rid, C, MF)
