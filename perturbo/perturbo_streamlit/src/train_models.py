# src/train_models.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.utils import load_npy, train_ae, save_joblib

def train_models(
    emb_path="data/e5_embeddings.npy",
    feat_csv="data/tripadvisor_hybrid_features.csv",
    out_dir="models",
    random_state=42
):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading embeddings and features...")
    emb = load_npy(emb_path)
    df = pd.read_csv(feat_csv)

    if "label" not in df.columns:
        raise ValueError("Feature CSV must contain 'label' column with values 'clean'/'perturbed'.")

    # binary label mapping: perturbed -> 1, clean -> 0
    y = (df["label"].astype(str).str.lower() == "perturbed").astype(int).values

    # optional extra features
    feat_cols = [c for c in ["reconstruction_error","latent_distance","anomaly_score",
                             "adverb_count","adverb_weight","avg_adverb_weight"] if c in df.columns]
    X_extra = df[feat_cols].fillna(0).values if len(feat_cols) else None
    if X_extra is not None:
        X = np.hstack([emb, X_extra])
    else:
        X = emb

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    # Train AE (save)
    ae_path = os.path.join(out_dir, "ae.pth")
    if not os.path.exists(ae_path):
        print("Training autoencoder...")
        ae = train_ae(embeddings=emb, epochs=12, save_path=ae_path)
        print("Saved AE to", ae_path)
    else:
        print("AE already present:", ae_path)

    # Define classification models (with scaler where appropriate)
    models = {
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(solver="liblinear", random_state=random_state))]),
        "svm": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=random_state))]),
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
        "xgb": xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        "lgbm": lgb.LGBMClassifier(n_estimators=200, random_state=random_state)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(out_dir, f"{name}.joblib"))
        print("Saved", name)

    # Optionally print test scores
    print("Evaluating on test set...")
    for name in models.keys():
        m = joblib.load(os.path.join(out_dir, f"{name}.joblib"))
        acc = m.score(X_test, y_test)
        print(f"{name} test acc: {acc:.4f}")

if __name__ == "__main__":
    train_models()
