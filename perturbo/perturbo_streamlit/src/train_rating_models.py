# src/train_rating_models.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.utils import load_npy

def train_rating_models(
    emb_path="data/e5_embeddings.npy",
    raw_csv="data/tripadvisor_clean_meaning_safe.csv",
    feat_csv="data/tripadvisor_hybrid_features.csv",
    out_dir="models",
    random_state=42
):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading data...")
    emb = load_npy(emb_path)
    df_raw = pd.read_csv(raw_csv)
    df_feat = pd.read_csv(feat_csv)

    if "Rating" not in df_raw.columns:
        raise ValueError("Raw CSV must contain 'Rating' column (1-5).")
    y = df_raw["Rating"].astype(int).values
    # shift to 0..4 for internal training (XGBoost expects 0-based)
    y_shift = y - 1

    feat_cols = [c for c in ["reconstruction_error","latent_distance","anomaly_score",
                             "adverb_count","adverb_weight","avg_adverb_weight"] if c in df_feat.columns]
    X_extra = df_feat[feat_cols].fillna(0).values if len(feat_cols) else None
    if X_extra is not None:
        X = np.hstack([emb, X_extra])
    else:
        X = emb

    Xtr, Xte, ytr, yte = train_test_split(X, y_shift, test_size=0.2, stratify=y_shift, random_state=random_state)

    # models
    print("Training Logistic Regression (rating)...")
    logreg = LogisticRegression(max_iter=500, multi_class="multinomial")
    logreg.fit(Xtr, ytr); joblib.dump(logreg, os.path.join(out_dir, "rating_logreg.joblib"))
    print("LogReg acc:", accuracy_score(yte, logreg.predict(Xte)))

    print("Training SVM (rating)...")
    svm = SVC(probability=True, kernel="rbf")
    svm.fit(Xtr, ytr); joblib.dump(svm, os.path.join(out_dir, "rating_svm.joblib"))
    print("SVM acc:", accuracy_score(yte, svm.predict(Xte)))

    print("Training RandomForest (rating)...")
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    rf.fit(Xtr, ytr); joblib.dump(rf, os.path.join(out_dir, "rating_rf.joblib"))
    print("RF acc:", accuracy_score(yte, rf.predict(Xte)))

    print("Training XGBoost (rating)...")
    xgbm = xgb.XGBClassifier(eval_metric="mlogloss", objective="multi:softprob", num_class=5)
    xgbm.fit(Xtr, ytr); joblib.dump(xgbm, os.path.join(out_dir, "rating_xgb.joblib"))
    print("XGB acc:", accuracy_score(yte, xgbm.predict(Xte)))

    print("Training LightGBM (rating)...")
    lgbm = lgb.LGBMClassifier()
    lgbm.fit(Xtr, ytr); joblib.dump(lgbm, os.path.join(out_dir, "rating_lgbm.joblib"))
    print("LGBM acc:", accuracy_score(yte, lgbm.predict(Xte)))

    print("All rating models saved in", out_dir)

if __name__ == "__main__":
    train_rating_models()
