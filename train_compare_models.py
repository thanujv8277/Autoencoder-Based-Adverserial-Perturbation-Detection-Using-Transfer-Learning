import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
CSV_PATH = "tripadvisor_hybrid_features.csv"
EMB_PATH = "e5_embeddings.npy"
OUTPUT_DIR = "model_results_no_hybrid"

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
emb = np.load(EMB_PATH)

assert len(df) == emb.shape[0], "Row mismatch between CSV and embeddings"

# ----------------------------------------------------------
# LABELS
# ----------------------------------------------------------
df['label_num'] = (df['label'] == 'perturbed').astype(int)
y = df['label_num'].values

# ----------------------------------------------------------
# FEATURES (NO HYBRID SCORE)
# ----------------------------------------------------------
feature_cols = [
    "reconstruction_error",
    "latent_distance",
    "anomaly_score",
    "adverb_count",
    "adverb_weight",
    "avg_adverb_weight"
]

feature_cols = [c for c in feature_cols if c in df.columns]

print("Numeric features:", feature_cols)

X_numeric = df[feature_cols].fillna(0).values
X = np.hstack([emb, X_numeric])

print("Feature shape:", X.shape)

# ----------------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print("Train:", X_train.shape[0], "Test:", X_test.shape[0])

# ----------------------------------------------------------
# MODEL EVALUATION
# ----------------------------------------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test, do_scale=False):
    if do_scale:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    else:
        pipe = Pipeline([("clf", model)])

    print(f"\nTraining {name}...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # cross-val accuracy
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    # cross-val roc
    try:
        cv_roc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    except:
        cv_roc = None

    pipe.fit(X_train, y_train)

    # predictions
    y_pred = pipe.predict(X_test)

    # probabilities (fallback if needed)
    try:
        y_score = pipe.predict_proba(X_test)[:,1]
    except:
        try:
            y_score = pipe.decision_function(X_test)
        except:
            y_score = y_pred

    # metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_score)
    except:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_test, y_score)
    except:
        pr_auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)

    print(f"{name} RESULTS:")
    print(f" Accuracy:  {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall:    {rec:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    print(f" ROC AUC:   {roc_auc:.4f}")
    print(f" PR  AUC:   {pr_auc:.4f}")
    print(cm)

    return {
        "name": name,
        "model": pipe,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_score": y_score
    }

# ----------------------------------------------------------
# MODELS
# ----------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(solver="liblinear"),
    "SVM(RBF)": SVC(kernel="rbf", probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=300, eval_metric="logloss"),
    "LightGBM": lgb.LGBMClassifier(n_estimators=300)
}

results = []

for name, model in models.items():
    do_scale = name in ("LogisticRegression", "SVM(RBF)")
    res = evaluate_model(name, model, X_train, y_train, X_test, y_test, do_scale)
    results.append(res)

# ----------------------------------------------------------
# PLOTS
# ----------------------------------------------------------
def plot_confusion(res):
    cm = res["confusion_matrix"]
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["clean","perturbed"],
                yticklabels=["clean","perturbed"])
    plt.title(f"Confusion Matrix - {res['name']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{res['name']}.png"))
    plt.close()

def plot_roc(results):
    plt.figure(figsize=(8,6))
    for res in results:
        y_test = res["y_test"]
        y_score = res["y_score"]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{res['name']} (AUC={auc_val:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plt.close()

def plot_pr(results):
    plt.figure(figsize=(8,6))
    for res in results:
        y_test = res["y_test"]
        y_score = res["y_score"]
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        auc_val = auc(rec, prec)
        plt.plot(rec, prec, label=f"{res['name']} (AUC={auc_val:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curves.png"))
    plt.close()

def plot_model_comparison(results):
    dfm = pd.DataFrame([{
        "model": r["name"],
        "accuracy": r["test_accuracy"],
        "f1": r["test_f1"],
        "roc_auc": r["test_roc_auc"]
    } for r in results]).set_index("model")
    
    dfm.plot(kind="bar", figsize=(10,6))
    plt.title("Model Comparison (Accuracy, F1, ROC AUC)")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plt.close()

# Save all plots
for r in results:
    plot_confusion(r)

plot_roc(results)
plot_pr(results)
plot_model_comparison(results)

summary = pd.DataFrame([{
    "model": r["name"],
    "accuracy": r["test_accuracy"],
    "precision": r["test_precision"],
    "recall": r["test_recall"],
    "f1": r["test_f1"],
    "roc_auc": r["test_roc_auc"],
    "pr_auc": r["test_pr_auc"]
} for r in results])

summary.to_csv(os.path.join(OUTPUT_DIR, "summary_no_hybrid.csv"), index=False)
print("\nSaved all results and plots in:", OUTPUT_DIR)
