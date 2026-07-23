import os
import re
import time
from html import escape
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

import src.explainers as expl
from src.utils import (
    AE,
    INTENSIFIERS,
    adverb_influence,
    ae_reconstruction,
    embed_texts,
    load_npy,
    normalize_text,
    spacy_adverbs,
)

st.set_page_config(page_title="Perturbo XAI Dashboard", page_icon="📊", layout="wide")

DATA_DIR = "data"
MODELS_DIR = "models"
EMB_PATH = os.path.join(DATA_DIR, "e5_embeddings.npy")
FEAT_CSV = os.path.join(DATA_DIR, "tripadvisor_hybrid_features.csv")
RAW_CSV = os.path.join(DATA_DIR, "tripadvisor_clean_meaning_safe.csv")
FEATURE_ORDER = [
    "reconstruction_error",
    "latent_distance",
    "anomaly_score",
    "adverb_count",
    "adverb_weight",
    "avg_adverb_weight",
]
CF_REPLACE = {
    "extremely": "very",
    "very": "quite",
    "really": "quite",
    "highly": "fairly",
    "absolutely": "mostly",
    "totally": "mostly",
    "super": "quite",
    "incredibly": "rather",
    "remarkably": "rather",
    "barely": "somewhat",
    "slightly": "moderately",
}


def percentile_norm(value, col_values):
    if col_values is None or len(col_values) == 0:
        return 0.0
    sorted_vals = np.sort(col_values)
    pos = np.searchsorted(sorted_vals, value, side="right")
    return float(pos / len(sorted_vals))


def remove_one_adverb(text, adverb):
    stripped = re.sub(rf"\b{re.escape(adverb)}\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", stripped).strip()


def replace_one_adverb(text, adverb, replacement):
    replaced = re.sub(rf"\b{re.escape(adverb)}\b", replacement, text, flags=re.I)
    return re.sub(r"\s+", " ", replaced).strip()


def get_prob_batch(model, X):
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba[:, 0]
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X)).reshape(-1)
        return 1.0 / (1.0 + np.exp(-s))
    return np.asarray(model.predict(X)).reshape(-1).astype(float)


def get_prob(model, X):
    return float(get_prob_batch(model, X)[0])


def confidence_label(prob):
    dist = abs(float(prob) - 0.5)
    if dist >= 0.3:
        return "High"
    if dist >= 0.15:
        return "Moderate"
    return "Low"


def build_features(text, emb, ae_model, df_feat, recon_col, adv_col):
    rec_err = 0.0
    lat_dist = 0.0
    if ae_model is not None:
        mse_arr, lat_arr, _ = ae_reconstruction(ae_model, emb.reshape(1, -1))
        rec_err = float(mse_arr[0])
        lat_dist = float(lat_arr[0])

    anomaly_score = percentile_norm(rec_err, recon_col) if recon_col is not None else 0.0
    advs, dom_adv, adv_diffs = adverb_influence(text, emb)
    adv_count = len(advs)
    adv_weight_raw = float(max(adv_diffs)) if adv_diffs else 0.0
    adv_weight = percentile_norm(adv_weight_raw, adv_col) if adv_col is not None else adv_weight_raw
    avg_adv_w = float(np.mean(adv_diffs)) if adv_diffs else 0.0
    avg_adv_w_norm = avg_adv_w
    if "avg_adverb_weight" in df_feat.columns:
        avg_adv_w_norm = percentile_norm(avg_adv_w, df_feat["avg_adverb_weight"].fillna(0).values)

    feat_cols = [c for c in FEATURE_ORDER if c in df_feat.columns]
    fmap = {
        "reconstruction_error": rec_err,
        "latent_distance": lat_dist,
        "anomaly_score": anomaly_score,
        "adverb_count": adv_count,
        "adverb_weight": adv_weight,
        "avg_adverb_weight": avg_adv_w_norm,
    }
    feat_vals = [fmap[c] for c in feat_cols]
    X_input = np.hstack([emb, np.asarray(feat_vals)]).reshape(1, -1)

    return {
        "X_input": X_input,
        "feat_cols": feat_cols,
        "fmap": fmap,
        "rec_err": rec_err,
        "lat_dist": lat_dist,
        "advs": advs,
        "dom_adv": dom_adv,
        "adv_weight_raw": adv_weight_raw,
        "adv_weight": adv_weight,
    }


def make_shap_plot(df_sh, title):
    fig = px.bar(
        df_sh.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale="RdBu",
        height=420,
        title=title,
    )
    fig.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=55, b=8), coloraxis_showscale=False, title_x=0.02)
    return fig


def make_ablation_table(text, emb, base_input, xai_models, ae_model, df_feat, recon_col, adv_col):
    advs = spacy_adverbs(text)
    if not advs:
        return pd.DataFrame(), pd.DataFrame()

    base_probs = {name: get_prob(m, base_input) for name, m in xai_models.items()}
    rows = []
    for adv in advs:
        variant = remove_one_adverb(text, adv)
        if not variant:
            continue
        emb2 = embed_texts([variant], batch_size=1)[0]
        mod = build_features(variant, emb2, ae_model, df_feat, recon_col, adv_col)
        X2 = mod["X_input"]
        sem_shift = float(np.linalg.norm(emb - emb2))
        for name, m in xai_models.items():
            p2 = get_prob(m, X2)
            d = p2 - base_probs[name]
            rows.append((adv, name, base_probs[name], p2, d, abs(d), sem_shift))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["adverb", "model", "p_base", "p_without_adverb", "delta_prob", "abs_delta", "semantic_shift"],
    )
    summary = (
        df.groupby("adverb", as_index=False)
        .agg(mean_abs_delta=("abs_delta", "mean"), mean_semantic_shift=("semantic_shift", "mean"))
        .sort_values("mean_abs_delta", ascending=False)
    )
    return df, summary


def counterfactual_search(text, base_label, base_prob, xai_models, ae_model, df_feat, recon_col, adv_col):
    advs = spacy_adverbs(text)
    candidates = []
    for adv in advs:
        candidates.append(("remove", adv, remove_one_adverb(text, adv)))
        repl = CF_REPLACE.get(adv.lower(), "quite")
        candidates.append(("replace", adv, replace_one_adverb(text, adv, repl)))

    for token in sorted(INTENSIFIERS)[:10]:
        candidates.append(("replace", token, replace_one_adverb(text, token, "quite")))

    seen = set()
    out = []
    for op, token, cand in candidates:
        cand = cand.strip()
        if not cand or cand == text or cand in seen:
            continue
        seen.add(cand)
        emb2 = embed_texts([cand], batch_size=1)[0]
        mod = build_features(cand, emb2, ae_model, df_feat, recon_col, adv_col)
        X2 = mod["X_input"]
        probs = [get_prob(m, X2) for m in xai_models.values()]
        p_avg = float(np.mean(probs))
        pred = int(p_avg >= 0.5)
        flip = int(pred != base_label)
        out.append((op, token, cand, p_avg, p_avg - base_prob, flip))
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["operation", "token", "candidate_text", "avg_prob", "delta_prob", "flip"])
    return df.sort_values(["flip", "avg_prob"], ascending=[False, True]).reset_index(drop=True)


def token_importance(text, base_input, xai_models, ae_model, df_feat, recon_col, adv_col):
    tokens = [t for t in re.findall(r"\w+|[^\w\s]", text) if t.strip()]
    if len(tokens) < 2:
        return pd.DataFrame()
    base_probs = {name: get_prob(m, base_input) for name, m in xai_models.items()}
    rows = []
    for i, tok in enumerate(tokens):
        if not re.search(r"[A-Za-z]", tok):
            continue
        reduced = " ".join(tokens[:i] + tokens[i + 1 :])
        reduced = re.sub(r"\s+", " ", reduced).strip()
        if not reduced:
            continue
        emb2 = embed_texts([reduced], batch_size=1)[0]
        mod = build_features(reduced, emb2, ae_model, df_feat, recon_col, adv_col)
        X2 = mod["X_input"]
        deltas = []
        for name, model in xai_models.items():
            deltas.append(get_prob(model, X2) - base_probs[name])
        rows.append((tok, float(np.mean(deltas)), float(np.mean(np.abs(deltas)))))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["token", "avg_delta_prob", "mean_abs_delta"]).sort_values("mean_abs_delta", ascending=False)


def rule_anchors_from_ablation(df_summary):
    if df_summary.empty:
        return []
    anchors = []
    for _, row in df_summary.head(5).iterrows():
        adv = row["adverb"]
        imp = float(row["mean_abs_delta"])
        strength = "strong" if imp >= 0.15 else "moderate" if imp >= 0.08 else "weak"
        anchors.append(f"If '{adv}' is present, prediction shift is {strength} (mean |delta p| = {imp:.3f}).")
    return anchors


def lime_explanation(x_model, X_bg, X_input, feat_names):
    explainer = LimeTabularExplainer(
        training_data=X_bg,
        mode="classification",
        feature_names=feat_names,
        class_names=["clean", "perturbed"],
        discretize_continuous=True,
    )
    exp = explainer.explain_instance(X_input[0], x_model.predict_proba, num_features=10)
    out = pd.DataFrame(exp.as_list(), columns=["feature", "weight"])
    out["abs_weight"] = out["weight"].abs()
    return out.sort_values("abs_weight", ascending=False)


def pdp_ice_for_feature(model, X_input, feat_idx, grid):
    X_grid = np.repeat(X_input, len(grid), axis=0)
    X_grid[:, feat_idx] = grid
    probs = get_prob_batch(model, X_grid)
    return pd.DataFrame({"feature_value": grid, "prob_perturbed": probs})


def extra_feature_permutation_stability(model, X_sample, y_sample, feat_indices, feat_names, seeds):
    baseline_acc = float(np.mean(model.predict(X_sample) == y_sample))
    rows = []
    for fi, fname in zip(feat_indices, feat_names):
        drops = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            Xp = X_sample.copy()
            Xp[:, fi] = rng.permutation(Xp[:, fi])
            acc = float(np.mean(model.predict(Xp) == y_sample))
            drops.append(baseline_acc - acc)
        rows.append((fname, float(np.mean(drops)), float(np.std(drops))))
    return pd.DataFrame(rows, columns=["feature", "importance_drop", "stability_std"]).sort_values("importance_drop", ascending=False)


def calibration_curve_df(y_true, prob, bins=10):
    prob = np.asarray(prob).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (prob >= lo) & (prob < hi if i < bins - 1 else prob <= hi)
        if mask.sum() == 0:
            continue
        conf = float(prob[mask].mean())
        acc = float(y_true[mask].mean())
        rows.append((conf, acc, int(mask.sum())))
    return pd.DataFrame(rows, columns=["confidence", "accuracy", "count"])


def show_insight(title, lines):
    body = "<br>".join([f"- {line}" for line in lines if line])
    st.markdown(
        f"""
<div class="insight-box">
  <div class="insight-title">{title}</div>
  <div class="insight-body">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def highlight_text_diff(original, updated):
    orig_tokens = re.findall(r"\w+|[^\w\s]", original)
    upd_tokens = re.findall(r"\w+|[^\w\s]", updated)
    upd_set = {t.lower() for t in upd_tokens}
    orig_html = []
    upd_html = []
    for tok in orig_tokens:
        cls = "diff-removed" if tok.lower() not in upd_set and re.search(r"[A-Za-z]", tok) else ""
        orig_html.append(f"<span class='{cls}'>{escape(tok)}</span>")
    orig_set = {t.lower() for t in orig_tokens}
    for tok in upd_tokens:
        cls = "diff-added" if tok.lower() not in orig_set and re.search(r"[A-Za-z]", tok) else ""
        upd_html.append(f"<span class='{cls}'>{escape(tok)}</span>")
    return " ".join(orig_html), " ".join(upd_html)


def build_pipeline_figure():
    fig = go.Figure()
    steps = [
        ("Input Review", 0),
        ("Embedding", 1),
        ("Hybrid Features", 2),
        ("Prediction", 3),
        ("Explanation", 4),
    ]
    for i in range(len(steps) - 1):
        fig.add_trace(
            go.Scatter(
                x=[steps[i][1], steps[i + 1][1]],
                y=[0, 0],
                mode="lines",
                line=dict(color="#49708a", width=5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[s[1] for s in steps],
            y=[0] * len(steps),
            mode="markers+text",
            text=[s[0] for s in steps],
            textposition="top center",
            marker=dict(size=24, color=["#0b2545", "#134074", "#2a6f97", "#5fa8d3", "#8ecae6"]),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-0.3, 4.3]),
        yaxis=dict(visible=False, range=[-0.5, 0.8]),
    )
    return fig


def build_export_text(query, verdict, avg_prob, conf_label, base, df_adv_summary, df_cf):
    lines = [
        "Perturbo XAI Summary",
        "",
        f"Input: {query}",
        f"Prediction: {verdict}",
        f"Average perturbed probability: {avg_prob:.3f}",
        f"Confidence: {conf_label}",
        f"Adverbs detected: {', '.join(base['advs']) if base['advs'] else 'none'}",
        f"Dominant adverb: {base['dom_adv'] if base['dom_adv'] else 'N/A'}",
    ]
    if not df_adv_summary.empty:
        top_adv = df_adv_summary.iloc[0]
        lines.append(
            f"Top adverb impact: {top_adv['adverb']} (mean absolute probability shift {float(top_adv['mean_abs_delta']):.3f})"
        )
    if not df_cf.empty:
        best = df_cf.iloc[0]
        lines.append(
            f"Best counterfactual: {best['operation']} '{best['token']}' -> avg p(perturbed) {float(best['avg_prob']):.3f}"
        )
    lines.append("")
    lines.append("Why this matters: the dashboard links prediction, linguistic perturbation, and interpretable evidence in one view.")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def load_data():
    emb_np = load_npy(EMB_PATH)
    df_feat = pd.read_csv(FEAT_CSV)
    df_raw = pd.read_csv(RAW_CSV)
    return emb_np, df_feat, df_raw


@st.cache_resource(show_spinner=False)
def load_models(emb_dim):
    ae_model = None
    ae_path = os.path.join(MODELS_DIR, "ae.pth")
    if os.path.exists(ae_path):
        try:
            ae_model = AE(emb_dim)
            ae_model.load_state_dict(__import__("torch").load(ae_path, map_location="cpu"))
            ae_model.eval()
        except Exception:
            ae_model = None

    class_models = {}
    for n in ["logreg", "svm", "rf", "xgb", "lgbm"]:
        p = os.path.join(MODELS_DIR, f"{n}.joblib")
        if os.path.exists(p):
            class_models[n] = joblib.load(p)

    rating_models = {}
    for n in ["rating_logreg.joblib", "rating_svm.joblib", "rating_rf.joblib", "rating_xgb.joblib", "rating_lgbm.joblib"]:
        p = os.path.join(MODELS_DIR, n)
        if os.path.exists(p):
            rating_models[n] = joblib.load(p)
    return ae_model, class_models, rating_models


st.markdown(
    """
<style>
section.main > div {max-width: 1400px;}
.hero {
    background: radial-gradient(circle at 20% 20%, #0b2545 0%, #134074 45%, #8da9c4 100%);
    color: #f8fafc;
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 0.8rem;
}
.metric-card {
    border: 1px solid #dde6ef;
    border-left: 5px solid #134074;
    background: #f8fbff;
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
}
.block {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    background: #ffffff;
    padding: 0.7rem 0.8rem 0.3rem 0.8rem;
}
.insight-box {
    border: 1px solid #dbe4ee;
    border-left: 5px solid #2a6f97;
    background: #f7fbff;
    border-radius: 10px;
    padding: 0.65rem 0.8rem;
    margin: 0.4rem 0 0.9rem 0;
}
.insight-title {
    font-weight: 700;
    color: #12344d;
    margin-bottom: 0.2rem;
}
.insight-body {
    color: #1f2d3d;
    line-height: 1.5;
}
.summary-banner {
    background: linear-gradient(90deg, #0b2545 0%, #134074 45%, #5fa8d3 100%);
    color: #f8fafc;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0 1rem 0;
}
.compare-box {
    border: 1px solid #dbe4ee;
    border-radius: 12px;
    background: #fbfdff;
    padding: 0.8rem;
    min-height: 150px;
}
.diff-removed {
    background: #fde2e4;
    color: #8d0801;
    padding: 0.08rem 0.22rem;
    border-radius: 5px;
}
.diff-added {
    background: #d8f3dc;
    color: #1b4332;
    padding: 0.08rem 0.22rem;
    border-radius: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)

if not os.path.exists(EMB_PATH) or not os.path.exists(FEAT_CSV) or not os.path.exists(RAW_CSV):
    st.error("Missing required data files in data/.")
    st.stop()

emb_np, df_feat, df_raw = load_data()
ae_model, class_models, rating_models = load_models(emb_np.shape[1])
xai_models = {k: class_models[k] for k in ["logreg", "lgbm"] if k in class_models}

if not xai_models:
    st.error("Required XAI models not found. Keep models/logreg.joblib and models/lgbm.joblib.")
    st.stop()

recon_col = df_feat["reconstruction_error"].fillna(0).values if "reconstruction_error" in df_feat.columns else None
adv_col = df_feat["adverb_weight"].fillna(0).values if "adverb_weight" in df_feat.columns else None
feat_extra = [c for c in FEATURE_ORDER if c in df_feat.columns]
feat_names = [f"emb_{i}" for i in range(emb_np.shape[1])] + feat_extra

st.sidebar.title("XAI Controls")
shap_bg = st.sidebar.slider("Background samples", min_value=40, max_value=400, value=120, step=20)
global_sample = st.sidebar.slider("Global analysis sample", min_value=200, max_value=min(2000, len(df_feat)), value=min(900, len(df_feat)), step=100)
pdp_feature = st.sidebar.selectbox("PDP/ICE feature", options=feat_extra if feat_extra else FEATURE_ORDER)

st.markdown(
    """
<div class="hero">
  <h2 style="margin:0 0 0.35rem 0;">Perturbo Explainable AI Dashboard</h2>
  <div style="font-size:1rem;">
    SHAP + LIME + Adverb Ablation + Counterfactuals + Anchor Rules + Token Attribution + PDP/ICE + Global Stability + Calibration
  </div>
</div>
""",
    unsafe_allow_html=True,
)

query = st.text_area(
    "Review text",
    height=180,
    placeholder="Example: The hotel was extremely clean but the service was barely acceptable and really slow.",
)
run_btn = st.button("Run Full XAI Analysis", type="primary")

if run_btn:
    query = normalize_text(query)
    if not query:
        st.warning("Type a review first.")
        st.stop()

    t0 = time.time()
    with st.spinner("Computing embeddings, predictions, and explainability blocks..."):
        emb = embed_texts([query], batch_size=1)[0]
        base = build_features(query, emb, ae_model, df_feat, recon_col, adv_col)
        X_input = base["X_input"]
        bg_n = min(shap_bg, emb_np.shape[0])
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(np.arange(emb_np.shape[0]), size=bg_n, replace=False)
        X_bg = np.hstack([emb_np[bg_idx], df_feat[feat_extra].fillna(0).values[bg_idx] if feat_extra else np.zeros((bg_n, 0))])
    majority = "unknown"
    avg_prob = 0.0
    conf_label = "Low"

    st.subheader("Presentation Summary")
    st.plotly_chart(build_pipeline_figure(), use_container_width=True)

    st.subheader("1) Predictions Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><b>Adverbs</b><br>{len(base['advs'])}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><b>Reconstruction Error</b><br>{base['rec_err']:.6f}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><b>Latent Distance</b><br>{base['lat_dist']:.4f}</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><b>Dominant Adverb</b><br>{base['dom_adv'] if base['dom_adv'] else 'N/A'}</div>", unsafe_allow_html=True)

    cls_rows = []
    for name, model in class_models.items():
        try:
            p = get_prob(model, X_input)
            cls_rows.append((name, "perturbed" if p >= 0.5 else "clean", p))
        except Exception as e:
            cls_rows.append((name, "ERROR", str(e)))
    df_cls = pd.DataFrame(cls_rows, columns=["model", "prediction", "p_perturbed"])
    st.dataframe(df_cls, use_container_width=True)
    if not df_cls.empty and "p_perturbed" in df_cls.columns:
        valid_probs = pd.to_numeric(df_cls["p_perturbed"], errors="coerce").dropna()
        if not valid_probs.empty:
            majority = "perturbed" if (valid_probs >= 0.5).sum() >= (len(valid_probs) / 2.0) else "clean"
            avg_prob = float(valid_probs.mean())
            conf_label = confidence_label(avg_prob)
            top_reason = base["dom_adv"] if base["dom_adv"] else "hybrid semantic features"
            st.markdown(
                f"""
<div class="summary-banner">
  <div><b>Prediction:</b> {majority.title()}</div>
  <div><b>Main reason:</b> {top_reason} drives the strongest observed linguistic shift.</div>
  <div><b>Confidence:</b> {conf_label} ({avg_prob:.3f} average perturbed probability)</div>
</div>
""",
                unsafe_allow_html=True,
            )
            show_insight(
                "Prediction Summary",
                [
                    f"Model consensus is mostly '{majority}'.",
                    f"Average perturbed probability across classifiers is {avg_prob:.3f}.",
                    f"Presentation confidence label is '{conf_label}'.",
                    f"Detected adverbs in text: {', '.join(base['advs']) if base['advs'] else 'none'}.",
                ],
            )
            vote_counts = df_cls["prediction"].value_counts().reset_index()
            vote_counts.columns = ["prediction", "votes"]
            fig_agree = px.bar(
                vote_counts,
                x="prediction",
                y="votes",
                color="prediction",
                title="Model agreement",
                text="votes",
                color_discrete_map={"perturbed": "#c1121f", "clean": "#2a9d8f", "ERROR": "#6c757d"},
                height=260,
            )
            fig_agree.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=48, b=8), title_x=0.02, showlegend=False)
            st.plotly_chart(fig_agree, use_container_width=True)

    rating_rows = []
    for name, model in rating_models.items():
        try:
            pred = int(model.predict(X_input)[0]) + 1
            conf = float(np.max(model.predict_proba(X_input)[0])) if hasattr(model, "predict_proba") else None
            rating_rows.append((name, pred, conf))
        except Exception as e:
            rating_rows.append((name, "ERROR", str(e)))
    if rating_rows:
        df_rating = pd.DataFrame(rating_rows, columns=["model", "predicted_stars", "confidence"])

        def star_visual(x):
            if isinstance(x, (int, np.integer)):
                x = max(1, min(5, int(x)))
                return " ".join(["★"] * x + ["☆"] * (5 - x))
            return x

        df_rating["stars_visual"] = df_rating["predicted_stars"].apply(star_visual)
        st.dataframe(df_rating, use_container_width=True)
        rating_valid = pd.to_numeric(df_rating["predicted_stars"], errors="coerce").dropna()
        if not rating_valid.empty:
            show_insight(
                "Rating Summary",
                [
                    f"Average predicted rating is {rating_valid.mean():.2f} stars.",
                    f"Most common rating prediction is {int(rating_valid.round().mode().iloc[0])} stars.",
                ],
            )

    st.subheader("2) SHAP (Logistic Regression + LightGBM)")
    col_a, col_b = st.columns(2)
    shap_tops = {}
    for i, mname in enumerate(["logreg", "lgbm"]):
        if mname not in xai_models:
            continue
        sv = np.asarray(expl.explain_with_shap(xai_models[mname], X_bg, X_input, nb_background=shap_bg)).reshape(-1)
        n = min(len(sv), len(feat_names))
        sv = sv[:n]
        local_names = feat_names[:n]
        top = np.argsort(np.abs(sv))[-12:][::-1]
        df_sh = pd.DataFrame({"feature": [local_names[j] for j in top], "shap_value": [float(sv[j]) for j in top]})
        if not df_sh.empty:
            shap_tops[mname] = (df_sh.iloc[0]["feature"], float(df_sh.iloc[0]["shap_value"]))
        target_col = col_a if i == 0 else col_b
        target_col.plotly_chart(make_shap_plot(df_sh, f"{mname.upper()} SHAP"), use_container_width=True)
    if shap_tops:
        lines = []
        for mname, (feat, sval) in shap_tops.items():
            direction = "increases perturbed score" if sval > 0 else "reduces perturbed score"
            lines.append(f"{mname.upper()}: top driver is '{feat}' (SHAP {sval:.4f}), which {direction}.")
        show_insight("How To Read The SHAP Graphs", lines)

    st.subheader("3) Adverb Ablation Impact")
    df_ablation, df_adv_summary = make_ablation_table(query, emb, X_input, xai_models, ae_model, df_feat, recon_col, adv_col)
    if df_ablation.empty:
        st.info("No adverb was detected for ablation analysis.")
    else:
        st.dataframe(df_ablation.sort_values("abs_delta", ascending=False), use_container_width=True)
        fig = px.bar(
            df_ablation,
            x="adverb",
            y="delta_prob",
            color="model",
            barmode="group",
            title="Change in perturbed probability when an adverb is removed",
            text_auto=".3f",
            height=360,
        )
        fig.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=52, b=8), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)
        top_adv = df_adv_summary.iloc[0]
        show_insight(
            "Adverb Impact Finding",
            [
                f"Most influential adverb is '{top_adv['adverb']}' with mean absolute probability shift {top_adv['mean_abs_delta']:.3f}.",
                "Positive delta means removing the adverb increases perturbed probability; negative delta means it decreases.",
            ],
        )

    st.subheader("4) Counterfactual Explanations")
    base_prob = float(np.mean([get_prob(m, X_input) for m in xai_models.values()]))
    base_label = int(base_prob >= 0.5)
    df_cf = counterfactual_search(query, base_label, base_prob, xai_models, ae_model, df_feat, recon_col, adv_col)
    if df_cf.empty:
        st.info("No valid counterfactual candidate was found.")
    else:
        st.dataframe(df_cf.head(8), use_container_width=True)
        best = df_cf.iloc[0]
        st.markdown(
            f"Best counterfactual: `{best['operation']}` `{best['token']}` -> avg p(perturbed) `{best['avg_prob']:.3f}` (delta `{best['delta_prob']:.3f}`)"
        )
        show_insight(
            "Counterfactual Interpretation",
            [
                f"The best edit is to {best['operation']} '{best['token']}'.",
                f"This changes average perturbed probability by {best['delta_prob']:.3f}.",
                "Counterfactuals show minimal edits that can flip or strongly shift model decisions.",
            ],
        )
        orig_html, upd_html = highlight_text_diff(query, best["candidate_text"])
        cmp_a, cmp_b = st.columns(2)
        cmp_a.markdown(f"<div class='compare-box'><b>Original</b><br><br>{orig_html}</div>", unsafe_allow_html=True)
        cmp_b.markdown(f"<div class='compare-box'><b>Best Counterfactual</b><br><br>{upd_html}</div>", unsafe_allow_html=True)

    st.subheader("5) LIME Local Explanation (LightGBM)")
    df_lime = pd.DataFrame()
    if "lgbm" in xai_models and hasattr(xai_models["lgbm"], "predict_proba"):
        try:
            df_lime = lime_explanation(xai_models["lgbm"], X_bg, X_input, feat_names)
            fig = px.bar(
                df_lime.sort_values("weight"),
                x="weight",
                y="feature",
                orientation="h",
                title="LIME Feature Weights (local)",
                color="weight",
                color_continuous_scale="BrBG",
                height=380,
            )
            fig.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=52, b=8), coloraxis_showscale=False, title_x=0.02)
            st.plotly_chart(fig, use_container_width=True)
            if not df_lime.empty:
                lime_top = df_lime.iloc[0]
                direction = "pushes toward perturbed" if float(lime_top["weight"]) > 0 else "pushes toward clean"
                show_insight(
                    "LIME Interpretation",
                    [f"Top local rule '{lime_top['feature']}' has weight {float(lime_top['weight']):.4f} and {direction}."],
                )
        except Exception as e:
            st.warning(f"LIME unavailable for this run: {e}")
    else:
        st.info("LIME skipped: LightGBM probability model not available.")

    st.subheader("6) Anchor-Style Local Rules")
    anchors = rule_anchors_from_ablation(df_adv_summary)
    if anchors:
        st.markdown("<div class='block'>" + "<br>".join([f"• {a}" for a in anchors]) + "</div>", unsafe_allow_html=True)
    else:
        st.info("No robust local anchor-style rules could be generated.")

    st.subheader("7) Token Attribution (Deletion Sensitivity)")
    df_tok = token_importance(query, X_input, xai_models, ae_model, df_feat, recon_col, adv_col)
    if df_tok.empty:
        st.info("Not enough tokens for token attribution.")
    else:
        fig = px.bar(
            df_tok.head(15).sort_values("avg_delta_prob"),
            x="avg_delta_prob",
            y="token",
            orientation="h",
            title="Mean probability change when token is removed",
            color="avg_delta_prob",
            color_continuous_scale="RdBu",
            height=420,
        )
        fig.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=52, b=8), coloraxis_showscale=False, title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)
        tok_top = df_tok.iloc[0]
        show_insight(
            "Token-Level Finding",
            [
                f"Most sensitive token is '{tok_top['token']}' with mean absolute probability impact {tok_top['mean_abs_delta']:.3f}.",
                "Higher absolute delta indicates the model relies more on that token.",
            ],
        )

    st.subheader("8) PDP / ICE for Hybrid Feature")
    if pdp_feature in feat_extra:
        fi = feat_extra.index(pdp_feature)
        input_col_idx = emb_np.shape[1] + fi
        grid = np.quantile(df_feat[pdp_feature].fillna(0).values, np.linspace(0.05, 0.95, 15))
        fig = go.Figure()
        for mname, model in xai_models.items():
            d = pdp_ice_for_feature(model, X_input, input_col_idx, grid)
            fig.add_trace(go.Scatter(x=d["feature_value"], y=d["prob_perturbed"], mode="lines+markers", name=mname))
        d_ref = pdp_ice_for_feature(xai_models[list(xai_models.keys())[0]], X_input, input_col_idx, grid)
        trend = d_ref["prob_perturbed"].iloc[-1] - d_ref["prob_perturbed"].iloc[0]
        fig.update_layout(
            template="plotly_white",
            title=f"Partial dependence on {pdp_feature}",
            xaxis_title=pdp_feature,
            yaxis_title="P(perturbed)",
            height=370,
            margin=dict(l=8, r=8, t=52, b=8),
            title_x=0.02,
        )
        st.plotly_chart(fig, use_container_width=True)
        show_insight(
            "PDP/ICE Interpretation",
            [
                f"As '{pdp_feature}' moves from low to high range, perturbed probability changes by about {trend:.3f} in the reference model.",
                "Use this to understand directionality: whether increasing this feature raises or lowers risk.",
            ],
        )
    else:
        st.info("PDP/ICE unavailable because selected feature is not in the trained feature set.")

    st.subheader("9) Global Feature Importance + Stability")
    y_all = (df_feat["label"].astype(str).str.lower() == "perturbed").astype(int).values
    idx = np.random.default_rng(7).choice(np.arange(len(df_feat)), size=min(global_sample, len(df_feat)), replace=False)
    X_all = np.hstack([emb_np, df_feat[feat_extra].fillna(0).values if feat_extra else np.zeros((len(df_feat), 0))])
    Xs = X_all[idx]
    ys = y_all[idx]
    if feat_extra:
        feat_indices = [emb_np.shape[1] + i for i in range(len(feat_extra))]
        seeds = [11, 29, 47]
        gcols = st.columns(2)
        global_findings = []
        for j, mname in enumerate(["logreg", "lgbm"]):
            if mname not in xai_models:
                continue
            df_g = extra_feature_permutation_stability(xai_models[mname], Xs, ys, feat_indices, feat_extra, seeds)
            if not df_g.empty:
                top_g = df_g.iloc[-1] if "importance_drop" not in df_g.columns else df_g.sort_values("importance_drop", ascending=False).iloc[0]
                global_findings.append(
                    f"{mname.upper()}: '{top_g['feature']}' is most important (accuracy drop {top_g['importance_drop']:.3f}, std {top_g['stability_std']:.3f})."
                )
            fig = px.bar(
                df_g.sort_values("importance_drop"),
                x="importance_drop",
                y="feature",
                orientation="h",
                error_x="stability_std",
                title=f"{mname.upper()} global importance (permute drop)",
                height=360,
            )
            fig.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=52, b=8), title_x=0.02)
            gcols[j].plotly_chart(fig, use_container_width=True)
        if global_findings:
            show_insight("Global Importance Interpretation", global_findings)
    else:
        st.info("Global extra-feature importance unavailable (no hybrid features found).")

    st.subheader("10) Uncertainty and Calibration")
    cal_cols = st.columns(2)
    cal_summaries = []
    for i, mname in enumerate(["logreg", "lgbm"]):
        if mname not in xai_models:
            continue
        p_all = get_prob_batch(xai_models[mname], Xs)
        df_cal = calibration_curve_df(ys, p_all, bins=10)
        if not df_cal.empty:
            ece = float(np.average(np.abs(df_cal["accuracy"] - df_cal["confidence"]), weights=df_cal["count"]))
            cal_summaries.append(f"{mname.upper()} expected calibration gap is {ece:.3f} (lower is better).")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="ideal", line=dict(dash="dash")))
        fig.add_trace(
            go.Scatter(
                x=df_cal["confidence"],
                y=df_cal["accuracy"],
                mode="lines+markers",
                marker=dict(size=np.clip(df_cal["count"] / 4, 6, 18)),
                name=mname,
            )
        )
        fig.update_layout(
            template="plotly_white",
            title=f"{mname.upper()} calibration",
            xaxis_title="Predicted confidence",
            yaxis_title="Observed accuracy",
            height=340,
            margin=dict(l=8, r=8, t=52, b=8),
            title_x=0.02,
        )
        cal_cols[i].plotly_chart(fig, use_container_width=True)
    if cal_summaries:
        show_insight("Calibration Interpretation", cal_summaries)

    with st.expander("Feature Glossary", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- `reconstruction_error`: how far the review embedding is from the autoencoder reconstruction.",
                    "- `latent_distance`: how unusual the sample looks in latent space.",
                    "- `anomaly_score`: normalized anomaly indicator derived from reconstruction behavior.",
                    "- `adverb_count`: number of detected adverbs and intensity markers.",
                    "- `adverb_weight`: strongest adverb-induced embedding drift.",
                    "- `avg_adverb_weight`: average semantic drift caused by removing detected adverbs.",
                ]
            )
        )

    show_insight(
        "Why This Matters",
        [
            "The dashboard links model output to specific language patterns instead of showing a raw label only.",
            "This makes the system easier to defend in a presentation because each prediction is backed by measurable evidence.",
            "Adverb ablation and counterfactual edits are especially useful because they translate model behavior into sentence-level reasoning.",
        ],
    )

    export_text = build_export_text(query, majority.title(), avg_prob, conf_label, base, df_adv_summary, df_cf)
    st.download_button(
        "Download XAI Summary",
        data=export_text,
        file_name="perturbo_xai_summary.txt",
        mime="text/plain",
        use_container_width=False,
    )

    st.caption(f"Completed in {time.time() - t0:.2f}s")
