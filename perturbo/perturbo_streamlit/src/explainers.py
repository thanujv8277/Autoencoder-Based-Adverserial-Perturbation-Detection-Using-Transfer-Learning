# src/explainers.py
import numpy as np
import shap

def _to_1d_shap(shap_values, pred_class=0):
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[int(pred_class)])
        if arr.ndim == 2:
            return arr[0]
        return arr.reshape(-1)

    arr = np.asarray(shap_values)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3:
        # Common binary/multiclass layouts:
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if arr.shape[0] == 1 and pred_class < arr.shape[2]:
            return arr[0, :, int(pred_class)]
        if pred_class < arr.shape[0] and arr.shape[1] == 1:
            return arr[int(pred_class), 0, :]
        if arr.shape[0] > 1:
            return arr[0, :, 0]
    return arr.reshape(-1)


def explain_with_shap(model, background, sample, nb_background=100):
    """
    Return 1D SHAP values for one sample with robust fallback logic.
    """
    background = np.asarray(background)
    sample = np.asarray(sample)
    bg = background[: min(nb_background, len(background))]

    try:
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            clf = model.named_steps["clf"]
            clf_name = clf.__class__.__name__.lower()
            if "logistic" in clf_name:
                transformed_bg = model[:-1].transform(bg)
                transformed_sample = model[:-1].transform(sample)
                expl = shap.LinearExplainer(clf, transformed_bg)
                sv = expl.shap_values(transformed_sample)
                pred = int(clf.predict(transformed_sample)[0])
                return _to_1d_shap(sv, pred)

        cls_name = model.__class__.__name__.lower()
        if any(k in cls_name for k in ["xgb", "lgbm", "randomforest", "decisiontree", "extratrees"]):
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(sample)
            pred = int(model.predict(sample)[0])
            return _to_1d_shap(sv, pred)

        if "logistic" in cls_name or "linear" in cls_name:
            expl = shap.LinearExplainer(model, bg)
            sv = expl.shap_values(sample)
            pred = int(model.predict(sample)[0])
            return _to_1d_shap(sv, pred)

        if hasattr(model, "predict_proba"):
            expl = shap.KernelExplainer(lambda x: model.predict_proba(x), bg)
            sv = expl.shap_values(sample, nsamples=100)
            pred = int(model.predict(sample)[0])
            return _to_1d_shap(sv, pred)

    except Exception as e:
        pass

    return np.zeros((sample.shape[1],))
