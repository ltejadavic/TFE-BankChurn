# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np

def load_xgb_from_manifest(manifest_path):
    import xgboost as xgb
    mp = Path(manifest_path)
    m = json.loads(mp.read_text(encoding="utf-8"))
    mdl_file = mp.parent / m["files"]["model_json"]
    booster = xgb.Booster()
    booster.load_model(str(mdl_file))

    best_it = m.get("training", {}).get("best_iteration", None)
    feat_names = m["features"]["names"]

    def predict_proba(X):
        d = xgb.DMatrix(X, feature_names=feat_names)
        if best_it is not None and isinstance(best_it, int):
            p = booster.predict(d, iteration_range=(0, best_it+1))
        else:
            p = booster.predict(d)
        return np.column_stack([1.0 - p, p])

    return m, predict_proba

def load_lgbm_from_manifest(manifest_path):
    import lightgbm as lgb
    mp = Path(manifest_path)
    m = json.loads(mp.read_text(encoding="utf-8"))
    mdl_file = mp.parent / m["files"]["model_txt"]
    booster = lgb.Booster(model_file=str(mdl_file))

    feat_names = m["features"]["names"]
    best_it = m.get("training", {}).get("best_iteration", None)

    def predict_proba(X):
        p1 = booster.predict(X, num_iteration=best_it)
        return np.column_stack([1.0 - p1, p1])

    return m, predict_proba
