"""Decision-threshold calibration for the band-gap classifier (rare positive class).

The threshold that maximises F1 is chosen on 3-fold out-of-fold predictions of the TRAINING
set only, then applied unchanged to the test set; the applicability-domain metrics are
recomputed at this threshold. Updates band_gap_summary.json and the saved predictor.

Usage:  uv run python pipeline/band_gap_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssl_core as core  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402,F401  (unpickling)

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT, MODELS = (ROOT / "data" / "processed", ROOT / "data" / "predictions" / "band_gap",
                     ROOT / "app" / "models")


def main() -> None:
    import __main__

    __main__.Preprocessor = Preprocessor
    pkg = joblib.load(MODELS / "band_gap_model.pkl")
    prep = joblib.load(MODELS / "band_gap_scaler.pkl")
    df = pd.read_parquet(PROC / "band_gap_features.parquet")
    feat = [c for c in df.columns if c.startswith(("MagpieData", "frac ")) or c.endswith("-norm")]
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["target"],
                              random_state=42)          # identical split to band_gap_ga_ad.py
    sel = pkg["selected_idx"]
    X_tr = prep.transform(df.iloc[tr][feat])[:, sel]
    X_te = prep.transform(df.iloc[te][feat])[:, sel]
    y_tr, y_te = df["target"].values[tr], df["target"].values[te]

    oof = np.zeros(len(y_tr))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(X_tr, y_tr):
        m = clone(pkg["model"]).fit(X_tr[a], y_tr[a])
        oof[b] = core.proba_pos(m, X_tr[b])
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    t = float(thr[np.argmax(f1[:-1])])
    print(f"F1-optimal threshold on training out-of-fold predictions: {t:.3f} "
          f"(OOF F1={f1[:-1].max():.4f})")

    pr = core.proba_pos(pkg["model"], X_te)
    res = core.evaluate(y_te, (pr >= t).astype(int), pr, "RF-ga tuned threshold",
                        threshold=t)
    print(f"test at t={t:.3f}: F1={res['F1-Score']:.4f} precision={res['Precision']:.4f} "
          f"recall={res['Recall']:.4f} ROC-AUC={res['ROC-AUC']:.4f}")
    d_te = pkg["nn"].kneighbors(X_te, n_neighbors=5)[0].mean(axis=1)
    inside = d_te <= pkg["ad_threshold"]
    ad_in = core.evaluate(y_te[inside], (pr[inside] >= t).astype(int), pr[inside], "in")
    ad_out = core.evaluate(y_te[~inside], (pr[~inside] >= t).astype(int), pr[~inside], "out")
    print(f"inside AD F1={ad_in['F1-Score']:.4f} recall={ad_in['Recall']:.4f} | outside AD "
          f"F1={ad_out['F1-Score']:.4f} recall={ad_out['Recall']:.4f}")

    pkg["threshold"] = t
    joblib.dump(pkg, MODELS / "band_gap_model.pkl", compress=("xz", 3))
    S = json.loads((OUT / "band_gap_summary.json").read_text())
    S["decision_threshold"] = {"value": t, "method": "max F1 on 3-fold OOF training predictions",
                               "oof_f1": float(f1[:-1].max())}
    S["test_ga_descriptors_tuned_threshold"] = res
    S["applicability_domain"]["metrics_inside_tuned"] = ad_in
    S["applicability_domain"]["metrics_outside_tuned"] = ad_out
    (OUT / "band_gap_summary.json").write_text(json.dumps(S, indent=2, default=float))
    print("updated band_gap_summary.json and band_gap_model.pkl")


if __name__ == "__main__":
    main()
