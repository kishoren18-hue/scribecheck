"""Train sentence-level error detectors on ScribeErr-train and score every evaluation set.

Detectors
---------
lexical     fraction of content words absent from the transcript (the usual grounding heuristic)
similarity  1 - best TF-IDF similarity to any transcript window
SC-LR       logistic regression on all evidence features (no reversal feature)
SC          gradient-boosted trees on all evidence features (no reversal feature)
SC+R        SC plus the generic plan-reversal feature (trained on retraction phrasing set A,
            evaluated on disjoint phrasing set B)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
P, R = ROOT / "data/processed", ROOT / "results"
R.mkdir(exist_ok=True)


def feats(df, reversal):
    cols = [c for c in df.columns if c.startswith("f_")]
    if not reversal:
        cols = [c for c in cols if c not in ("f_reversal", "f_plan_med")]
    return cols


def hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight=None, random_state=0)


def ece(y, p, bins=15):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def main():
    tr = pd.read_parquet(P / "feat_train.parquet")
    y, g = tr.label.values, tr.encounter.values
    models = {}
    # out-of-fold predictions on train -> isotonic calibrator (for the ECE comparison only)
    for name, rev, mk in [("SC", False, hgb), ("SC+R", True, hgb),
                          ("SC-LR", False, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)))]:
        cols = feats(tr, rev)
        oof = np.zeros(len(tr))
        for a, b in GroupKFold(5).split(tr, y, g):
            m = mk().fit(tr.iloc[a][cols], y[a])
            oof[b] = m.predict_proba(tr.iloc[b][cols])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
        m = mk().fit(tr[cols], y)
        models[name] = (m, cols, iso)
        print(f"{name:6s} train-OOF AUROC {roc_auc_score(y, oof):.3f} AUPRC {average_precision_score(y, oof):.3f}")
    pickle.dump(models, open(P / "detectors.pkl", "wb"))

    for name in ["test", "paired", "mts"]:
        df = pd.read_parquet(P / f"feat_{name}.parquet")
        df["s_lexical"] = df.f_unsup_any
        df["s_similarity"] = 1 - (0.6 * df.f_simw_max + 0.4 * df.f_simc_max)
        for mname, (m, cols, iso) in models.items():
            raw = m.predict_proba(df[cols])[:, 1]
            df["s_" + mname] = raw
            df["p_" + mname] = iso.predict(raw)
        keep = [c for c in df.columns if not c.startswith("f_")] + ["f_n_med", "f_n_num", "f_bind_n"]
        df[keep].to_parquet(P / f"scored_{name}.parquet")
        print("scored", name, len(df))

    # ---- detection quality on the in-distribution clean-transcript slice
    te = pd.read_parquet(P / "scored_test.parquet")
    base = te[(te.asr == "0.0") & (te.mix == "uniform")]
    rows = []
    for det in ["lexical", "similarity", "SC-LR", "SC", "SC+R"]:
        s = base["s_" + det].values
        row = dict(detector=det, AUROC=roc_auc_score(base.label, s), AUPRC=average_precision_score(base.label, s))
        if det.startswith("SC"):
            row["ECE_raw"] = ece(base.label.values, s)
            row["ECE_iso"] = ece(base.label.values, base["p_" + det].values)
        # per error type: AUROC of that type vs clean sentences
        clean = base[base.label == 0]
        for et in sorted(base.etype.unique()):
            if et == "none":
                continue
            sub = pd.concat([clean, base[base.etype == et]])
            row["AUROC_" + et] = roc_auc_score(sub.label, sub["s_" + det])
        # stratum-matched AUROC: compare each error only against clean sentences of the same
        # (section, mentions-drug, has-number) stratum, so section/drug base rates cannot do the work
        key = base.section + "|" + (base.f_n_med > 0).astype(str) + "|" + (base.f_n_num > 0).astype(str)
        for et in sorted(base.etype.unique()):
            if et == "none":
                continue
            num = den = 0.0
            for k in key[base.etype == et].unique():
                sub = base[(key == k) & ((base.etype == et) | (base.label == 0))]
                if sub.label.nunique() < 2:
                    continue
                w = (sub.label == 1).sum()
                num += w * roc_auc_score(sub.label, sub["s_" + det]); den += w
            row["mAUROC_" + et] = num / den if den else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(R / "detector_quality.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).T)


if __name__ == "__main__":
    main()
