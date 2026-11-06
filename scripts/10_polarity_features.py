"""Compute term-anchored polarity features for every benchmark file and train the SC+N auditor.

Rows align with data/processed/feat_<name>.parquet (same instance order, same scorable filter).
Writes data/processed/pol_<name>.parquet and adds model 'SC+N' to detectors.pkl.
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scribecheck import polarity as PL  # noqa: E402

P = ROOT / "data/processed"


def work(line):
    r = json.loads(line)
    sents = [s for s in r["sentences"] if s["scorable"]]
    if not sents:
        return []
    vp = PL.VisitPolarity(r["dialogue"])
    return [PL.sentence_features(s["text"], vp) for s in sents]


def build(name):
    t = time.time()
    lines = open(P / f"scribeerr_{name}.jsonl").read().splitlines()
    with Pool(2) as pool:
        out = [x for o in pool.imap(work, lines, chunksize=16) for x in o]
    df = pd.DataFrame(np.array(out, dtype=float), columns=["f_" + n for n in PL.NAMES])
    df.to_parquet(P / f"pol_{name}.parquet")
    print(f"{name}: {len(df)} rows, {time.time() - t:.0f}s", flush=True)


def load(name):
    f = pd.read_parquet(P / f"feat_{name}.parquet")
    p = pd.read_parquet(P / f"pol_{name}.parquet")
    assert len(f) == len(p), (name, len(f), len(p))
    return pd.concat([f, p], axis=1)


if __name__ == "__main__":
    for n in ["train", "test", "mts", "paired"]:
        if not (P / f"pol_{n}.parquet").exists():
            build(n)
    models = pickle.load(open(P / "detectors.pkl", "rb"))
    base_cols = models["SC"][1]
    cols = base_cols + ["f_" + n for n in PL.NAMES]
    tr = load("train")
    y, g = tr.label.values, tr.encounter.values

    def mk():
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0)
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(tr, y, g):
        oof[b] = mk().fit(tr.iloc[a][cols], y[a]).predict_proba(tr.iloc[b][cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    m = mk().fit(tr[cols], y)
    models["SC+N"] = (m, cols, iso)
    # SC+RN: retraction (reversal) features + polarity features
    cols_rn = models["SC+R"][1] + ["f_" + n for n in PL.NAMES]
    m_rn = mk().fit(tr[cols_rn], y)
    models["SC+RN"] = (m_rn, cols_rn, None)
    pickle.dump(models, open(P / "detectors.pkl", "wb"))
    te = load("test")
    base = te[(te.asr == "0.0") & (te.mix == "uniform")]
    s_sc = models["SC"][0].predict_proba(base[base_cols])[:, 1]
    s_scn = m.predict_proba(base[cols])[:, 1]
    key = base.section + "|" + (base.f_n_med > 0).astype(str) + "|" + (base.f_n_num > 0).astype(str)
    rows = []
    for et in sorted(base.etype.unique()):
        if et == "none":
            continue
        res = {}
        for nm, s in [("SC", s_sc), ("SC+N", s_scn)]:
            num = den = 0.0
            for k in key[base.etype == et].unique():
                msk = ((key == k) & ((base.etype == et) | (base.label == 0))).values
                if base.label.values[msk].min() == base.label.values[msk].max():
                    continue
                w = (base.etype.values[msk] == et).sum()
                num += w * roc_auc_score(base.label.values[msk], s[msk]); den += w
            res[nm] = num / den
        rows.append(dict(etype=et, mAUROC_SC=res["SC"], mAUROC_SCN=res["SC+N"]))
    out = pd.DataFrame(rows)
    out.loc[len(out)] = dict(etype="ALL", mAUROC_SC=roc_auc_score(base.label, s_sc),
                             mAUROC_SCN=roc_auc_score(base.label, s_scn))
    out.to_csv(ROOT / "results/negation_aware_detector.csv", index=False)
    print(out.round(3).to_string())
