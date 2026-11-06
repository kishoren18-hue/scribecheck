"""Mix-robust review guarantees and a shift alarm.

Problem (Section 4): marginal CRC bounds the *average* share of errors a reviewer never sees, but a
scribe whose errors skew toward grounded or negation errors breaks it, and in distribution it
already misses 1 grounded error in 4.

Fix 1, type-conditional CRC (TC-CRC). Partition errors into families g in {invented (U_*),
negation (NEG_FLIP), grounded (G_*, RETRACT)}. Under error-mix shift, a family's errors look the
same to the auditor; only the family proportions change (label-shift assumption). For each family,
CRC picks lambda_g so that the family's miss rate is <= alpha (sentence-level exchangeability within
the family). Using lambda = min_g lambda_g gives miss rate <= alpha in every family, hence for ANY
mixture of families. No deployment labels and no estimation are needed.

Fix 2, the trust dial MR-CRC(h). Guarantee the miss rate for every error mix whose hard share
(negation + grounded) is at most h:  (1 - h) * miss_U + h * max(miss_N, miss_G) <= alpha, each term
with its finite-sample correction. h = 0 is CRC on invented errors only; h = 1 is TC-CRC over the
hard families. A buyer sets h from what they are willing to assume about a vendor.

Shift alarm. A 4-way family classifier trained on ScribeErr-train, its probabilities bias-corrected
on the labeled calibration notes (vector scaling) and re-estimated on unlabeled deployment notes by
EM (Saerens et al., 2002; Alexandari et al., 2020). The alarm fires when the estimated hard share of
deployment errors exceeds the 95th percentile seen on in-distribution deployment draws.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scribecheck.conformal import grid_for  # noqa: E402

P, R = ROOT / "data/processed", ROOT / "results"
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DET = sys.argv[2] if len(sys.argv) > 2 else "SC"
ALPHA = 0.10
FAM = {"none": 0, "NEG_FLIP": 2}
FAM.update({t: 1 for t in ["U_DOSE", "U_NUM", "U_MED", "U_LAT", "U_TIME", "U_FAB"]})
FAM.update({t: 3 for t in ["G_NUM", "G_MED", "G_LAT", "RETRACT"]})
FNAME = {1: "invented", 2: "negation", 3: "grounded"}


def crc_from(scores_err, contrib, n, alpha, grid, B=1.0):
    order = np.argsort(scores_err)
    s, cc = scores_err[order], np.cumsum(contrib[order])
    k = np.searchsorted(s, grid, side="left")
    r = np.where(k > 0, cc[np.maximum(k - 1, 0)], 0.0) / max(n, 1)
    ok = (n * r + B) / (n + 1) <= alpha
    return grid[np.where(ok)[0].max()] if ok.any() else grid[0], r


def miss_curve(scores_err, grid):
    """Finite-sample-corrected miss rate (n*miss + 1)/(n + 1) of one family, on the grid."""
    s = np.sort(scores_err)
    n = len(s)
    k = np.searchsorted(s, grid, side="left")
    return (k + 1) / (n + 1)


def gates(cal, grid, hs):
    e = cal[cal.label == 1]
    n_notes = cal.iid.nunique()
    c = 1.0 / e.groupby("iid").label.transform("size").values
    out = {}
    out["CRC"], _ = crc_from(e.score.values, c, n_notes, ALPHA, grid)
    curves = {f: miss_curve(e[e.fam == f].score.values, grid) for f in (1, 2, 3)}
    ok_tc = np.all([curves[f] <= ALPHA for f in (1, 2, 3)], axis=0)
    out["TC-CRC"] = grid[np.where(ok_tc)[0].max()] if ok_tc.any() else grid[0]
    ok_nn = np.all([curves[f] <= ALPHA for f in (1, 3)], axis=0)   # diagnostic: negation left out
    out["TC-CRC(no-neg)"] = grid[np.where(ok_nn)[0].max()] if ok_nn.any() else grid[0]
    hard = np.maximum(curves[2], curves[3])
    for h in hs:
        ok = (1 - h) * curves[1] + h * hard <= ALPHA
        out[f"MR-CRC(h={h})"] = grid[np.where(ok)[0].max()] if ok.any() else grid[0]
    return out


def realized(d, lam):
    f = d.score.values >= lam
    err = d.label.values == 1
    t = d.assign(_m=(err & ~f).astype(float), _t=err.astype(float)).groupby("iid")[["_m", "_t"]].sum()
    out = dict(risk=np.where(t._t > 0, t._m / t._t.clip(lower=1e-9), 0).mean(), workload=f.mean(),
               miss_all=1 - f[err].mean())
    for k, name in FNAME.items():
        m = d.fam.values == k
        out["miss_" + name] = 1 - f[m].mean() if m.any() else np.nan
    return out


def em(post, prior, it=500):
    q = prior.copy()
    for _ in range(it):
        a = post * (q / prior)
        a /= a.sum(1, keepdims=True)
        qn = a.mean(0)
        if np.abs(qn - q).max() < 1e-8:
            break
        q = qn
    return q


def main():
    det, cols, _ = pickle.load(open(P / "detectors.pkl", "rb"))[DET]
    def read(name):
        f = pd.read_parquet(P / f"feat_{name}.parquet")
        pol = P / f"pol_{name}.parquet"
        return pd.concat([f, pd.read_parquet(pol)], axis=1) if pol.exists() else f

    tr = read("train")
    fcols = [c for c in tr.columns if c.startswith("f_")]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(tr[fcols], tr.etype.map(FAM).values)

    def prep(df):
        df = df.copy()
        df["score"] = det.predict_proba(df[cols])[:, 1]
        df["fam"] = df.etype.map(FAM).fillna(0).astype(int)
        df["logp"] = list(np.log(np.clip(clf.predict_proba(df[fcols]), 1e-6, 1)))
        return df[["iid", "encounter", "label", "fam", "score", "logp", "asr", "mix"]]

    te = prep(read("test"))
    mts = prep(read("mts"))
    src = te[(te.asr == "0.0") & (te.mix == "uniform")]
    targets = {"in-dist": src,
               "mix:subtle": te[(te.asr == "0.0") & (te.mix == "subtle")],
               "mix:crude": te[(te.asr == "0.0") & (te.mix == "crude")],
               "asr:0.2": te[(te.asr == "0.2") & (te.mix == "uniform")],
               "asr:0.2+subtle": te[(te.asr == "0.2") & (te.mix == "subtle")],
               "mts": mts}
    targets = {k: v.reset_index(drop=True) for k, v in targets.items()}
    src = targets["in-dist"]
    grid = grid_for(src.score.values, k=3000)
    encs = np.array(sorted(src.encounter.unique()))
    hs = [0.4, 0.6, 0.8]
    rng = np.random.default_rng(11)
    rows, alarm_rows = [], []
    for rep in range(REPEATS):
        cal_e = set(rng.choice(encs, size=len(encs) // 2, replace=False))
        cal = src[src.encounter.isin(cal_e)]
        G = gates(cal, grid, hs)
        Lc = np.vstack(cal.logp.values)
        vs = LogisticRegression(C=10, max_iter=3000).fit(Lc, cal.fam.values)
        p_cal = np.bincount(cal.fam.values, minlength=4) / len(cal)
        q_cal = em(vs.predict_proba(Lc), p_cal)
        hard_cal = q_cal[2:].sum() / q_cal[1:].sum()
        rate_cal = q_cal[1:].sum()
        for tname, tdf in targets.items():
            dep = tdf[~tdf.encounter.isin(cal_e)] if tname != "mts" else tdf
            for meth, lam in G.items():
                rows.append(dict(rep=rep, target=tname, method=meth, **realized(dep, lam)))
            q = em(vs.predict_proba(np.vstack(dep.logp.values)), p_cal)
            t = np.bincount(dep.fam.values, minlength=4) / len(dep)
            alarm_rows.append(dict(rep=rep, target=tname, hard_est=q[2:].sum() / q[1:].sum(),
                                   hard_ratio=(q[2:].sum() / q[1:].sum()) / hard_cal,
                                   hard_true=t[2:].sum() / t[1:].sum(), err_rate_est=q[1:].sum(),
                                   rate_ratio=q[1:].sum() / rate_cal,
                                   err_rate_true=t[1:].sum()))
        if rep % 25 == 0:
            print("rep", rep, flush=True)
    df = pd.DataFrame(rows)
    A = pd.DataFrame(alarm_rows)
    # two-signal alarm: estimated hard share OR estimated error rate moved; each threshold at the
    # 97.5th percentile of in-distribution draws, so the combined false-alarm rate is about 5%
    ind = A[A.target == "in-dist"]
    thr = ind.hard_ratio.quantile(0.975)
    thr_r = ind.rate_ratio.quantile(0.975)
    A["fire"] = (A.hard_ratio > thr) | (A.rate_ratio > thr_r)
    A["fire_mix_only"] = A.hard_ratio > ind.hard_ratio.quantile(0.95)
    A.to_parquet(R / f"shift_alarm_raw_{DET}.parquet")
    fire = A[["rep", "target", "fire"]]
    base = df[df.method == "CRC"].merge(fire, on=["rep", "target"])
    tc = df[df.method == "TC-CRC"].set_index(["rep", "target"])
    pol = base.set_index(["rep", "target"])
    cols_m = ["risk", "workload", "miss_all", "miss_invented", "miss_negation", "miss_grounded"]
    pol.loc[pol.fire, cols_m] = tc.loc[pol.index[pol.fire], cols_m].values
    pol = pol.reset_index().assign(method="Alarm-gated").drop(columns="fire")
    df = pd.concat([df, pol], ignore_index=True)
    df.to_parquet(R / f"mixrobust_raw_{DET}.parquet")
    summ = df.groupby(["target", "method"]).agg(
        risk=("risk", "mean"), viol=("risk", lambda x: (x > ALPHA).mean()), workload=("workload", "mean"),
        miss_invented=("miss_invented", "mean"), miss_negation=("miss_negation", "mean"),
        miss_grounded=("miss_grounded", "mean")).reset_index()
    summ.to_csv(R / f"mixrobust_summary_{DET}.csv", index=False)
    al = A.groupby("target").agg(hard_true=("hard_true", "mean"), hard_est=("hard_est", "mean"),
                                 hard_ratio=("hard_ratio", "mean"), err_rate_true=("err_rate_true", "mean"),
                                 err_rate_est=("err_rate_est", "mean"),
                                 rate_ratio=("rate_ratio", "mean"), alarm_mix_only=("fire_mix_only", "mean"),
                                 alarm_rate=("fire", "mean")).reset_index()
    al["thr_mix"], al["thr_rate"] = thr, thr_r
    al.to_csv(R / f"shift_alarm_{DET}.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string())
    print(al.round(3).to_string())


if __name__ == "__main__":
    main()
