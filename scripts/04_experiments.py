"""Main experiments: does the conformal review guarantee survive deployment shift?

E1  in-distribution validity of CRC (marginal, severity-weighted, high-probability, stratified)
E2  shift: error-mix (subtle / crude vendor), simulated ASR noise, real paired ASR, cross-corpus
E3  where misses concentrate: per error type, grounded vs ungrounded, high-risk stratum
E4  repair: recalibrate on m labeled notes from the deployment site
E5  ASR simulator validation against real paired ASR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scribecheck.conformal import crc_threshold, grid_for, note_contribs, realized  # noqa: E402

P, R = ROOT / "data/processed", ROOT / "results"
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ALPHAS = [0.05, 0.10, 0.20]
HIGH = {"ap", "meds", "allergies", "results"}


def add_stratum(df):
    df = df.copy()
    df["high"] = df.section.isin(HIGH) | (df.f_n_med > 0) | (df.f_bind_n > 0)
    return df


def thresholds(cal, score, alpha, grid):
    n = cal.iid.nunique()
    out = {}
    s, c, _ = note_contribs(cal, score)
    out["CRC"] = crc_threshold(s, c, n, alpha, grid)
    s, c, _ = note_contribs(cal, score, "sev_w")
    out["CRC-sev"] = crc_threshold(s, c, n, alpha, grid)
    s, c, _ = note_contribs(cal, score)
    out["CRC-HP"] = crc_threshold(s, c, n, alpha, grid, delta=0.1)
    hi, lo = cal.high.values, ~cal.high.values
    s, c, _ = note_contribs(cal, score, mask=hi)
    t_hi = crc_threshold(s, c, n, alpha, grid)
    s, c, _ = note_contribs(cal, score, mask=lo)
    t_lo = crc_threshold(s, c, n, alpha, grid)
    out["Strat-CRC"] = (t_hi, t_lo)
    return out


def evaluate(te, score, lam):
    """Return metrics dict for one threshold (scalar or (hi, lo) pair)."""
    d = te
    if isinstance(lam, tuple):
        thr = np.where(d.high.values, lam[0], lam[1])
    else:
        thr = np.full(len(d), lam)
    flagged = d[score].values >= thr
    d = d.assign(_f=flagged)
    err = d.label.values == 1
    m = {}
    tmp = d.assign(_m=(err & ~flagged).astype(float), _t=err.astype(float),
                   _ms=(err & ~flagged) * d.sev_w.values, _ts=err * d.sev_w.values,
                   _mh=(err & ~flagged & d.high.values).astype(float), _th=(err & d.high.values).astype(float))
    g = tmp.groupby("iid")[["_m", "_t", "_ms", "_ts", "_mh", "_th"]].sum()
    m["risk"] = np.where(g._t > 0, g._m / g._t.clip(lower=1e-9), 0).mean()
    m["risk_sev"] = np.where(g._ts > 0, g._ms / g._ts.clip(lower=1e-9), 0).mean()
    m["risk_high"] = np.where(g._th > 0, g._mh / g._th.clip(lower=1e-9), 0).mean()
    m["workload"] = flagged.mean()
    e = d[err]
    m["miss_grounded"] = 1 - e[e.grounded]._f.mean()
    m["miss_ungrounded"] = 1 - e[~e.grounded]._f.mean()
    m["miss_major"] = 1 - e[e.severity == "major"]._f.mean()
    for et, sub in e.groupby("etype"):
        m["miss_" + et] = 1 - sub._f.mean()
    return m


def run_shift(src, targets, score, rng, repeats):
    encs = np.array(sorted(src.encounter.unique()))
    grid = grid_for(src[score].values)
    rows = []
    for rep in range(repeats):
        cal_e = set(rng.choice(encs, size=len(encs) // 2, replace=False))
        cal = src[src.encounter.isin(cal_e)]
        for alpha in ALPHAS:
            th = thresholds(cal, score, alpha, grid)
            for tname, tdf in targets.items():
                te = tdf[~tdf.encounter.isin(cal_e)] if tname != "mts" else tdf
                for meth, lam in th.items():
                    m = evaluate(te, score, lam)
                    rows.append(dict(rep=rep, alpha=alpha, target=tname, method=meth, score=score, **m))
    return pd.DataFrame(rows)


def summarize(df):
    agg = df.groupby(["score", "target", "method", "alpha"]).agg(
        risk=("risk", "mean"), risk_sev=("risk_sev", "mean"), risk_high=("risk_high", "mean"),
        workload=("workload", "mean"), miss_grounded=("miss_grounded", "mean"),
        miss_ungrounded=("miss_ungrounded", "mean"), miss_major=("miss_major", "mean"))
    viol = df.assign(v=df.risk > df.alpha, vh=df.risk_high > df.alpha).groupby(
        ["score", "target", "method", "alpha"])[["v", "vh"]].mean().rename(
        columns={"v": "P(risk>alpha)", "vh": "P(risk_high>alpha)"})
    return agg.join(viol).reset_index()


def main():
    rng = np.random.default_rng(0)
    te = add_stratum(pd.read_parquet(P / "scored_test.parquet"))
    mts = add_stratum(pd.read_parquet(P / "scored_mts.parquet"))
    pa = add_stratum(pd.read_parquet(P / "scored_paired.parquet"))
    src = te[(te.asr == "0.0") & (te.mix == "uniform")]
    targets = {"in-dist": src}
    for mix in ["subtle", "crude"]:
        targets["mix:" + mix] = te[(te.asr == "0.0") & (te.mix == mix)]
    for a in ["0.05", "0.1", "0.2", "0.3"]:
        targets["asr:" + a] = te[(te.asr == a) & (te.mix == "uniform")]
    targets["asr:0.2+subtle"] = te[(te.asr == "0.2") & (te.mix == "subtle")]
    targets["mts"] = mts

    allres = []
    for score in ["s_SC", "s_SC+R"]:
        allres.append(run_shift(src, targets, score, rng, REPEATS))
        print("done", score)
    # real paired ASR: calibrate on human transcripts, deploy on raw ASR of held-out encounters
    for score in ["s_SC", "s_SC+R"]:
        psrc = pa[pa.asr == "human"]
        allres.append(run_shift(psrc, {"paired:human": psrc, "paired:asr_real": pa[pa.asr == "asr_real"]},
                                score, rng, REPEATS))
    res = pd.concat(allres)
    res.to_parquet(R / "shift_raw.parquet")
    summ = summarize(res)
    summ.to_csv(R / "shift_summary.csv", index=False)
    per_type = res.filter(regex="^(score|target|method|alpha|miss_[A-Z].*)$").groupby(
        ["score", "target", "method", "alpha"]).mean().reset_index()
    per_type.to_csv(R / "miss_by_type.csv", index=False)

    # ---- E4: recalibration with m labeled deployment notes (target-only CRC)
    rows = []
    for tname in ["asr:0.2+subtle", "mix:subtle", "mts"]:
        tdf = targets[tname]
        encs = np.array(sorted(tdf.encounter.unique()))
        grid = grid_for(tdf["s_SC+R"].values)
        for m in [10, 20, 40, 80, 160, 300]:
            for rep in range(REPEATS):
                cal_e = rng.choice(encs, size=len(encs) // 2, replace=False)
                pool = tdf[tdf.encounter.isin(cal_e)]
                ids = rng.choice(pool.iid.unique(), size=min(m, pool.iid.nunique()), replace=False)
                cal = pool[pool.iid.isin(ids)]
                hold = tdf[~tdf.encounter.isin(cal_e)]
                s, c, _ = note_contribs(cal, "s_SC+R")
                lam = crc_threshold(s, c, len(ids), 0.10, grid)
                r, w = realized(hold, "s_SC+R", lam)
                rows.append(dict(target=tname, m=len(ids), rep=rep, risk=r, workload=w))
    rc = pd.DataFrame(rows)
    rc_s = rc.groupby(["target", "m"]).agg(risk=("risk", "mean"), workload=("workload", "mean"),
                                           viol=("risk", lambda x: (x > 0.10).mean())).reset_index()
    rc_s.to_csv(R / "recalibration.csv", index=False)

    # ---- E5: detector AUROC and clean-sentence score shift by transcript condition
    rows = []
    for name, d in list(targets.items()) + [("paired:human", pa[pa.asr == "human"]),
                                            ("paired:asr_real", pa[pa.asr == "asr_real"])]:
        rows.append(dict(condition=name, n_notes=d.iid.nunique(), n_sent=len(d), err_rate=d.label.mean(),
                         AUROC_SC=roc_auc_score(d.label, d.s_SC), AUROC_SCR=roc_auc_score(d.label, d["s_SC+R"]),
                         clean_mean_SC=d[d.label == 0].s_SC.mean(),
                         err_mean_SC=d[d.label == 1].s_SC.mean(),
                         grounded_share=d[d.label == 1].grounded.mean()))
    pd.DataFrame(rows).to_csv(R / "conditions.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
