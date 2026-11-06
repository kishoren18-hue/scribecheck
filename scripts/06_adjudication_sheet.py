"""Export every hard-check flag (human reference + GPT-4) and a per-system sample, with transcript
context, for adjudication. Labels go in results/hardcheck_labels.csv (aid, label, note)."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scribecheck import hardchecks as H  # noqa: E402

R = ROOT / "results"
CD = ROOT / "data/raw/aci-bench/challenge_data"
S = pd.read_parquet(R / "real_sentences.parquet")
dl = pd.concat([pd.read_csv(CD / f"{f}.csv") for f in ["clinicalnlp_taskB_test1", "clinicalnlp_taskC_test2",
                                                       "clef_taskC_test3"]]).drop_duplicates("encounter_id")
dl = dl.set_index("encounter_id").dialogue
h = S[S.hc_any].copy()
full = h[h.system.isin(["Human reference", "GPT-4"])]
rest = pd.concat([d.sample(min(8, len(d)), random_state=1) for _, d in h[~h.system.isin(["Human reference", "GPT-4"])].groupby("system")])
samp = pd.concat([full, rest]).reset_index(drop=True)
samp["aid"] = range(len(samp))


def context(eid, detail):
    d = H.norm_spoken(dl[eid])
    keys = set()
    for item in detail.split("; "):
        c, v = item.split(":", 1)
        w = v.split()[0]
        keys |= {x for x in H.DRUGS if H.CANON.get(x) == H.CANON.get(w)} if w in H.DRUGS else {w}
        if c.startswith("SIDE"):
            keys |= {"left", "right"}
        if c.startswith("DOSE") and len(v.split()) > 1:
            keys.add(v.split()[1])
    sn = []
    for k in keys:
        for m in list(re.finditer(r"\b" + re.escape(k) + r"\b", d))[:2]:
            sn.append(d[max(0, m.start() - 70): m.end() + 45])
    return " || ".join(sn[:4]) if sn else "(no mention of these terms anywhere in the visit)"


samp["transcript_context"] = [context(r.encounter, r.hc_detail) for r in samp.itertuples()]
samp[["aid", "system", "encounter", "section", "text", "hc_detail", "transcript_context"]].to_csv(
    R / "hardcheck_adjudication_sample.csv", index=False)
if __name__ == "__main__":
    for r in samp.itertuples():
        print(f"[{r.aid}] {r.system} {r.encounter} {r.hc_detail} | {r.text[:200]}\n     > {r.transcript_context[:420]}")
