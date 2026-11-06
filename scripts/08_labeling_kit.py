"""Clinician labeling kit: build blinded labeling sheets, then score inter-rater reliability.

  python scripts/08_labeling_kit.py build --n 200 --raters 3   -> labeling/sheet_rater{1..k}.csv
  python scripts/08_labeling_kit.py score labeling/sheet_rater*.csv  -> agreement report

Sampling: every hard-check flag, every hedged-diagnosis claim, plus a stratified random sample of
unflagged sentences (so the corpus also measures what the auditor MISSES). System names are hidden
and row order is shuffled per rater, so labelers cannot learn which vendor wrote what.

Labels (see labeling/RUBRIC.md):
  support   SUPPORTED | UNSUPPORTED | CONTRADICTED | TRANSCRIPT_ERROR
  severity  NONE | MINOR | MODERATE | MAJOR          (only if not SUPPORTED)
  certainty KEPT | INFLATED | DEFLATED | N/A        (diagnostic statements only)
"""
import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "labeling"
CD = ROOT / "data/raw/aci-bench/challenge_data"


def build(n, raters, seed=0):
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    S = pd.read_parquet(R / "real_sentences.parquet")
    S = S[~S.system.isin(["LED", "LED-pubmed"])]
    flagged = S[S.hc_any]
    unflagged = S[~S.hc_any]
    # bias the random sample toward high auditor scores so labels inform the score threshold too
    w = (unflagged.p_err + 0.02).values
    pick = rng.choice(len(unflagged), size=min(n, len(unflagged)), replace=False, p=w / w.sum())
    C = pd.read_csv(R / "real_epistemic_claims.csv")
    C = C[(C.status != "omitted") & ~C.system.isin(["LED", "LED-pubmed"])]
    hedge = C.rename(columns={"note_sentence": "text"})[["system", "encounter", "text"]]
    items = pd.concat([flagged[["system", "encounter", "section", "text"]].assign(source="hard_check"),
                       unflagged.iloc[pick][["system", "encounter", "section", "text"]].assign(source="random_weighted"),
                       hedge.assign(section="", source="hedge_claim")]).drop_duplicates(["system", "encounter", "text"])
    items = items.reset_index(drop=True)
    items["item_id"] = [f"SC{i:05d}" for i in range(len(items))]
    items[["item_id", "system", "encounter", "source"]].to_csv(OUT / "key_DO_NOT_SHARE.csv", index=False)
    for r in range(1, raters + 1):
        sh = items.sample(frac=1, random_state=seed + r)[["item_id", "encounter", "section", "text"]].copy()
        sh["support"] = ""; sh["severity"] = ""; sh["certainty"] = ""; sh["comment"] = ""
        sh.to_csv(OUT / f"sheet_rater{r}.csv", index=False)
    print(f"{len(items)} items -> {raters} sheets in {OUT}. Transcripts: {CD}")


def krippendorff_alpha_nominal(matrix):
    """matrix: items x raters, values are category strings or NaN."""
    pairs, cats = [], set()
    for row in matrix:
        v = [x for x in row if isinstance(x, str) and x]
        if len(v) < 2:
            continue
        cats.update(v)
        for a, b in itertools.permutations(v, 2):
            pairs.append((a, b, 1 / (len(v) - 1)))
    if not pairs:
        return np.nan
    cats = sorted(cats)
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((len(cats), len(cats)))
    for a, b, w in pairs:
        O[idx[a], idx[b]] += w
    n_c = O.sum(1); n = n_c.sum()
    Do = (O.sum() - np.trace(O)) / n
    De = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return 1 - Do / De if De > 0 else 1.0


def score(files):
    sheets = [pd.read_csv(f).set_index("item_id") for f in files]
    ids = sorted(set.intersection(*[set(s.index) for s in sheets]))
    rep = []
    for col in ["support", "severity", "certainty"]:
        M = np.array([[s.loc[i, col] if isinstance(s.loc[i, col], str) else np.nan for s in sheets] for i in ids],
                     dtype=object)
        done = sum(sum(isinstance(x, str) and x != "" for x in row) >= 2 for row in M)
        rep.append(dict(field=col, items_with_2plus_labels=done, krippendorff_alpha=krippendorff_alpha_nominal(M)))
    out = pd.DataFrame(rep)
    print(out.to_string(index=False))
    out.to_csv(ROOT / "labeling/agreement.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--n", type=int, default=200); b.add_argument("--raters", type=int, default=3)
    s = sub.add_parser("score"); s.add_argument("files", nargs="+")
    a = ap.parse_args()
    build(a.n, a.raters) if a.cmd == "build" else score(sorted(set(sum([glob.glob(f) for f in a.files], []))))
