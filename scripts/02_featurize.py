"""Featurize every benchmark file -> data/processed/feat_<name>.parquet (one row per scorable sentence)."""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scribecheck import features as F  # noqa: E402

P = ROOT / "data/processed"
_FZ = None


def init(vec_path):
    global _FZ
    wv, cv = pickle.load(open(vec_path, "rb"))
    _FZ = F.Featurizer(wv, cv, reversal=True)


def work(args):
    iid, line = args
    r = json.loads(line)
    tx = _FZ.transcript(r["dialogue"])
    sents = [s for s in r["sentences"] if s["scorable"]]
    if not sents:
        return None
    X = _FZ.sentences(sents, tx)
    meta = [dict(iid=iid, corpus=r["corpus"], subset=r["subset"], encounter=r["encounter"], seed=r["seed"],
                 mix=r["mix"], asr=str(r["asr"]), pos=s["pos"], label=s["label"], etype=s["etype"] or "none",
                 severity=s["severity"] or "none", sev_w=s["sev_w"], grounded=bool(s["grounded"]),
                 section=s["section"], text=s["text"]) for s in sents]
    return meta, X, F.FEATURE_NAMES


def run(name, vec_path, procs=2):
    t = time.time()
    lines = open(P / f"scribeerr_{name}.jsonl").read().splitlines()
    with Pool(procs, initializer=init, initargs=(vec_path,)) as pool:
        out = [o for o in pool.imap(work, enumerate(lines), chunksize=8) if o]
    names = out[0][2]
    meta = pd.DataFrame([m for o in out for m in o[0]])
    X = pd.DataFrame(np.vstack([o[1] for o in out]), columns=["f_" + n for n in names])
    df = pd.concat([meta, X], axis=1)
    df.to_parquet(P / f"feat_{name}.parquet")
    print(f"{name}: {len(df)} sentences, {df.label.sum()} errors, {time.time() - t:.0f}s")


if __name__ == "__main__":
    vec_path = P / "vectorizers.pkl"
    if not vec_path.exists():
        cd = ROOT / "data/raw/aci-bench/challenge_data"
        tr = pd.concat([pd.read_csv(cd / "train.csv"), pd.read_csv(cd / "valid.csv")])
        from scribecheck import text as T
        texts = [T.norm(x) for x in list(tr.dialogue) + list(tr.note)]
        pickle.dump(F.fit_vectorizers(texts), open(vec_path, "wb"))
    for name in sys.argv[1:] or ["train", "paired", "mts", "test"]:
        run(name, vec_path)
