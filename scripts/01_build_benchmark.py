"""Build the ScribeErr benchmark (JSONL of corrupted note instances).

Outputs data/processed/scribeerr_<split>.jsonl with one record per (encounter, seed, mix,
transcript condition). Each record carries the transcript the detector will see and the
sentence list with ground-truth error labels, type, severity, and grounded flag.
"""
import json
import zlib
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scribecheck import text as T  # noqa: E402
from scribecheck.asr_noise import add_asr_noise  # noqa: E402
from scribecheck.inject import corrupt_note  # noqa: E402

RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)


def aci_frames():
    cd = RAW / "aci-bench/challenge_data"
    frames = {
        "train": pd.concat([pd.read_csv(cd / "train.csv"), pd.read_csv(cd / "valid.csv")]),
        "test": pd.concat([pd.read_csv(cd / f) for f in
                           ["clinicalnlp_taskB_test1.csv", "clinicalnlp_taskC_test2.csv", "clef_taskC_test3.csv"]]),
    }
    return frames


def paired_virtscribe():
    """Real paired transcripts for the same encounters: human transcription vs raw ASR."""
    sd = RAW / "aci-bench/src_experiment_data"
    rows = []
    for s in ["valid", "test1", "test2", "test3"]:
        h = pd.read_csv(sd / f"{s}_virtscribe_humantrans.csv")
        a = pd.read_csv(sd / f"{s}_virtscribe_asr.csv")
        m = h.merge(a, on="id", suffixes=("_h", "_a"))
        for _, r in m.iterrows():
            rows.append(dict(id=r.id, note=r.note_h, human=r.dialogue_h, asr=r.dialogue_a))
    return rows


def write(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs):5d} instances -> {path.name}")


def build_aci(df, split, seeds, mixes, retraction_set, asr_rates=(0.0,)):
    recs = []
    for _, row in df.iterrows():
        sents = T.split_note(row.note)
        for seed in seeds:
            for mix in mixes:
                rng = random.Random(zlib.crc32(repr((row.encounter_id, seed, mix)).encode()))
                sent_recs, dlg = corrupt_note(sents, row.dialogue, rng, mix=mix, retraction_set=retraction_set)
                for rate in asr_rates:
                    d = dlg if rate == 0 else add_asr_noise(dlg, rate, seed=seed * 1000 + int(rate * 100))
                    recs.append(dict(corpus="aci", subset=row.dataset, encounter=row.encounter_id,
                                     seed=seed, mix=mix, asr=rate, split=split, dialogue=d,
                                     sentences=sent_recs))
    return recs


def build_paired(rows, seeds, mixes):
    recs = []
    for r in rows:
        sents = T.split_note(r["note"])
        for seed in seeds:
            for mix in mixes:
                rng = random.Random(zlib.crc32(repr((r["id"], seed, mix)).encode()))
                sent_recs, dlg_h = corrupt_note(sents, r["human"], rng, mix=mix, retraction_set="B")
                # replay the same retraction (if any) on the ASR transcript: insert the identical
                # retraction turns at the same relative position
                dlg_a = r["asr"]
                if any(x["etype"] == "RETRACT" for x in sent_recs):
                    h_turns, nh = T.split_turns(dlg_h), T.split_turns(r["human"])
                    added = [t for t in h_turns if t not in nh]
                    a_turns = T.split_turns(r["asr"])
                    pos = next(i for i, t in enumerate(h_turns) if t == added[0])
                    pos_a = int(round(pos / max(1, len(h_turns)) * len(a_turns)))
                    dlg_a = T.join_turns(a_turns[:pos_a] + added + a_turns[pos_a:])
                for cond, d in (("human", dlg_h), ("asr_real", dlg_a)):
                    recs.append(dict(corpus="aci-virtscribe-paired", subset="virtscribe", encounter=r["id"],
                                     seed=seed, mix=mix, asr=cond, split="paired", dialogue=d,
                                     sentences=sent_recs))
    return recs


def build_mts(seeds, mix="uniform", max_per_split=None):
    md = RAW / "mts-dialog"
    files = {"mts_test": ["MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv", "MTS-Dialog-ValidationSet.csv"]}
    recs = []
    for split, fs in files.items():
        df = pd.concat([pd.read_csv(md / f) for f in fs])
        df = df[df.section_text.str.len() > 120]  # need multi-sentence sections
        for _, row in df.iterrows():
            sec = T.MTS_SECTION.get(str(row.section_header).upper(), "other")
            sents = T.split_text_section(row.section_text, sec)
            if len(sents) < 2:
                continue
            for seed in seeds:
                rng = random.Random(zlib.crc32(repr(("mts", row.ID, seed)).encode()))
                sent_recs, dlg = corrupt_note(sents, row.dialogue, rng, mix=mix,
                                              n_errors=None if rng.random() > 0.2 else 0,
                                              retraction_set="B")
                recs.append(dict(corpus="mts", subset=sec, encounter=f"MTS{row.ID}", seed=seed, mix=mix,
                                 asr=0.0, split=split, dialogue=dlg, sentences=sent_recs))
    return recs


if __name__ == "__main__":
    fr = aci_frames()
    write(OUT / "scribeerr_train.jsonl", build_aci(fr["train"], "train", seeds=range(20), mixes=["uniform"],
                                                   retraction_set="A"))
    write(OUT / "scribeerr_test.jsonl", build_aci(fr["test"], "test", seeds=range(5),
                                                  mixes=["uniform", "subtle", "crude"], retraction_set="B",
                                                  asr_rates=(0.0, 0.05, 0.10, 0.20, 0.30)))
    write(OUT / "scribeerr_paired.jsonl", build_paired(paired_virtscribe(), seeds=range(10), mixes=["uniform"]))
    write(OUT / "scribeerr_mts.jsonl", build_mts(seeds=range(6)))
