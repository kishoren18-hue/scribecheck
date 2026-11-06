"""ScribeCheck Report Card v0: audit notes written by real note-generation systems.

ACI-Bench (Yim et al., 2023) released the notes that nine systems produced for the same 120 test
encounters, plus the human reference note. We run every note through:

  A. hard checks      deterministic, verifiable facts that contradict or are absent from the visit
                      (drug never mentioned, dose number never said, dose bound to the wrong drug,
                      side stated when only the opposite side was discussed)
  B. auditor score    SC+R sentence-level error probability (trained only on ScribeErr-train)
  C. epistemic audit  certainty inflation / Hedge Preservation Rate (scribecheck.epistemic)

Every metric gets an encounter-level bootstrap CI, and systems get bootstrap rank intervals,
because a ranking without uncertainty is the thing vendor leaderboards get wrong.
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scribecheck import epistemic as E  # noqa: E402
from scribecheck import features as F  # noqa: E402
from scribecheck import hardchecks as H  # noqa: E402
from scribecheck import text as T  # noqa: E402

PRED = ROOT.parent / "aci-bench/baselines/predictions"
CD = ROOT / "data/raw/aci-bench/challenge_data"
P, R = ROOT / "data/processed", ROOT / "results"
SETS = {"test1": "clinicalnlp_taskB_test1", "test2": "clinicalnlp_taskC_test2", "test3": "clef_taskC_test3"}
LLM = {"GPT-4": "GPT-4", "ChatGPT": "ChatGPT", "Davinci-003": "Text-Davinci-003", "Davinci-002": "Text-Davinci-002"}
FT = {"BART-large": "BART_large", "BART-xsum-samsum": "bart-large-xsum-samsum", "BioBART": "BioBART",
      "LED": "LED", "LED-pubmed": "LED_pubmed"}
HEADER_FIX = re.compile(r"\b(" + "|".join(sorted(T.SECTION_CANON, key=len, reverse=True)) + r")\s*:\s*", re.I)


def normalize_note(note: str) -> str:
    """LLM notes put headers inline ('HISTORY OF PRESENT ILLNESS: The patient...') and chain
    bullets on one line; put both on their own lines so the section splitter sees them."""
    note = re.sub(r"^\s*(possible|sample|here is a(?:n)?)?\s*clinical note\s*:?", "", str(note), flags=re.I)
    note = HEADER_FIX.sub(lambda m: "\n" + m.group(1).upper() + "\n", note)
    note = re.sub(r"\s+-\s+(?=[A-Z])", "\n- ", note)
    return note.strip()


def load_systems():
    dlg = pd.concat([pd.read_csv(CD / f"{v}.csv") for v in SETS.values()])
    dlg = dlg.drop_duplicates("encounter_id").set_index("encounter_id")
    rows = []
    for eid, r in dlg.iterrows():
        rows.append(dict(system="Human reference", encounter=eid, dataset=r.dataset, note=r.note))
    for k, (tn, setname) in enumerate(SETS.items()):
        for name, f in LLM.items():
            d = pd.read_csv(PRED / f"{tn}{f}_.csv")
            rows += [dict(system=name, encounter=x.encounter_id, dataset=dlg.loc[x.encounter_id].dataset,
                          note=x.note) for x in d.itertuples()]
        for name, f in FT.items():
            fp = PRED / f"{f}_{setname}_full.csv"
            if not fp.exists():
                fp = PRED / f"{f}_{setname}_division_combined.csv"
            d = pd.read_csv(fp)
            rows += [dict(system=name, encounter=x.encounter_id, dataset=dlg.loc[x.encounter_id].dataset,
                          note=x.note) for x in d.itertuples()]
    df = pd.DataFrame(rows)
    df["dialogue"] = df.encounter.map(dlg.dialogue)
    df["note"] = df.note.fillna("").map(normalize_note)
    return df


CHECKS = ["DRUG_UNSAID", "DOSE_UNSAID", "DOSE_MISBOUND", "SIDE_CONFLICT"]


def boot_ci(per_enc, encs, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = per_enc.reindex(encs).fillna(0).values
    idx = rng.integers(0, len(v), size=(B, len(v)))
    bs = v[idx].mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), idx


def main():
    df = load_systems()
    print(df.groupby("system").size())
    wv, cv = pickle.load(open(P / "vectorizers.pkl", "rb"))
    fz = F.Featurizer(wv, cv, reversal=True)
    models = pickle.load(open(P / "detectors.pkl", "rb"))
    m, cols, iso = models["SC+R"]

    tr = pd.concat([pd.read_csv(CD / "train.csv"), pd.read_csv(CD / "valid.csv")])
    ap = [s.text for n in tr.note for s in T.split_note(n) if s.section == "ap" and not s.is_header]
    dx_vocab = E.build_dx_vocab()  # fixed lexicon; E.build_dx_vocab(ap) is the data-driven ablation

    sent_rows, feats, epi_rows = [], [], []
    tx_cache, visit_cache, cov_rows = {}, {}, []
    for r in df.itertuples():
        if r.encounter not in tx_cache:
            tx_cache[r.encounter] = fz.transcript(r.dialogue)
            visit_cache[r.encounter] = H.Visit(r.dialogue)
        tx, visit = tx_cache[r.encounter], visit_cache[r.encounter]
        note_drugs = {H.CANON[w] for w in H._toks(H.norm_spoken(r.note)) if w in H.DRUGS}
        cov_rows.append(dict(system=r.system, encounter=r.encounter, visit_drugs=len(visit.drugs),
                             covered=len(visit.drugs & note_drugs)))
        sents = [s for s in T.split_note(r.note) if not s.is_header and len(T.tokens(s.text)) >= 3]
        for e in E.audit_note([(s.section, s.text) for s in sents], r.dialogue, dx_vocab):
            epi_rows.append(dict(system=r.system, encounter=r.encounter, **e))
        if not sents:
            continue
        X = fz.sentences([dict(text=s.text, section=s.section) for s in sents], tx)
        feats.append(X)
        for s in sents:
            v = H.check_sentence(s.text, visit)
            row = dict(system=r.system, encounter=r.encounter, dataset=r.dataset, section=s.section,
                       text=s.text, hc_detail="; ".join(f"{a}:{b}" for a, b in v))
            for c in CHECKS:
                row["hc_" + c] = any(a == c for a, _ in v)
            sent_rows.append(row)
    S = pd.concat([pd.DataFrame(sent_rows),
                   pd.DataFrame(np.vstack(feats), columns=["f_" + n for n in F.FEATURE_NAMES])], axis=1)
    S["p_err"] = iso.predict(m.predict_proba(S[cols])[:, 1])
    S["hc_any"] = S[[c for c in S.columns if c.startswith("hc_")]].any(axis=1)
    keep = [c for c in S.columns if not c.startswith("f_")] + ["f_simw_max", "f_unsup_any"]
    S[keep].to_parquet(R / "real_sentences.parquet")
    EP = pd.DataFrame(epi_rows)
    EP.to_csv(R / "real_epistemic.csv", index=False)

    # ---- per-note metrics
    note = S.groupby(["system", "encounter"]).agg(
        n_sent=("text", "size"), exp_err=("p_err", "sum"), hc_any=("hc_any", "sum"),
        **{"hc_" + c: ("hc_" + c, "sum") for c in CHECKS}).reset_index()
    note = note.merge(pd.DataFrame(cov_rows), on=["system", "encounter"], how="right").fillna(0)
    # one claim per (system, encounter, doctor turn): multi-word diagnoses ("rotator cuff") count once;
    # a claim is preserved if any of its terms is hedged in the note, omitted only if all are absent
    rank = {"preserved": 0, "inflated": 1, "omitted": 2}
    EP["_r"] = EP.status.map(rank)
    CL = EP.sort_values("_r").groupby(["system", "encounter", "transcript_turn"], as_index=False).first()
    CL.drop(columns="_r").to_csv(R / "real_epistemic_claims.csv", index=False)
    ep = CL[CL.status != "omitted"].assign(inf=lambda d: d.status == "inflated")
    epn = ep.groupby(["system", "encounter"]).agg(n_hedged=("inf", "size"), n_inflated=("inf", "sum")).reset_index()
    note = note.merge(epn, on=["system", "encounter"], how="left").fillna({"n_hedged": 0, "n_inflated": 0})
    note.to_csv(R / "real_notes.csv", index=False)

    encs = sorted(df.encounter.unique())
    systems = sorted(note.system.unique())
    rows, boot = [], {}
    for s in systems:
        d = note[note.system == s].set_index("encounter")
        row = dict(system=s, n_notes=len(d), sentences_per_note=d.n_sent.mean())
        for k in ["hc_any", "exp_err"] + ["hc_" + c for c in CHECKS]:
            mu, lo, hi, _ = boot_ci(d[k], encs)
            row[k + "_per_note"], row[k + "_lo"], row[k + "_hi"] = mu, lo, hi
        # notes with at least one hard-check violation
        mu, lo, hi, idx = boot_ci((d.hc_any > 0).astype(float), encs)
        row["pct_notes_hc"], row["pct_notes_hc_lo"], row["pct_notes_hc_hi"] = mu, lo, hi
        row["hc_per_100_sent"] = 100 * d.hc_any.sum() / max(d.n_sent.sum(), 1)
        row["drug_coverage"] = d.covered.sum() / max(d.visit_drugs.sum(), 1)
        boot[s] = (d.hc_any.reindex(encs).fillna(0).values, idx)
        # Hedge Preservation Rate: ratio estimator with encounter bootstrap
        hd = d.n_hedged.reindex(encs).fillna(0).values
        inf = d.n_inflated.reindex(encs).fillna(0).values
        rng = np.random.default_rng(1)
        bi = rng.integers(0, len(encs), size=(2000, len(encs)))
        hpr = 1 - inf.sum() / max(hd.sum(), 1)
        bs = 1 - inf[bi].sum(1) / np.maximum(hd[bi].sum(1), 1)
        row.update(hedged_claims=int(hd.sum()), HPR=hpr, HPR_lo=np.percentile(bs, 2.5), HPR_hi=np.percentile(bs, 97.5))
        om = CL[(CL.system == s)]
        row["hedged_omitted_pct"] = (om.status == "omitted").mean() if len(om) else np.nan
        rows.append(row)
    card = pd.DataFrame(rows)

    # bootstrap rank intervals on hard-check violations per note (lower is better), shared resamples
    rng = np.random.default_rng(2)
    bi = rng.integers(0, len(encs), size=(2000, len(encs)))
    M = np.vstack([boot[s][0][bi].mean(1) for s in systems])  # systems x B
    ranks = M.argsort(0).argsort(0) + 1
    card["rank_hc"] = [int(np.median(ranks[i])) for i in range(len(systems))]
    card["rank_lo"] = [int(np.percentile(ranks[i], 2.5)) for i in range(len(systems))]
    card["rank_hi"] = [int(np.percentile(ranks[i], 97.5)) for i in range(len(systems))]
    card = card.sort_values("hc_any_per_note")
    card.to_csv(R / "report_card.csv", index=False)
    pd.set_option("display.width", 250)
    print(card[["system", "n_notes", "sentences_per_note", "hc_any_per_note", "hc_any_lo", "hc_any_hi",
                "pct_notes_hc", "hc_per_100_sent", "drug_coverage"] + ["hc_%s_per_note" % c for c in CHECKS] + [
                "exp_err_per_note", "hedged_claims", "HPR", "HPR_lo", "HPR_hi",
                "hedged_omitted_pct", "rank_hc", "rank_lo", "rank_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
