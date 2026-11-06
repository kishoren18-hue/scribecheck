"""Sentence-level evidence features: how well is each note sentence supported by the transcript?

Groups
------
retrieval   word/char TF-IDF similarity to the best 3-turn transcript window
support     unsupported content tokens / numbers / drugs / symptoms, anywhere vs in-window
binding     is the dose next to this drug in the note also next to this drug in the visit?
laterality  is the stated side present in the matched window, or only the opposite side?
polarity    note negation vs transcript evidence polarity (incl. doctor-question/patient-answer)
reversal    (+R variant only) was the drug's last mention in the visit a stop/hold/cancel turn?
"""
import re

import numpy as np

from . import text as T
from .lexicon import NEGATION_CUES, SYMPTOMS

SECTIONS = ["cc", "hpi", "ros", "exam", "results", "ap", "pmh", "meds", "other"]
YES = re.compile(r"^\s*(yes|yeah|yep|yup|mm-hmm|uh-huh|i do|i have|sometimes|correct|right|a little|definitely)\b")
NO = re.compile(r"^\s*(no|nope|nah|not really|none|never|i don't|i haven't|denies|negative)\b")
NEG_RE = re.compile(r"(?<![a-z])(" + "|".join(re.escape(c) for c in NEGATION_CUES) + r")(?![a-z])")
REVERSAL_RE = re.compile(r"(?<![a-z])(hold off|hold|stop|cancel|not do|won't start|take back|scratch that|"
                         r"rather not|don't want|discontinue|after all|instead|never mind|revisit)(?![a-z])")
DOSE_AFTER = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|milligrams?|mcg|micrograms?|units?)")

FEATURE_NAMES = None


class Featurizer:
    def __init__(self, word_vec, char_vec, reversal=False):
        self.wv, self.cv, self.reversal = word_vec, char_vec, reversal

    def transcript(self, dialogue):
        turns = T.split_turns(dialogue)
        nt = [T.norm(t) for _, t in turns]
        wins = [" ".join(nt[max(0, i - 1): i + 2]) for i in range(len(nt))]  # centered 3-turn windows
        win_tok = [set(T.TOKEN_RE.findall(w)) for w in wins]
        alltok = T.TOKEN_RE.findall(" ".join(nt))
        # drug -> numbers that follow it within 12 tokens (dose binding), and drug -> turn ids
        med_nums, med_turns = {}, {}
        for ti, t in enumerate(nt):
            toks = T.TOKEN_RE.findall(t)
            for k, w in enumerate(toks):
                if w in T.MEDICATIONS:
                    med_turns.setdefault(w, []).append(ti)
                    for v in toks[k + 1: k + 13]:
                        if re.fullmatch(r"\d+(?:\.\d+)?", v):
                            med_nums.setdefault(w, set()).add(float(v))
        # polarity of each turn: -1 negated / answered no, +1 answered yes, 0 unknown
        pol = np.zeros(len(nt))
        for i, t in enumerate(nt):
            if NEG_RE.search(t):
                pol[i] = -1
            if turns[i][0] == "doctor" and i + 1 < len(nt) and turns[i + 1][0] == "patient":
                if NO.search(nt[i + 1]):
                    pol[i] = -1
                elif YES.search(nt[i + 1]):
                    pol[i] = 1
        W = self.wv.transform(wins) if wins else None
        C = self.cv.transform(wins) if wins else None
        return dict(turns=turns, nt=nt, wins=wins, win_tok=win_tok, vocab=set(alltok),
                    nums={float(x) for x in alltok if re.fullmatch(r"\d+(?:\.\d+)?", x)},
                    med_nums=med_nums, med_turns=med_turns, pol=pol, W=W, C=C,
                    text=" " + " ".join(nt) + " ")

    def sentences(self, sents, tx):
        texts = [T.norm(s["text"]) for s in sents]
        Ws, Cs = self.wv.transform(texts), self.cv.transform(texts)
        simw = (Ws @ tx["W"].T).toarray() if tx["W"] is not None else np.zeros((len(texts), 1))
        simc = (Cs @ tx["C"].T).toarray() if tx["C"] is not None else np.zeros((len(texts), 1))
        rows = []
        for i, s in enumerate(sents):
            rows.append(self._one(s, texts[i], simw[i], simc[i], tx))
        return np.array(rows, dtype=float)

    def _one(self, s, nt, simw, simc, tx):
        sim = 0.6 * simw + 0.4 * simc
        top = np.argsort(-sim)[:3]
        top_tok = set().union(*[tx["win_tok"][j] for j in top]) if len(top) else set()
        toks = T.TOKEN_RE.findall(nt)
        content = [t for t in toks if t not in T.STOPWORDS and (len(t) > 2 or t.isdigit())]
        nums = [float(t) for t in toks if re.fullmatch(r"\d+(?:\.\d+)?", t)]
        meds = sorted({t for t in toks if t in T.MEDICATIONS})
        lat = {t for t in toks if t in ("left", "right")}
        nc = max(1, len(content))
        f = {}
        f["simw_max"] = simw.max() if simw.size else 0
        f["simc_max"] = simc.max() if simc.size else 0
        srt = np.sort(sim)[::-1]
        f["sim_gap"] = srt[0] - srt[1] if srt.size > 1 else 0
        f["len"] = len(toks)
        f["n_content"] = len(content)
        f["unsup_any"] = sum(t not in tx["vocab"] for t in content) / nc
        f["unsup_top"] = sum(t not in top_tok for t in content) / nc
        f["unsup_any_n"] = sum(t not in tx["vocab"] for t in content)
        f["n_num"] = len(nums)
        f["num_unsup_any"] = sum(v not in tx["nums"] for v in nums)
        f["num_unsup_top"] = sum(T.fmt_num(v) not in top_tok for v in nums)
        f["n_med"] = len(meds)
        f["med_unsup_any"] = sum(m not in tx["vocab"] for m in meds)
        f["med_unsup_top"] = sum(m not in top_tok for m in meds)
        # dose binding: doses written right after a drug in the note vs numbers near that drug in the visit
        bind_v, bind_n = 0, 0
        for m in meds:
            for mm in re.finditer(r"\b" + m + r"\b(.{0,40})", nt):
                for d in DOSE_AFTER.finditer(mm.group(1)):
                    bind_n += 1
                    if float(d.group(1)) not in tx["med_nums"].get(m, set()):
                        bind_v += 1
        f["bind_n"], f["bind_viol"] = bind_n, bind_v
        # any number in the note sentence that is near *some* drug in the visit but not in this window
        f["num_elsewhere"] = sum((v in tx["nums"]) and (T.fmt_num(v) not in top_tok) for v in nums)
        f["n_lat"] = len(lat)
        f["lat_unsup_top"] = sum(l not in top_tok for l in lat)
        f["lat_opp_only"] = sum((l not in top_tok) and ({"left": "right", "right": "left"}[l] in top_tok) for l in lat)
        f["lat_both_vocab"] = float({"left", "right"} <= tx["vocab"])
        # symptoms
        syms = [x for x in SYMPTOMS if x in nt]
        f["sym_unsup"] = sum(x.split()[-1] not in tx["vocab"] for x in syms)
        # polarity
        s_neg = 1.0 if NEG_RE.search(nt) or re.search(r"\bdenies\b|\bnegative for\b", nt) else 0.0
        f["s_neg"] = s_neg
        ev = [tx["pol"][j] for j in top]
        f["ev_neg"] = float(np.mean([e < 0 for e in ev])) if ev else 0
        f["ev_pos"] = float(np.mean([e > 0 for e in ev])) if ev else 0
        f["pol_mismatch"] = s_neg * f["ev_pos"] + (1 - s_neg) * f["ev_neg"]
        f["endorses"] = float(bool(re.search(r"\b(endorses|positive for|reports|abnormal)\b", nt)))
        # time phrases
        tm = re.findall(r"(\d+|a|an|several|a couple of|a few)\s+(days?|weeks?|months?|years?|hours?)", nt)
        f["time_n"] = len(tm)
        f["time_unsup"] = sum((" " + a + " " + b) not in tx["text"] for a, b in tm)
        for sec in SECTIONS:
            f["sec_" + sec] = float(s["section"] == sec)
        if self.reversal:
            rv = 0
            for m in meds:
                ts = tx["med_turns"].get(m, [])
                if ts:
                    last = ts[-1]
                    window = " ".join(tx["nt"][last: last + 2])
                    if REVERSAL_RE.search(window):
                        rv += 1
            f["reversal"] = rv
            f["plan_med"] = float(s["section"] == "ap" and len(meds) > 0)
        global FEATURE_NAMES
        FEATURE_NAMES = list(f)
        return [f[k] for k in f]


def fit_vectorizers(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    wv = TfidfVectorizer(token_pattern=r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", ngram_range=(1, 2),
                         min_df=2, sublinear_tf=True, stop_words=list(T.STOPWORDS))
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200000)
    wv.fit(texts); cv.fit(texts)
    return wv, cv
