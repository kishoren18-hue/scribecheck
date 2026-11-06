"""Term-anchored polarity evidence: was each finding in the note affirmed or denied in the visit?

The window-level polarity feature in features.py asks whether the best-matching 3-turn window
is negated. Negation errors defeat it: "denies chest pain" and "endorses chest pain" retrieve the
same window. Here every clinical term in the note sentence is traced to every turn that mentions it,
and each mention is scored:
  -1  a negation cue within 5 tokens before the term in that turn, or the term sits in a doctor
      question that the patient answers "no"
  +1  the term sits in a doctor question the patient answers "yes", or a patient statement
      without negation
   0  otherwise (e.g. the doctor mentions it in passing)
The note's own polarity for the term is -1 if a negation cue precedes it within 5 tokens.
Deliberately no grammar cues ("endorses any ...") so the detector cannot learn injection artifacts.
"""
import re

import numpy as np

from . import text as T

NEG_CUES = {"no", "not", "denies", "deny", "denied", "without", "never", "negative", "none", "nor",
            "don't", "doesn't", "didn't", "haven't", "hasn't", "isn't", "wasn't", "aren't", "can't",
            "unremarkable"}
YES = re.compile(r"^\s*(yes|yeah|yep|yup|mm-hmm|uh-huh|i do|i have|i did|sometimes|correct|right|"
                 r"a little|definitely|kind of|a bit)\b")
NO = re.compile(r"^\s*(no|nope|nah|not really|none|never|i don't|i haven't|i didn't|not at all)\b")
QUESTION = re.compile(r"\?|^\s*(do|did|have|has|are|is|any|how about|what about|does|were|was)\b")
SKIP = set("""patient doctor reports report states state today plan visit history continue also time
noted note well good able feel feels feeling some have been with that this which there their about
would could should will just like""".split())

NAMES = ["pol_terms", "pol_terms_ev", "pol_agree", "pol_disagree", "pol_frac_disagree", "pol_note_neg",
         "pol_ev_neg", "pol_ev_pos", "pol_strong_disagree"]


def _terms(toks):
    return [w for w in toks if len(w) >= 4 and w not in T.STOPWORDS and w not in SKIP and not w.isdigit()
            and w not in NEG_CUES]


class VisitPolarity:
    def __init__(self, dialogue):
        turns = T.split_turns(dialogue)
        self.toks = [T.TOKEN_RE.findall(T.norm(t)) for _, t in turns]
        self.spk = [s for s, _ in turns]
        self.raw = [T.norm(t) for _, t in turns]
        self.ev = {}  # term -> list of evidence values
        for i, tt in enumerate(self.toks):
            q = self.spk[i] == "doctor" and QUESTION.search(self.raw[i]) is not None
            ans = 0
            if q and i + 1 < len(self.toks) and self.spk[i + 1] != "doctor":
                ans = -1 if NO.search(self.raw[i + 1]) else (1 if YES.search(self.raw[i + 1]) else 0)
            for k, w in enumerate(tt):
                if len(w) < 4:
                    continue
                neg = any(x in NEG_CUES for x in tt[max(0, k - 5): k])
                if q:
                    v = ans if not neg else (-ans if ans else 0)
                elif neg:
                    v = -1
                elif self.spk[i] != "doctor":
                    v = 1
                else:
                    v = 0
                self.ev.setdefault(w, []).append(v)


def sentence_features(text, vp):
    toks = T.TOKEN_RE.findall(T.norm(text))
    terms = _terms(toks)
    agree = dis = strong = ev_neg = ev_pos = n_ev = 0
    note_neg_any = 0
    for k, w in enumerate(toks):
        if w not in terms:
            continue
        nneg = any(x in NEG_CUES for x in toks[max(0, k - 5): k])
        note_neg_any |= nneg
        ev = [v for v in vp.ev.get(w, []) if v != 0]
        if not ev:
            continue
        n_ev += 1
        m = float(np.mean(ev))
        ev_neg += m < 0
        ev_pos += m > 0
        note_v = -1 if nneg else 1
        if m * note_v > 0:
            agree += 1
        elif m * note_v < 0:
            dis += 1
            strong += abs(m) == 1.0
    return [len(set(terms)), n_ev, agree, dis, dis / max(n_ev, 1), float(note_neg_any), ev_neg, ev_pos, strong]
