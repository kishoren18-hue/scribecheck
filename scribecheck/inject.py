"""ScribeErr: transcript-aware error injection for clinical notes.

Every injected error is checked against the encounter transcript so we know whether its content
is *ungrounded* (some fact is absent from the transcript) or *grounded* (every content word of
the wrong sentence still appears in the transcript, so lexical support checks cannot see it).

Error types
-----------
U_DOSE      dose changed to a plausible value never said in the visit
U_NUM       other number (vital, lab, duration, age) changed to an unsaid value
U_MED       medication swapped for a same-class drug never mentioned
U_LAT       left/right flipped when the visit only ever mentions one side
U_TIME      time unit shifted (days -> weeks) to an unsaid phrase
U_FAB       fabricated symptom sentence inserted
NEG_FLIP    polarity flipped (denies <-> endorses, no X -> X); words grounded, meaning reversed
G_NUM       number swapped for another number said elsewhere in the visit (cross-binding)
G_MED       medication swapped for another drug discussed in the visit (cross-binding)
G_LAT       left/right flipped when the visit mentions both sides
RETRACT     transcript edited: clinician retracts a plan the note still records as delivered
"""
import random
import re

from . import text as T
from .lexicon import (FREQUENCIES, LATERALITY, MEDICATIONS, RETRACTION_TEMPLATES_A,
                      RETRACTION_TEMPLATES_B, SYMPTOMS, TIME_SWAP)

DOSES = [1, 2, 2.5, 4, 5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200, 250,
         300, 325, 400, 500, 600, 750, 800, 850, 1000, 2000]
SEVERITY_WEIGHT = {"major": 1.0, "moderate": 0.5, "minor": 0.25}
GROUNDED_TYPES = {"G_NUM", "G_MED", "G_LAT", "RETRACT"}
ALL_TYPES = ["U_DOSE", "U_NUM", "U_MED", "U_LAT", "U_TIME", "U_FAB", "NEG_FLIP",
             "G_NUM", "G_MED", "G_LAT", "RETRACT"]

MIXES = {
    # roughly uniform over types (calibration mix)
    "uniform": {t: 1.0 for t in ALL_TYPES},
    # "subtle vendor": a mature scribe that rarely invents unsaid facts but mis-binds and misses retractions
    "subtle": {"U_DOSE": 0.3, "U_NUM": 0.3, "U_MED": 0.2, "U_LAT": 0.2, "U_TIME": 0.3, "U_FAB": 0.3,
               "NEG_FLIP": 1.5, "G_NUM": 2.0, "G_MED": 2.0, "G_LAT": 1.0, "RETRACT": 2.0},
    # "crude vendor": an early scribe that mostly invents things
    "crude": {"U_DOSE": 2.0, "U_NUM": 2.0, "U_MED": 2.0, "U_LAT": 1.0, "U_TIME": 1.5, "U_FAB": 2.5,
              "NEG_FLIP": 0.5, "G_NUM": 0.2, "G_MED": 0.2, "G_LAT": 0.2, "RETRACT": 0.2},
}

HIGH_RISK_SECTIONS = {"ap", "meds", "allergies", "results", "procedure"}
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*\d)")


def fmt(v):
    return str(int(v)) if float(v).is_integer() else str(v)


class TranscriptIndex:
    def __init__(self, dialogue: str):
        self.raw = dialogue
        self.turns = T.split_turns(dialogue)
        self.normtext = " " + T.norm(dialogue) + " "
        self.vocab = set(T.tokens(dialogue))
        self.nums = set(T.numbers(dialogue))
        self.meds = set(T.meds(dialogue))
        # numbers that appear right before a dose unit in the transcript
        self.dose_nums = set()
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(mg|milligrams?|mcg|micrograms?|units?)", self.normtext):
            self.dose_nums.add(float(m.group(1)))

    def has_phrase(self, phrase: str) -> bool:
        return (" " + T.norm(phrase) + " ") in self.normtext or T.norm(phrase) in self.normtext


def _severity(etype, sent):
    sec = sent.section
    med_or_dose = bool(T.meds(sent.text)) or bool(re.search(r"\d\s*(mg|mcg|units)", sent.text.lower()))
    if etype in {"U_DOSE", "U_MED", "G_MED", "RETRACT"}:
        return "major"
    if etype in {"G_NUM", "U_NUM"}:
        return "major" if (med_or_dose or sec in HIGH_RISK_SECTIONS) else "moderate"
    if etype in {"U_LAT", "G_LAT"}:
        return "major" if sec in {"ap", "exam", "results", "procedure", "hpi"} else "moderate"
    if etype == "NEG_FLIP":
        return "major" if sec in {"ap", "hpi", "allergies", "results"} else "moderate"
    if etype == "U_TIME":
        return "moderate"
    if etype == "U_FAB":
        return "minor" if sec == "ros" else "moderate"
    return "moderate"


# ------------------------------------------------------------------ individual mutators
def _numbers_in(s):
    return [(m.start(1), m.end(1), float(m.group(1))) for m in NUM_RE.finditer(s)]


def mut_u_dose(s, tx, rng):
    hits = [(a, b, v) for a, b, v in _numbers_in(s)
            if re.match(r"\s*(mg|mcg|milligram|units|grams?\b|g\b)", s[b:b + 12].lower())]
    if not hits:
        return None
    a, b, v = rng.choice(hits)
    cands = [d for d in DOSES if d != v and d not in tx.nums and 0.2 <= d / max(v, 0.1) <= 5]
    if not cands:
        return None
    d = min(cands, key=lambda x: abs(x - v) + rng.random() * v)
    return s[:a] + fmt(d) + s[b:]


def mut_u_num(s, tx, rng):
    hits = [(a, b, v) for a, b, v in _numbers_in(s)
            if not re.match(r"\s*(mg|mcg|milligram|units)", s[b:b + 12].lower()) and 0 < v < 400]
    if not hits:
        return None
    a, b, v = rng.choice(hits)
    for _ in range(20):
        if v < 10:
            nv = v + rng.choice([-3, -2, -1, 1, 2, 3, 4])
        else:
            nv = round(v * rng.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5]))
        if nv > 0 and nv != v and float(nv) not in tx.nums:
            return s[:a] + fmt(nv) + s[b:]
    return None


def mut_u_med(s, tx, rng):
    ms = T.meds(s)
    if not ms:
        return None
    m = rng.choice(ms)
    cls = MEDICATIONS[m]
    cands = [x for x, c in MEDICATIONS.items() if c == cls and x != m and x not in tx.vocab]
    if not cands:
        cands = [x for x in MEDICATIONS if x not in tx.vocab and x != m]
    new = rng.choice(cands)
    return re.sub(r"(?i)\b" + m + r"\b", lambda mm: new.capitalize() if mm.group()[0].isupper() else new, s, count=1)


def _flip_lat(s):
    def rep(mm):
        w = mm.group()
        new = LATERALITY[w.lower()]
        return new.capitalize() if w[0].isupper() else new
    return re.sub(r"(?i)\b(left|right)\b", rep, s, count=1)


def mut_u_lat(s, tx, rng):
    lat = T.laterality(s) & {"left", "right"}
    if len(lat) != 1:
        return None
    side = next(iter(lat))
    if LATERALITY[side] in tx.vocab:
        return None
    return _flip_lat(s)


def mut_g_lat(s, tx, rng):
    lat = T.laterality(s) & {"left", "right"}
    if len(lat) != 1 or not {"left", "right"} <= tx.vocab:
        return None
    return _flip_lat(s)


def mut_u_time(s, tx, rng):
    m = re.search(r"(?i)\b(\d+|a|an|one|two|three|four|five|six|several|a couple of|a few)\s+(day|days|week|weeks|month|months|year|years|hour|hours)\b", s)
    if not m:
        return None
    unit = m.group(2).lower()
    new_phrase = m.group(1) + " " + TIME_SWAP[unit]
    if tx.has_phrase(new_phrase):
        return None
    return s[:m.start()] + new_phrase + s[m.end():]


def mut_neg_flip(s, tx, rng):
    rules = [
        (r"\bDenies\b", "Endorses"), (r"\bdenies\b", "endorses"), (r"\bEndorses\b", "Denies"),
        (r"\bendorses\b", "denies"), (r"\bnegative for\b", "positive for"),
        (r"\bpositive for\b", "negative for"), (r"\bNegative for\b", "Positive for"),
        (r"\bPositive for\b", "Negative for"), (r"\bno evidence of\b", "evidence of"),
        (r"\bhas not\b", "has"), (r"\bis not\b", "is"), (r"\bdid not\b", "did"),
        (r"\bdoes not\b", "does"), (r"\bwithout\b", "with"), (r"\bNo\b ", ""),
        (r"\bnot\b ", ""), (r"\bunremarkable\b", "abnormal"), (r"\bnormal\b", "abnormal"),
        (r"\breports\b", "denies"), (r"\bhas been\b", "has not been"),
        (r"\bis\b (?=tender|swollen|positive|elevated|present)", "is not "),
    ]
    order = rules[:]
    rng.shuffle(order)
    # keep the strong lexical flips first so the mutation reads naturally
    order.sort(key=lambda r: 0 if r[0] in (r"\bDenies\b", r"\bdenies\b", r"\bEndorses\b", r"\bendorses\b",
                                           r"\bnegative for\b", r"\bpositive for\b") else 1)
    for pat, rep in order:
        if re.search(pat, s):
            out = re.sub(pat, rep, s, count=1)
            out = out[0].upper() + out[1:] if out else out
            if out != s:
                return out
    return None


def mut_g_num(s, tx, rng):
    hits = _numbers_in(s)
    if not hits:
        return None
    a, b, v = rng.choice(hits)
    is_dose = bool(re.match(r"\s*(mg|mcg|milligram|units)", s[b:b + 12].lower()))
    pool = (tx.dose_nums if is_dose else tx.nums) - set(v2 for _, _, v2 in hits)
    pool = [p for p in pool if p != v and p > 0 and (0.1 <= p / max(v, 0.1) <= 10)]
    if not pool:
        return None
    return s[:a] + fmt(rng.choice(sorted(pool))) + s[b:]


def mut_g_med(s, tx, rng):
    ms = T.meds(s)
    if not ms:
        return None
    m = rng.choice(ms)
    pool = sorted(tx.meds - set(ms))
    if not pool:
        return None
    new = rng.choice(pool)
    return re.sub(r"(?i)\b" + m + r"\b", lambda mm: new.capitalize() if mm.group()[0].isupper() else new, s, count=1)


MUTATORS = {"U_DOSE": mut_u_dose, "U_NUM": mut_u_num, "U_MED": mut_u_med, "U_LAT": mut_u_lat,
            "U_TIME": mut_u_time, "NEG_FLIP": mut_neg_flip, "G_NUM": mut_g_num, "G_MED": mut_g_med,
            "G_LAT": mut_g_lat}


def fabricate(sent, tx, rng):
    cands = [x for x in SYMPTOMS if x.split()[-1] not in tx.vocab]
    if not cands:
        return None
    sym = rng.choice(cands)
    if sent.section == "ros":
        return rng.choice([f"Endorses {sym}.", f"Positive for {sym}."])
    return rng.choice([f"The patient also reports intermittent {sym}.",
                       f"He also notes some {sym} over the past week.",
                       f"She additionally describes episodes of {sym}."])


def insert_retraction(turns, med, rng, template_set):
    """Insert a clinician retraction after the last turn that mentions `med`."""
    idx = [i for i, (_, t) in enumerate(turns) if re.search(r"\b" + med + r"\b", T.norm(t))]
    if not idx:
        return None
    i = idx[-1]
    # put it after the patient's response, if there is one
    j = i + 2 if i + 1 < len(turns) and turns[i + 1][0] == "patient" else i + 1
    tmpl = rng.choice(template_set).format(med=med)
    spk, txt = re.match(r"\[(\w+)\]\s*(.*)", tmpl).groups()
    new = turns[:j] + [(spk, txt), ("patient", "okay .")] + turns[j:]
    return new


# ------------------------------------------------------------------ note-level driver
def scorable(sent):
    return (not sent.is_header) and len(T.tokens(sent.text)) >= 3


def corrupt_note(sentences, dialogue, rng, mix="uniform", n_errors=None, retraction_set="A",
                 p_clean=0.2):
    """Return (records, dialogue_out). records: list of dicts per sentence with labels."""
    tx = TranscriptIndex(dialogue)
    weights = MIXES[mix] if isinstance(mix, str) else mix
    recs = [dict(idx=s.idx, text=s.text, section=s.section, is_header=s.is_header, orig=s.text,
                 label=0, etype=None, severity=None, grounded=None) for s in sentences]
    sents = {s.idx: s for s in sentences}
    if n_errors is None:
        n_errors = 0 if rng.random() < p_clean else int(rng.choice([1, 1, 2, 2, 3, 4]))
    used, turns, inserted = set(), T.split_turns(dialogue), []
    types, w = list(weights), [weights[t] for t in weights]
    attempts = 0
    while len(used) + len(inserted) < n_errors and attempts < 60:
        attempts += 1
        et = rng.choices(types, weights=w)[0]
        cands = [s for s in sentences if scorable(s) and s.idx not in used]
        rng.shuffle(cands)
        done = False
        if et == "U_FAB":
            hosts = [s for s in cands if s.section in ("ros", "hpi")]
            if hosts:
                h = hosts[0]
                new_text = fabricate(h, tx, rng)
                if new_text:
                    inserted.append((h.idx, dict(idx=None, text=new_text, section=h.section,
                                                 is_header=False, orig=None, label=1, etype="U_FAB",
                                                 severity=_severity("U_FAB", h), grounded=False)))
                    done = True
        elif et == "RETRACT":
            if any(r["etype"] == "RETRACT" for r in recs):
                continue
            ap = [s for s in cands if s.section == "ap" and T.meds(s.text)]
            for s in ap:
                med = [m for m in T.meds(s.text) if m in tx.meds]
                if not med:
                    continue
                med = rng.choice(med)
                if re.search(r"(?i)\b(stop|discontinu|hold|avoid|no longer|not)\b", s.text):
                    continue  # retracting a 'stop' would not make the sentence wrong
                tset = RETRACTION_TEMPLATES_A if retraction_set == "A" else RETRACTION_TEMPLATES_B
                new_turns = insert_retraction(turns, med, rng, tset)
                if new_turns is None:
                    continue
                turns = new_turns
                for s2 in sentences:  # every A&P sentence prescribing that med is now wrong
                    if s2.section == "ap" and scorable(s2) and med in T.meds(s2.text) and s2.idx not in used:
                        r = recs[s2.idx]
                        r.update(label=1, etype="RETRACT", severity="major", grounded=True)
                        used.add(s2.idx)
                done = True
                break
        else:
            for s in cands[:40]:
                out = MUTATORS[et](s.text, tx, rng)
                if out and out != s.text:
                    r = recs[s.idx]
                    r.update(text=out, label=1, etype=et, severity=_severity(et, s),
                             grounded=et in GROUNDED_TYPES or et == "NEG_FLIP")
                    used.add(s.idx)
                    done = True
                    break
        if not done:
            continue
    # splice inserted sentences after their hosts
    final = []
    ins_map = {}
    for host, rec in inserted:
        ins_map.setdefault(host, []).append(rec)
    for r in recs:
        final.append(r)
        final.extend(ins_map.get(r["idx"], []))
    for k, r in enumerate(final):
        r["pos"] = k
        r["scorable"] = (not r["is_header"]) and len(T.tokens(r["text"])) >= 3
        r["sev_w"] = SEVERITY_WEIGHT[r["severity"]] if r["label"] else 0.0
    return final, T.join_turns(turns)
