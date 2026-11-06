"""Text utilities: note sentence splitting, transcript turns, number normalisation, fact atoms."""
import re
from dataclasses import dataclass, field

from .lexicon import MEDICATIONS, NEGATION_CUES, STOPWORDS

SECTION_RE = re.compile(r"^[A-Z][A-Z &/\-]{3,}$")
SECTION_CANON = {
    "CHIEF COMPLAINT": "cc", "HISTORY OF PRESENT ILLNESS": "hpi", "HPI": "hpi",
    "REVIEW OF SYSTEMS": "ros", "PHYSICAL EXAMINATION": "exam", "PHYSICAL EXAM": "exam",
    "EXAM": "exam", "RESULTS": "results", "ASSESSMENT AND PLAN": "ap", "ASSESSMENT": "ap",
    "PLAN": "ap", "PAST MEDICAL HISTORY": "pmh", "MEDICATIONS": "meds", "CURRENT MEDICATIONS": "meds",
    "ALLERGIES": "allergies", "SOCIAL HISTORY": "social", "FAMILY HISTORY": "family",
    "PAST SURGICAL HISTORY": "psh", "INSTRUCTIONS": "ap", "VITALS": "exam", "VITALS REVIEWED": "exam",
    "PROCEDURE": "procedure", "IMPRESSION": "ap",
}
# MTS-Dialog section codes -> canonical section
MTS_SECTION = {
    "GENHX": "hpi", "CC": "cc", "ROS": "ros", "EXAM": "exam", "PASTMEDICALHX": "pmh",
    "MEDICATIONS": "meds", "ALLERGY": "allergies", "FAM/SOCHX": "family", "PASTSURGICAL": "psh",
    "ASSESSMENT": "ap", "PLAN": "ap", "DIAGNOSIS": "ap", "DISPOSITION": "ap", "PROCEDURES": "procedure",
    "IMAGING": "results", "LABS": "results", "IMMUNIZATIONS": "pmh", "GYNHX": "pmh", "EDCOURSE": "hpi",
    "OTHER_HISTORY": "pmh",
}

UNITS = r"(mg|milligrams?|mcg|micrograms?|g|grams?|ml|milliliters?|units?|%|percent|mmhg|bpm|lbs?|pounds?|kg|cm|mm|inches|degrees?)"

ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}
SCALE = {"hundred": 100, "thousand": 1000}
NUMWORDS = set(ONES) | set(TENS) | set(SCALE)


def words_to_digits(text: str) -> str:
    """Rewrite spelled-out numbers ('twenty five', 'one thousand') as digits.
    Leaves 'one' alone when it is clearly a pronoun-like use is impossible to know, so we only
    convert runs that contain at least one tens/scale word OR are followed by a unit/time word."""
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        w = toks[i].lower().strip(",.")
        if w in NUMWORDS:
            j, total, cur, has_strong = i, 0, 0, False
            while j < len(toks):
                t = toks[j].lower().strip(",.")
                t = t.replace("-", " ")
                parts = t.split()
                if not parts or not all(p in NUMWORDS or p == "and" for p in parts):
                    break
                for p in parts:
                    if p in ONES:
                        cur += ONES[p]
                    elif p in TENS:
                        cur += TENS[p]; has_strong = True
                    elif p in SCALE:
                        cur = max(cur, 1) * SCALE[p]; has_strong = True
                        if p == "thousand":
                            total += cur; cur = 0
                j += 1
            val = total + cur
            nxt = toks[j].lower().strip(",.") if j < len(toks) else ""
            unitish = re.fullmatch(UNITS, nxt) is not None or nxt.rstrip("s") in {"day", "week", "month", "year", "hour", "time", "minute"}
            if has_strong or unitish or (j - i) > 1:
                out.append(str(val)); i = j; continue
        out.append(toks[i]); i += 1
    return " ".join(out)


def norm(text: str) -> str:
    t = text.lower()
    t = t.replace("’", "'")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)  # 1,000 -> 1000
    t = re.sub(r"\s+", " ", t)
    t = words_to_digits(t)
    return t


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")


def tokens(text: str):
    return TOKEN_RE.findall(norm(text))


def content_tokens(text: str):
    return [t for t in tokens(text) if t not in STOPWORDS and (len(t) > 2 or t.isdigit())]


def numbers(text: str):
    """All numeric values in normalised text (fractions like 120/80 split into both parts)."""
    t = norm(text)
    vals = []
    for m in re.finditer(r"\d+(?:\.\d+)?", t):
        try:
            v = float(m.group())
        except ValueError:
            continue
        vals.append(v)
    return vals


def meds(text: str):
    toks = set(tokens(text))
    return sorted(m for m in MEDICATIONS if m in toks)


def laterality(text: str):
    toks = set(tokens(text))
    return {s for s in ("left", "right", "bilateral") if s in toks}


def is_negated_sentence(text: str) -> bool:
    t = " " + norm(text) + " "
    return any(re.search(r"(?<![a-z])" + re.escape(c) + r"(?![a-z])", t) for c in NEGATION_CUES)


@dataclass
class Sentence:
    idx: int
    text: str
    section: str
    is_header: bool = False


def split_note(note: str):
    """Split an ACI-Bench style note into sentences with their section label.
    Headers are kept as non-scorable lines so the note can be reassembled."""
    out, section, idx = [], "other", 0
    for raw_line in note.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if SECTION_RE.match(line) or line.rstrip(":").upper() in SECTION_CANON and len(line) < 40:
            section = SECTION_CANON.get(line.rstrip(":").upper(), section)
            out.append(Sentence(idx, line, section, True)); idx += 1
            continue
        line = re.sub(r"^[•\-\*]\s*", "", line)
        # split on sentence boundaries but not on decimals or abbreviations like 'Dr.'
        parts = re.split(r"(?<=[a-z0-9\)%][.!?])\s+(?=[A-Z])", line)
        for p in parts:
            p = p.strip()
            if len(p) < 3:
                continue
            short_label = (len(p.split()) <= 3 and p.endswith(":"))
            out.append(Sentence(idx, p, section, short_label)); idx += 1
    return out


def split_text_section(text: str, section: str):
    parts = re.split(r"(?<=[a-z0-9\)%][.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text.strip()))
    parts = [p.strip() for p in parts if len(p.strip()) >= 3]
    return [Sentence(i, p, section) for i, p in enumerate(parts)]


TURN_RE = re.compile(r"^\s*\[?(doctor|patient|patient_guest|guest_family|guest_clinician|nurse|[a-z_]+)\]?\s*:?\s*(.*)$", re.I)


def split_turns(dialogue: str):
    """Return list of (speaker, text). Handles ACI '[doctor] ...' and MTS 'Doctor: ...'."""
    turns = []
    for line in re.split(r"\r?\n", dialogue):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[([a-z_]+)\]\s*(.*)$", line, re.I) or re.match(r"^([A-Za-z_ ]{2,20}):\s*(.*)$", line)
        if m:
            spk = m.group(1).lower().strip()
            spk = "doctor" if spk.startswith(("doctor", "guest_clinician", "nurse")) else "patient"
            turns.append([spk, m.group(2)])
        elif turns:
            turns[-1][1] += " " + line
        else:
            turns.append(["patient", line])
    return [(s, t) for s, t in turns]


def join_turns(turns):
    return "\n".join(f"[{s}] {t}" for s, t in turns)


def fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


from .lexicon import STOPWORDS as STOPWORDS  # noqa: E402,F401  re-export for features
MEDICATIONS = MEDICATIONS  # re-export
