"""Clinical ASR noise simulator.

Applies word-level substitutions / deletions / insertions at a target rate, with clinically
nasty substitutions over-represented: sound-alike drug names, teen/ty number confusions
(fifteen <-> fifty), and laterality drops. Validated against the real paired
virtscribe ASR vs human transcripts in ACI-Bench (see scripts/04_experiments.py, E5).
"""
import random
import re

from .lexicon import SOUND_ALIKE

SA = {}
for a, b in SOUND_ALIKE:
    SA.setdefault(a, []).append(b)
    SA.setdefault(b, []).append(a)

TEEN_TY = {"13": "30", "14": "40", "15": "50", "16": "60", "17": "70", "18": "80", "19": "90",
           "thirteen": "thirty", "fourteen": "forty", "fifteen": "fifty", "sixteen": "sixty",
           "seventeen": "seventy", "eighteen": "eighty", "nineteen": "ninety"}
TEEN_TY.update({v: k for k, v in list(TEEN_TY.items())})
HOMO = {"two": "to", "to": "two", "four": "for", "for": "four", "right": "write", "no": "know",
        "know": "no", "weak": "week", "week": "weak", "ate": "eight", "eight": "ate", "heal": "heel",
        "heel": "heal", "pain": "pane", "bowel": "bowl", "dose": "doze", "patients": "patience",
        "hear": "here", "here": "hear", "their": "there", "there": "their", "one": "won"}
FILLERS = ["uh", "um", "the", "and", "a", "so", "like", "mm-hmm"]


def _sub(word, rng, vocab):
    w = word.lower()
    if w in SA and rng.random() < 0.9:
        return rng.choice(SA[w])
    if w in TEEN_TY and rng.random() < 0.7:
        return TEEN_TY[w]
    if w in HOMO and rng.random() < 0.8:
        return HOMO[w]
    if re.fullmatch(r"\d+", w) and rng.random() < 0.5:
        d = list(w)
        i = rng.randrange(len(d))
        d[i] = str(rng.randrange(10))
        return "".join(d)
    if len(w) > 4 and rng.random() < 0.5:  # truncation / mangling
        i = rng.randrange(1, len(w) - 1)
        return w[:i] + w[i + 1:]
    return rng.choice(vocab) if vocab else w


def add_asr_noise(dialogue: str, rate: float, seed: int = 0, clinical_bias: float = 3.0) -> str:
    """rate ~ target word error rate. clinical_bias multiplies the hit probability for
    drug names, numbers and laterality words (ASR errors concentrate on rare tokens)."""
    rng = random.Random(seed)
    lines = dialogue.split("\n")
    vocab = [w for w in re.findall(r"[a-z]+", dialogue.lower()) if len(w) > 3]
    out_lines = []
    for line in lines:
        m = re.match(r"^(\[[^\]]+\]|[A-Za-z_ ]{2,20}:)\s*(.*)$", line)
        tag, body = (m.group(1), m.group(2)) if m else ("", line)
        words = body.split()
        new = []
        for wd in words:
            lw = wd.lower()
            special = lw in SA or lw in TEEN_TY or re.fullmatch(r"\d+", lw) or lw in ("left", "right")
            p = min(0.95, rate * (clinical_bias if special else 1.0))
            if rng.random() < p:
                op = rng.random()
                if op < 0.55:
                    new.append(_sub(wd, rng, vocab))
                elif op < 0.85:
                    pass  # deletion
                else:
                    new.append(wd)
                    new.append(rng.choice(FILLERS))
            else:
                new.append(wd)
        out_lines.append((tag + " " if tag else "") + " ".join(new))
    return "\n".join(out_lines)
