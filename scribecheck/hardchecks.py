"""Deterministic hard checks: claims in a note that the visit transcript cannot support.

These are deliberately narrow so that a flag is checkable by anyone in seconds:

  DRUG_UNSAID     a medication (brand or generic) the visit never mentions
  DOSE_UNSAID     a dose number attached to a drug in the note that was never said in the visit
  DOSE_MISBOUND   the dose number was said, but never near that drug (it belongs to something else)
  SIDE_CONFLICT   'left/right <body part>' when the visit only ever names the opposite side for
                  body parts

Transcript normalization handles the ways people say doses out loud: "two point five",
"two . five", "five hundred", and dose-before-drug order ("10 milligrams of lisinopril").
"""
import re

from . import text as T

# brand <-> generic groups; any member counts as a mention of the group
SYN_GROUPS = [
    {"amlodipine", "norvasc"}, {"atorvastatin", "lipitor"}, {"rosuvastatin", "crestor"},
    {"simvastatin", "zocor"}, {"pravastatin", "pravachol"}, {"metoprolol", "toprol", "lopressor"},
    {"lisinopril", "zestril", "prinivil"}, {"losartan", "cozaar"}, {"hydrochlorothiazide", "hctz"},
    {"furosemide", "lasix"}, {"albuterol", "proair", "ventolin", "proventil"},
    {"fluticasone", "flovent", "flonase"}, {"montelukast", "singulair"}, {"cetirizine", "zyrtec"},
    {"loratadine", "claritin"}, {"acetaminophen", "tylenol", "paracetamol"},
    {"ibuprofen", "motrin", "advil", "ibuprofin"}, {"naproxen", "aleve", "naprosyn"}, {"meloxicam", "mobic"},
    {"escitalopram", "lexapro"}, {"citalopram", "celexa"}, {"sertraline", "zoloft"},
    {"fluoxetine", "prozac"}, {"bupropion", "wellbutrin"}, {"duloxetine", "cymbalta"},
    {"metformin", "glucophage"}, {"glipizide", "glucotrol"}, {"insulin", "lantus", "humalog", "novolog"},
    {"levothyroxine", "synthroid"}, {"omeprazole", "prilosec"}, {"pantoprazole", "protonix"},
    {"warfarin", "coumadin"}, {"apixaban", "eliquis"}, {"rivaroxaban", "xarelto"},
    {"clopidogrel", "plavix"}, {"ticagrelor", "brilinta"}, {"aspirin", "asa", "asprin", "asperin"},
    {"nitroglycerin", "nitro"}, {"gabapentin", "neurontin"}, {"prednisone", "deltasone"},
    {"cyclobenzaprine", "flexeril"}, {"tramadol", "ultram"}, {"oxycodone", "percocet"},
    {"amoxicillin", "amoxil"}, {"azithromycin", "zithromax", "z-pack", "zpack"},
    {"ciprofloxacin", "cipro"}, {"doxycycline", "vibramycin"}, {"sumatriptan", "imitrex"},
    {"topiramate", "topamax"}, {"acetazolamide", "diamox"}, {"ondansetron", "zofran"},
    {"dextromethorphan", "robitussin"}, {"guaifenesin", "mucinex"}, {"epinephrine", "epipen"},
    {"ciclesonide", "alvesco"}, {"budesonide", "pulmicort", "symbicort"}, {"carvedilol", "coreg"},
    {"spironolactone", "aldactone"}, {"tamsulosin", "flomax"}, {"finasteride", "proscar"},
    {"sildenafil", "viagra"}, {"alendronate", "fosamax"}, {"allopurinol", "zyloprim"},
    {"colchicine", "colcrys"}, {"methotrexate", "trexall"}, {"hydroxychloroquine", "plaquenil"},
    {"semaglutide", "ozempic", "wegovy"}, {"dulaglutide", "trulicity"}, {"empagliflozin", "jardiance"},
    {"dapagliflozin", "farxiga"}, {"sitagliptin", "januvia"}, {"lorazepam", "ativan"},
    {"alprazolam", "xanax"}, {"zolpidem", "ambien"}, {"trazodone", "desyrel"},
    {"cephalexin", "keflex"}, {"nitrofurantoin", "macrobid"},
    {"cortisone", "corticosteroid", "steroid", "kenalog", "triamcinolone", "depo-medrol", "methylprednisolone"}, {"fexofenadine", "allegra"},
]
DRUGS = set(T.MEDICATIONS) | set().union(*SYN_GROUPS)
CANON = {}
for g in SYN_GROUPS:
    c = sorted(g)[0]
    for d in g:
        CANON[d] = c
for d in DRUGS:
    CANON.setdefault(d, d)

BODY = set("""knee knees ankle ankles foot feet toe toes hip hips leg legs thigh calf shin shoulder shoulders
arm arms elbow elbows wrist wrists hand hands finger fingers thumb eye eyes ear ears breast breasts
lung lungs kidney kidneys flank side sided buttock groin heel achilles hamstring quad
quadriceps rotator cuff forearm temple jaw cheek testicle ovary extremity extremities radial femoral
leg's arm's knee's""".split())
NOT_LIMB = {"sternal", "coronary", "ventricle", "ventricular", "atrial", "atrium", "heart", "bundle",
            "handed", "hand-dominant", "dominant"}
SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-five": 25}
DEC_RE = re.compile(r"\b(" + "|".join(SMALL) + r"|\d+)\s*(?:point|\.)\s*(" + "|".join(SMALL) + r"|\d+)\b")
UNIT = r"(?:mg|milligrams?|mcg|micrograms?|units?|grams?|g|ml)"


def _dec(m):
    a, b = m.group(1), m.group(2)
    a = str(SMALL.get(a, a)); b = str(SMALL.get(b, b))
    return f" {a}.{b} "


def norm_spoken(text):
    t = DEC_RE.sub(_dec, str(text).lower().replace("’", "'"))
    t = re.sub(r"\b(" + "|".join(SMALL) + r"|\d+) and a half\b", lambda m: f"{SMALL.get(m.group(1), m.group(1))}.5", t)
    t = T.norm(t)
    t = re.sub(r"\b(\d+) and a half\b", lambda m: f"{m.group(1)}.5", t)
    t = re.sub(r"(\d+)\s*(?:point|\.)\s*(\d+)\b", r"\1.\2", t)   # two point five / two . five
    t = re.sub(r"(\d+)\s+hundred\b", lambda m: str(int(m.group(1)) * 100), t)
    t = re.sub(r"(\d+)\s+thousand\b", lambda m: str(int(m.group(1)) * 1000), t)
    return t


def _toks(t):
    out = []
    for w in T.TOKEN_RE.findall(t):
        if w not in DRUGS and w.endswith("s") and w[:-1] in DRUGS:   # "two advils", "the ibuprofens"
            w = w[:-1]
        out.append(w)
    return out


NOT_SIDE_NEXT = {"by", "now", "here", "there", "after", "before", "away", "at", "about", "so", "around",
                 "into", "off", "back", "okay", "then", "on", "in", "up", "down", "over", "where"}


def _isnum(w):
    return re.fullmatch(r"\d+(?:\.\d+)?", w) is not None


class Visit:
    def __init__(self, dialogue, window=40):
        turns = T.split_turns(dialogue)
        self.turn_toks = [_toks(norm_spoken(t)) for _, t in turns]
        self.toks = [w for tt in self.turn_toks for w in tt]
        self.vocab = set(self.toks)
        self.drugs = {CANON[w] for w in self.toks if w in DRUGS}
        self.nums = {float(w) for w in self.toks if _isnum(w)}
        # drug -> numbers within +-window tokens in the same or the next turn (dose before or after)
        self.drug_nums = {}
        for k, w in enumerate(self.toks):
            if w in DRUGS:
                near = self.toks[max(0, k - window): k + window + 1]
                self.drug_nums.setdefault(CANON[w], set()).update(float(v) for v in near if _isnum(v))
        # sides named for body parts anywhere ("right knee", "my left one", "on the left")
        self.sides = set()
        for tt in self.turn_toks:
            for k, w in enumerate(tt):
                if w not in ("left", "right"):
                    continue
                nxt, prv = tt[k + 1: k + 6], tt[max(0, k - 2): k]
                if any(x in NOT_LIMB for x in nxt[:2]) or (nxt[:1] and nxt[0] in NOT_SIDE_NEXT and w == "right"):
                    continue
                if any(x in BODY for x in nxt) or prv[-1:] in (["the"], ["my"], ["your"], ["her"], ["his"]):
                    self.sides.add(w)


def check_sentence(sentence, visit):
    """Return a list of (check, detail) violations for one note sentence."""
    t = norm_spoken(sentence)
    toks = _toks(t)
    out = []
    drugs_here = [(k, w) for k, w in enumerate(toks) if w in DRUGS]
    for k, w in drugs_here:
        if CANON[w] not in visit.drugs:
            out.append(("DRUG_UNSAID", w))
            continue
        # dose written right after the drug: "<drug> [words]{0,3} N mg"
        nxt = toks[k + 1: k + 6]
        j = next((i for i, x in enumerate(nxt) if _isnum(x)), None)
        if j is None or any(x in DRUGS for x in nxt[:j]) or not (j + 1 < len(nxt) and re.fullmatch(UNIT, nxt[j + 1])):
            continue
        m = re.match(r"(\d+(?:\.\d+)?)", nxt[j])
        v = float(m.group(1))
        if v not in visit.nums:
            out.append(("DOSE_UNSAID", f"{w} {m.group(1)}"))
        elif v not in visit.drug_nums.get(CANON[w], set()):
            out.append(("DOSE_MISBOUND", f"{w} {m.group(1)}"))
    for k, w in enumerate(toks):
        if w in ("left", "right") and any(x in BODY for x in toks[k + 1: k + 4]) and \
                not any(x in NOT_LIMB for x in toks[k + 1: k + 3]):
            opp = "right" if w == "left" else "left"
            if w not in visit.sides and opp in visit.sides:
                out.append(("SIDE_CONFLICT", " ".join(toks[k: k + 3])))
    return out
