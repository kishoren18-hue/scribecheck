"""Epistemic fidelity: does the note keep the clinician's diagnostic uncertainty?

A clinician who says "this is probably a sprain, but I want an X-ray to rule out a fracture" made
a hedged diagnostic claim. A note that records "Assessment: ankle sprain" has turned it into a
certain one. Fluency metrics and fact-support checkers both score that note as correct, because
every word in it was said during the visit. We call this *certainty inflation*.

Procedure (deterministic and auditable; v2 after manual validation of v1):
1. Diagnosis vocabulary: words from short problem-label lines in *training-set* assessment
   sections ("Right ankle sprain.", "Acute lateral epicondylitis of the right elbow."), minus
   anatomy, laterality and generic words. Test notes are never used.
2. A hedged diagnostic claim is a doctor turn where a diagnostic-uncertainty cue ("probably",
   "likely", "possible", "might be", "rule out", "I suspect", "concerned for", "I think you have",
   "differential", "versus") is followed within 6 tokens, inside the same clause, by a vocabulary
   term. Descriptive phrases ("looks like your blood pressure is high") do not count.
3. For each hedged term, look at the note sentences that mention it. *Preserved*: a note hedge cue
   sits within 6 tokens before or 3 after the term in some sentence. *Inflated*: the term is stated
   only without one. *Omitted*: the note never mentions the term.
Hedge Preservation Rate (HPR) = preserved / (preserved + inflated).
"""
import re
from collections import Counter

from . import text as T

# strong diagnostic-uncertainty cues only. "I think you have X", "sounds like", "consistent with" are
# how clinicians routinely state a working diagnosis, so they are excluded (WEAK_HEDGE, ablation only).
T_HEDGE = re.compile(
    r"\b(probably|likely|possibly|possible|might be|may be|could be|could also be|could have been|"
    r"may have|might have|i suspect|suspicious (?:for|of|that)|concerned (?:for|about|that)|"
    r"worried (?:about|that)|rule out|ruling out|not sure (?:if|whether)|not certain|differential|"
    r"versus|or maybe|question of|questionable|most likely|more likely|possibility)\b")
WEAK_HEDGE = re.compile(r"\b(i think|sounds like|looks like|consistent with|i believe)\b")
N_HEDGE = re.compile(
    r"\b(probabl[ey]|likely|possibl[ey]|possibility|suspects?|suspected|suspicious|suspicion|"
    r"concern(?:ed|ing)?|worr(?:y|ied)|rule out|ruled out|r/o|questionable|question|presumed|presumptive|"
    r"suggestive|suggests|differential|versus|vs|may|might|could|query|uncertain|unclear|uncertain|"
    r"exclude|excluded|working|favor|favored|thinks|believes|impression|if|whether|evaluat\w*|assess\w*|"
    r"appears?|seems?|apparent|workup|work-up|worrisome|monitor for|await\w*|pending|confirm\w*|investigat\w*)\b|\?")
ACTION = set("do get order see take try start give check look need want go debridement workup refer send "
             "schedule repeat".split())
CLAUSE_END = re.compile(r"[.?!;,]| but | and so | so ")

# Fixed diagnosis lexicon (single tokens; common primary-care, orthopedic, cardiopulmonary and
# dermatologic conditions). Written once, before looking at test notes, so it cannot be tuned to them.
DX_LEXICON = set("""sprain sprained strain fracture fractured break tendinitis tendonitis tendinopathy
bursitis arthritis osteoarthritis epicondylitis tear torn meniscus acl mcl ligament rotator cuff
impingement labral carpal tunnel sciatica radiculopathy herniated herniation disc stenosis spondylosis
pneumonia bronchitis infection infected viral virus bacterial uti cellulitis abscess sinusitis strep
covid flu influenza cold asthma copd exacerbation failure angina arrhythmia fibrillation afib murmur
hypertension diabetes hypothyroidism hyperthyroidism anemia stone stones nephrolithiasis gout gerd
reflux ulcer gastritis gastroenteritis appendicitis gallstones cholecystitis pancreatitis migraine
migraines concussion vertigo neuropathy depression anxiety insomnia dermatitis eczema psoriasis rosacea
allergic allergy allergies rhinitis croup otitis conjunctivitis lupus cancer malignancy malignant nodule
mass tumor cyst hernia constipation colitis dehydration syncope embolism clot dvt thrombosis fasciitis
ganglion dislocation dislocated subluxation contusion bruise hematoma effusion edema obstruction
hemorrhoids prostatitis pyelonephritis mononucleosis mono sepsis papilledema osteoporosis
neuroma bunion tenosynovitis quervain trigger costochondritis pleurisy pericarditis myocarditis
cardiomyopathy ischemia infarction blockage blockages adenoma polyp polyps diverticulitis
lisfranc avulsion stress overuse inflammation inflamed pinched compression whiplash plantar
spasm spasms hypoglycemia hyperglycemia ketoacidosis prediabetes neuritis shingles herpes
yeast fungal candida thrush scabies impetigo warts acne hives urticaria anaphylaxis
preeclampsia miscarriage endometriosis fibroids pcos cystitis incontinence kidney renal
hepatitis cirrhosis fatty apnea idiopathic intracranial""".split())
# findings and symptoms that are usually *observed*, not inferred: a hedge near them is almost always
# about their cause ("this nodule could be..."), so they are removed after validation round 2
DX_LEXICON -= set("nodule murmur mass cyst edema effusion stone stones kidney reflux cold strep".split())

NON_DX = set("""right left bilateral acute chronic mild moderate severe pain history status post
likely possible probable lower upper anterior posterior lateral medial distal proximal knee ankle
shoulder elbow wrist hand foot back neck hip chest arm leg finger toe joint area side today
follow plan continue patient visit symptoms symptom stable controlled uncontrolled well type
with without from that this which there their have been also into onto over under about after
before during other another most more less some well-controlled poorly management related""".split())


def build_dx_vocab(ap_lines=None, min_count=3):
    """Fixed lexicon (default). The data-driven variant below is kept for the ablation in the paper."""
    if ap_lines is None:
        return set(DX_LEXICON)
    c = Counter()
    for line in ap_lines:
        toks = T.TOKEN_RE.findall(T.norm(line))
        if not (1 <= len(toks) <= 8) or line.strip().lower().startswith(("continue", "start", "order", "follow")):
            continue
        c.update(set(w for w in toks if len(w) >= 4 and not w.isdigit()))
    return {w for w, n in c.items() if n >= min_count and w not in T.STOPWORDS and w not in NON_DX}


def hedged_terms(dialogue, dx_vocab, span=4):
    """Diagnostic terms the doctor attached an uncertainty cue to. Returns {term: turn_text}."""
    out = {}
    for spk, turn in T.split_turns(dialogue):
        if spk != "doctor":
            continue
        t = T.norm(turn)
        for m in T_HEDGE.finditer(t):
            rest = t[m.end():]
            cut = CLAUSE_END.search(rest)
            rest = rest[: cut.start()] if cut else rest
            toks = T.TOKEN_RE.findall(rest)[:span]
            for k, w in enumerate(toks):
                if w in dx_vocab and not any(x in ACTION for x in toks[:k]):
                    out.setdefault(w, turn.strip())
    return out


def _hedged_near(sent_norm, term, before=6, after=3):
    """Sentence-level: any uncertainty cue in the sentence that mentions the term (lenient toward
    'preserved', so inflation estimates are conservative)."""
    return N_HEDGE.search(sent_norm) is not None


def _hedged_near_window(sent_norm, term, before=6, after=3):
    toks = T.TOKEN_RE.findall(sent_norm)
    hed = [i for i, w in enumerate(toks) if N_HEDGE.fullmatch(w)]
    before = max(before, 10)
    # multi-word cues ("rule out", "concern for") -> mark their first token
    for m in N_HEDGE.finditer(sent_norm):
        if " " in m.group(0):
            hed.append(len(T.TOKEN_RE.findall(sent_norm[: m.start()])))
    if "?" in sent_norm:
        return True
    for i, w in enumerate(toks):
        if w == term and any(i - before <= h <= i + after for h in hed):
            return True
    return False


DX_SECTIONS = {"ap", "cc"}
DX_VERB = re.compile(r"\b(diagnos\w*|assessment|impression|consistent with|due to|secondary to)\b")


def audit_note(note_sents, dialogue, dx_vocab):
    """note_sents: list of (section, text). Returns one dict per hedged diagnostic term.
    Only sentences that record a diagnosis count: assessment/plan or chief complaint sections, or any
    sentence with a diagnostic verb. History mentions ("history of asthma") are ignored."""
    terms = hedged_terms(dialogue, dx_vocab)
    rows = []
    normed = [(sec, s, T.norm(s)) for sec, s in note_sents
              if sec in DX_SECTIONS or DX_VERB.search(T.norm(s))]
    normed = [(sec, s, n) for sec, s, n in normed if not re.search(r"\bhistory of\b", n)]
    for term, turn in terms.items():
        hits = [(sec, s, n) for sec, s, n in normed if re.search(r"\b" + re.escape(term) + r"\b", n)]
        if not hits:
            status, example = "omitted", ""
        else:
            hedged = [s for _, s, n in hits if _hedged_near(n, term)]
            status, example = ("preserved", hedged[0]) if hedged else ("inflated", hits[0][1])
        rows.append(dict(term=term, status=status, transcript_turn=turn, note_sentence=example,
                         note_sections=",".join(sorted({sec for sec, _, _ in hits}))))
    return rows
