"""Small clinical lexicons used for fact extraction, error injection and ASR simulation.

These are hand-curated, deliberately compact, and NOT a substitute for RxNorm/UMLS.
Swapping in RxNorm is a listed upgrade in the README.
"""

# medication -> pharmacologic class (used for plausible same-class substitutions)
MEDICATIONS = {
    # cardiovascular
    "lisinopril": "acei", "enalapril": "acei", "ramipril": "acei", "benazepril": "acei",
    "losartan": "arb", "valsartan": "arb", "irbesartan": "arb", "olmesartan": "arb",
    "amlodipine": "ccb", "nifedipine": "ccb", "diltiazem": "ccb", "verapamil": "ccb",
    "metoprolol": "bb", "carvedilol": "bb", "atenolol": "bb", "propranolol": "bb", "bisoprolol": "bb",
    "hydrochlorothiazide": "diuretic", "chlorthalidone": "diuretic", "furosemide": "diuretic",
    "lasix": "diuretic", "spironolactone": "diuretic", "bumetanide": "diuretic", "torsemide": "diuretic",
    "atorvastatin": "statin", "lipitor": "statin", "rosuvastatin": "statin", "crestor": "statin",
    "simvastatin": "statin", "pravastatin": "statin",
    "aspirin": "antiplatelet", "clopidogrel": "antiplatelet", "plavix": "antiplatelet",
    "warfarin": "anticoag", "coumadin": "anticoag", "apixaban": "anticoag", "eliquis": "anticoag",
    "rivaroxaban": "anticoag", "xarelto": "anticoag", "heparin": "anticoag",
    "nitroglycerin": "nitrate", "isosorbide": "nitrate", "hydralazine": "vasodilator", "clonidine": "alpha",
    "digoxin": "cardiac", "amiodarone": "antiarrhythmic", "entresto": "arni",
    # endocrine
    "metformin": "biguanide", "glipizide": "sulfonylurea", "glyburide": "sulfonylurea",
    "glimepiride": "sulfonylurea", "insulin": "insulin", "lantus": "insulin", "novolog": "insulin",
    "humalog": "insulin", "januvia": "dpp4", "sitagliptin": "dpp4", "jardiance": "sglt2",
    "empagliflozin": "sglt2", "farxiga": "sglt2", "ozempic": "glp1", "semaglutide": "glp1",
    "trulicity": "glp1", "levothyroxine": "thyroid", "synthroid": "thyroid", "methimazole": "thyroid",
    # pain / msk
    "ibuprofen": "nsaid", "motrin": "nsaid", "advil": "nsaid", "naproxen": "nsaid", "aleve": "nsaid",
    "meloxicam": "nsaid", "celebrex": "nsaid", "diclofenac": "nsaid", "toradol": "nsaid",
    "acetaminophen": "analgesic", "tylenol": "analgesic", "tramadol": "opioid", "oxycodone": "opioid",
    "hydrocodone": "opioid", "norco": "opioid", "percocet": "opioid", "morphine": "opioid",
    "gabapentin": "gabapentinoid", "neurontin": "gabapentinoid", "pregabalin": "gabapentinoid",
    "lyrica": "gabapentinoid", "cyclobenzaprine": "muscle_relaxant", "flexeril": "muscle_relaxant",
    "methocarbamol": "muscle_relaxant", "baclofen": "muscle_relaxant", "tizanidine": "muscle_relaxant",
    "prednisone": "steroid", "prednisolone": "steroid", "methylprednisolone": "steroid",
    "medrol": "steroid", "dexamethasone": "steroid", "cortisone": "steroid", "allopurinol": "gout",
    "colchicine": "gout",
    # psych / neuro
    "sertraline": "ssri", "zoloft": "ssri", "fluoxetine": "ssri", "prozac": "ssri",
    "citalopram": "ssri", "celexa": "ssri", "escitalopram": "ssri", "lexapro": "ssri",
    "paroxetine": "ssri", "bupropion": "ndri", "wellbutrin": "ndri", "venlafaxine": "snri",
    "duloxetine": "snri", "cymbalta": "snri", "trazodone": "sari", "mirtazapine": "nassa",
    "buspirone": "anxiolytic", "alprazolam": "benzo", "xanax": "benzo", "lorazepam": "benzo",
    "ativan": "benzo", "clonazepam": "benzo", "klonopin": "benzo", "diazepam": "benzo",
    "zolpidem": "hypnotic", "ambien": "hypnotic", "quetiapine": "antipsychotic",
    "seroquel": "antipsychotic", "risperidone": "antipsychotic", "aripiprazole": "antipsychotic",
    "olanzapine": "antipsychotic", "zyprexa": "antipsychotic", "lithium": "mood_stabilizer",
    "lamotrigine": "anticonvulsant", "lamictal": "anticonvulsant", "levetiracetam": "anticonvulsant",
    "keppra": "anticonvulsant", "topiramate": "anticonvulsant", "sumatriptan": "triptan",
    "imitrex": "triptan", "adderall": "stimulant", "methylphenidate": "stimulant",
    "ropinirole": "dopamine_agonist", "donepezil": "cholinesterase",
    # respiratory / allergy
    "albuterol": "saba", "proair": "saba", "fluticasone": "ics", "flonase": "ics",
    "budesonide": "ics", "symbicort": "ics_laba", "advair": "ics_laba", "montelukast": "ltra",
    "singulair": "ltra", "tiotropium": "lama", "spiriva": "lama", "cetirizine": "antihistamine",
    "zyrtec": "antihistamine", "loratadine": "antihistamine", "claritin": "antihistamine",
    "fexofenadine": "antihistamine", "allegra": "antihistamine", "benadryl": "antihistamine",
    "diphenhydramine": "antihistamine", "hydroxyzine": "antihistamine", "benzonatate": "antitussive",
    # gi
    "omeprazole": "ppi", "prilosec": "ppi", "pantoprazole": "ppi", "protonix": "ppi",
    "esomeprazole": "ppi", "nexium": "ppi", "famotidine": "h2", "pepcid": "h2",
    "ondansetron": "antiemetic", "zofran": "antiemetic", "metoclopramide": "antiemetic",
    "miralax": "laxative", "docusate": "laxative", "dicyclomine": "antispasmodic",
    # anti-infective
    "amoxicillin": "penicillin", "augmentin": "penicillin", "penicillin": "penicillin",
    "azithromycin": "macrolide", "zithromax": "macrolide", "doxycycline": "tetracycline",
    "cephalexin": "cephalosporin", "keflex": "cephalosporin", "ceftriaxone": "cephalosporin",
    "ciprofloxacin": "fluoroquinolone", "levofloxacin": "fluoroquinolone",
    "nitrofurantoin": "urinary_abx", "macrobid": "urinary_abx", "bactrim": "sulfa",
    "clindamycin": "lincosamide", "metronidazole": "nitroimidazole", "flagyl": "nitroimidazole",
    "fluconazole": "antifungal", "valacyclovir": "antiviral", "acyclovir": "antiviral",
    "tamiflu": "antiviral", "paxlovid": "antiviral",
    # gu / other
    "tamsulosin": "alpha_blocker", "flomax": "alpha_blocker", "finasteride": "5ari",
    "oxybutynin": "antimuscarinic", "sildenafil": "pde5", "methotrexate": "dmard",
    "hydroxychloroquine": "dmard", "humira": "biologic", "vitamin": "supplement",
    "iron": "supplement", "calcium": "supplement", "potassium": "electrolyte",
}

# Sound-alike / look-alike pairs (in the spirit of ISMP's confused drug name list).
# Used by the ASR simulator: a recognizer can turn one into the other.
SOUND_ALIKE = [
    ("hydroxyzine", "hydralazine"), ("celebrex", "celexa"), ("zyrtec", "zyprexa"),
    ("lamictal", "lamisil"), ("clonidine", "klonopin"), ("tramadol", "trazodone"),
    ("prednisone", "prednisolone"), ("glipizide", "glyburide"), ("zantac", "xanax"),
    ("toradol", "tramadol"), ("keppra", "kaletra"), ("novolog", "novolin"),
    ("humalog", "humulin"), ("amlodipine", "amiloride"), ("oxycodone", "oxycontin"),
    ("bupropion", "buspirone"), ("sertraline", "cetirizine"), ("risperidone", "ropinirole"),
    ("levetiracetam", "levofloxacin"), ("metformin", "metronidazole"), ("lasix", "luvox"),
    ("losartan", "valsartan"), ("lisinopril", "lipitor"), ("flomax", "fosamax"),
]

LATERALITY = {"left": "right", "right": "left"}

# Symptoms usable for fabricated review-of-systems / HPI insertions
SYMPTOMS = [
    "chest pain", "shortness of breath", "palpitations", "fever", "chills", "night sweats",
    "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain", "headache",
    "dizziness", "syncope", "blurred vision", "cough", "wheezing", "hematuria", "dysuria",
    "joint swelling", "numbness", "tingling", "weight loss", "rash", "leg swelling",
    "blood in the stool", "difficulty swallowing", "confusion", "tremor", "back pain",
]

TIME_UNITS = ["day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours"]
TIME_SWAP = {"day": "week", "days": "weeks", "week": "month", "weeks": "months",
             "month": "year", "months": "years", "year": "month", "years": "months",
             "hour": "day", "hours": "days"}

FAMILY = ["mother", "father", "brother", "sister", "son", "daughter", "grandmother", "grandfather"]

FREQUENCIES = ["once daily", "twice daily", "three times daily", "every 6 hours",
               "every 8 hours", "at bedtime", "as needed", "once weekly", "twice a day", "once a day"]

NEGATION_CUES = ["denies", "denied", "no", "not", "without", "negative", "never", "none",
                 "absent", "n't", "nor", "unremarkable", "free of", "doesn't", "don't",
                 "didn't", "hasn't", "haven't", "isn't", "wasn't"]

# Clinician retraction phrasings. Two disjoint sets so a detector can never be tuned
# on the phrasing it is tested on.
RETRACTION_TEMPLATES_A = [
    "[doctor] actually , let's hold off on the {med} for now .",
    "[doctor] you know what , scratch that , we won't start the {med} today .",
    "[doctor] on second thought , i don't want you on {med} right now .",
]
RETRACTION_TEMPLATES_B = [
    "[doctor] hmm , wait . given your kidneys , let's not do the {med} after all .",
    "[doctor] i'm going to take back the {med} , we'll revisit it next visit .",
    "[doctor] let's cancel that {med} order , i'd rather watch and wait .",
]

STOPWORDS = set("""a an the and or but if of to in on at for with by from as is are was were be been
being this that these those it its he she they them his her their we our you your i me my mine
has have had do does did will would can could should may might must shall not no yes so than then
there here about into over under again further once very just also too only own same some such
any each few more most other both all nor up down out off which who whom what when where why how
patient pt mr mrs ms dr doctor reports states notes today he's she's well okay ok""".split())
