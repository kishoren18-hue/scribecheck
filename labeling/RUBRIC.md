# ScribeCheck labeling rubric (v0.1)

You will see one sentence from an AI-written (or human-written) clinical note and the encounter ID.
Open the visit transcript for that encounter. Judge the sentence **only against what was said in
the visit**. Do not use outside medical knowledge to decide whether the sentence is "probably true".

## 1. support (required)
| Label | Use when |
|---|---|
| SUPPORTED | Everything the sentence claims was said or directly entailed ("baby aspirin" entails 81 mg). |
| UNSUPPORTED | The sentence adds something the visit never said: a drug, dose, side, finding, plan step, or history item. |
| CONTRADICTED | The visit says something incompatible (wrong side, wrong dose, denied symptom recorded as present, a plan the clinician retracted). |
| TRANSCRIPT_ERROR | The sentence is probably right, but the transcript mis-heard it ("57 twice a day" for 750 mg). |

## 2. severity (only if not SUPPORTED)
- MAJOR: could change treatment or cause harm if acted on (wrong drug or dose, wrong side for a procedure, invented allergy, retracted plan recorded as done).
- MODERATE: clinically relevant but unlikely to cause harm by itself (invented exam finding, wrong duration).
- MINOR: cosmetic or administrative.
- NONE: unsupported but harmless (generic counseling sentence).

## 3. certainty (diagnostic statements only; otherwise N/A)
- KEPT: the note keeps the clinician's level of certainty.
- INFLATED: the clinician hedged ("probably", "likely", "rule out", "could be") and the note states it as fact.
- DEFLATED: the clinician was definite and the note hedges.

## Process
- Label independently; do not discuss items until all sheets are in.
- Two or three raters per item. Agreement is reported as Krippendorff's alpha per field
  (`python scripts/08_labeling_kit.py score labeling/sheet_rater*.csv`).
- Disagreements are adjudicated by a third rater and kept in the corpus with both original labels.
  The disagreement itself is data: it shows which error types reviewers cannot see.
