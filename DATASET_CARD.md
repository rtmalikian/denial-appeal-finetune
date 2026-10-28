# Dataset Card — Synthetic Denial / Appeal Dataset

> **All data in this repository is SYNTHETIC.** No real insurer, member,
> patient, claim, or policy data is used, reproduced, or imitated. Denial
> codes (`DEN-###`) are synthetic workflow labels, not a real payer code set.

## Origin and methodology (the 10k workflow, reproduced openly)

Raphael's original medical-billing workflow generated **10,000 synthetic
denial/appeal letters with Codex, with human-in-the-loop review**: samples
were reviewed, labelling rules were corrected, and the corrected rules were
re-applied. This repository reproduces that methodology as inspectable code
at a scale anyone can run on a laptop:

1. A fixed cause taxonomy (8 categories, `src/denial_appeal/categories.py`).
2. Templated denial letters per cause (`generator.py`): 3 letter templates ×
   2 extra-detail variants per cause × 10 services, filled deterministically
   from a seed.
3. **Labels by construction**: the cause that generated a letter is its
   label. There is no separate noisy annotation step to audit — the
   labelling rule is the generator itself.
4. A template appeal per record that cites the denial reason phrase
   verbatim, attaches the cause-appropriate evidence type, and invents no
   codes or policy numbers (enforced by `quality.py`).
5. A **simulated** appeal outcome drawn from the record's success
   probability, so probability bands can be checked for internal
   calibration. Simulated outcomes are not real appeal results.

## Committed sample

| File | Records | Purpose |
|---|---|---|
| `data/sample/train.jsonl` | 600 | Training (seed 42) |
| `data/sample/test.jsonl` | 150 | Held-out test (seed 1042) |
| `data/sample/sft_train.jsonl` | 600 | Chat-format SFT export of train |
| `data/sample/preferences_train.jsonl` | 600 | DPO-style chosen/rejected pairs |
| `data/sample/hashes.json` | — | SHA-256 of each file above |

SHA-256 hashes are also embedded in `artifacts/run_manifest.json` for the
exact files the baseline was trained/evaluated on.

## Schema (per record)

`id`, `synthetic: true`, `denial_letter`, `cause`, `cause_label`,
`denial_code` (synthetic), `service`, `appeal_success_probability`,
`appeal_success_probability_band` (low <0.40, medium <0.65, high ≥0.65),
`simulated_appeal_succeeded`, `appeal_letter`.

## Categories

medical_necessity, prior_auth_missing, coding_bundling, eligibility,
timely_filing, duplicate_claim, documentation_insufficient,
experimental_investigational.

## Scaling to 10k

```bash
python - <<'PY'
import sys; sys.path.insert(0, "src")
from denial_appeal.generator import generate_records, write_jsonl
write_jsonl(generate_records(10000, seed=42), "data/denials_10k.jsonl")
PY
```
Generation is deterministic and linear; 10k records take seconds. Human-in-
the-loop at scale means: sample records per cause, read them, fix the
templates/rules in `categories.py`/`generator.py`, regenerate, and record
the data hash of the version you train on (the run manifest does this).

## Limits — read before citing anything

- Templated synthetic language is **separable by construction**: a
  classifier can score near-perfectly on this test set while being far from
  real-world performance. Accuracy on this data proves the pipeline works,
  not that the model works on real denial letters.
- Success probabilities are cause-level priors plus noise. They are not
  learned from, or validated against, real appeal outcomes.
- Real denial letters vary by payer, plan, and jurisdiction in ways this
  generator does not model. Do not train a production system on this data
  alone.
