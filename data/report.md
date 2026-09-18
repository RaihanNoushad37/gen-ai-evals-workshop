# Data preparation report

Source: `wandb/RAGTruth-processed`, fingerprint `531e1c94db2b6ce7`. Seed: 17.

## Filtering
- QA task rows: 5034
- Retained (n_tokens <= 400): 2079 (41.3%)
- Discarded (n_tokens > 400): 2955 (58.7%)

## Hallucination rate: retained vs discarded (transportability caveat)
- Retained pool hallucination rate: 0.2347
- Discarded pool hallucination rate: 0.3641
- If these differ meaningfully, sensitivity/specificity measured on the (short, retained) calibration set may not transport to longer contexts.

## Splits
| split | n | prevalence | median n_tokens |
|---|---|---|---|
| explore | 12 | 0.333 | 332 |
| dev | 16 | 0.500 | 352 |
| calib | 24 | 0.500 | 306 |
| prod | 24 | 0.208 | 318 |
