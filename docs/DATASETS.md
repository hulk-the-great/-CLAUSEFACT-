# Dataset preparation and annotation

## Required annotation contract

CLAUSEFACT trains from UTF-8 newline-delimited JSON objects (`.jsonl`). Each entry contains `source`, `summary`, and a document-level `label` (0/1). Add optional `id`, `clause_labels`, and `contradiction_labels`.

```json
{
  "id": "example-001",
  "source": "The agency reported 20 new cases on Monday.",
  "summary": "The agency reported 40 new cases on Monday.",
  "label": 0,
  "clause_labels": [0],
  "contradiction_labels": [1]
}
```

- `label=1`: every factual claim in the summary is supported by the source (faithful); otherwise `0`.
- `clause_labels[k]=1`: the k-th extracted clause is supported by the source. Use `0` for either contradicted or unsupported clauses.
- `contradiction_labels[k]=1`: the source actively contradicts the k-th clause (e.g., wrong date, number, entity, polarity). Use `0` for unsupported-but-not-contradicted clauses.
- If clause annotations are absent, the training script uses document supervision and calibration loss only; unannotated head supervision is skipped.
- Labels must agree with the exact clause split count. Run `clausefact inspect` on examples to see the splitter's actual clauses and adjust annotations accordingly.

## Data provenance

XSum and CNN/DailyMail are summarization datasets; article-summary pairs **do not automatically constitute gold factuality labels**. Obtain an independently annotated factuality dataset or collect and audit labels before making research claims. Synthetic entity/number/negation perturbations can augment training but must be segregated from held-out human-annotated evaluation.

Recommended schema for author-labeled CSV:

```csv
id,source,summary,label,split,clause_labels,contradiction_labels
```

For array columns, store JSON-formatted arrays using proper CSV quoting, e.g. `"[1, 0]"`.

Convert CSV:

```bash
python scripts/prepare_from_csv.py --input annotations.csv --out data/processed
python scripts/check_dataset.py data/processed/train.jsonl
```

## Split hygiene

1. Group all summaries derived from the **same source document** into only one partition to avoid document leakage.
2. Do not tune hyperparameters or adjust threshold decisions on test data.
3. Use train/validation/test partitions; preserve a held-out test partition.
4. Keep annotation guidelines and class prevalence consistent across evaluated methods.
5. For papers, report seeds, confidence intervals, model architecture, training budget, dataset version, label taxonomy and failure cases.

## Metrics

Positive-class `F1` here treats `faithful=1` as positive. Report separately if you also evaluate `hallucinated=1` as the positive class. AUROC is threshold-free, and expected calibration error is computed using ten equal-width confidence bins (`max(p,1-p)`). Clause-level metrics are computed when clause annotations are available. Scores are probabilities produced by a *trained* model, not estimates available from preprocessing alone.

## Demo files

The included `data/example/{train,val,test}.jsonl` files are **24 hand-authored toy examples** from distinct source-document groups. They are deliberately small and intended only for installation, pipeline demonstration and correctness checks. No research comparison or accuracy claim should be based on them.
