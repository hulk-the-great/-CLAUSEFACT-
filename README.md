# CLAUSEFACT

### Clause-Level Evidence Graphs for Faithfulness Verification in Abstractive Summarization

CLAUSEFACT is a research-oriented Python implementation of **clause-level factual consistency verification**. Given a source document and an abstractive summary, it decomposes the summary into factual clauses, connects evidence spans from the source into a graph, predicts **support and contradiction separately**, and aggregates clause predictions into a **document-level faithfulness score**.

A summary may sound convincing while containing one fabricated date or number. CLAUSEFACT is designed to make these local inconsistencies visible, rather than relying on a single sentence-level similarity score.

> **Project status:** Research prototype / new reference implementation derived from the supplied *CLAUSEFACT: Clause-Level Evidence Graphs for Faithfulness Verification in Abstractive Summarization* manuscript. The manuscript does **not** supply an original codebase, model weights, exact data annotations or all training hyperparameters. This repository implements the central method, but **does not claim to reproduce the paper's published experimental scores**.

## Example

**Source**

> Maya founded Orion Labs in 2018. Orion Labs is based in Boston.

**Generated summary**

> Maya founded Orion Labs in 2021.

The summary contains a **numerical inconsistency** (`2021` instead of `2018`). After training, CLAUSEFACT can produce a document-level probability, clause-level scores, and the highest-scoring source evidence for both support and contradiction. **Scores are learned; this example does not assert that an untrained model can detect the error.**

## Features

- **Atomic clause decomposition:** rule-assisted splitting of sentence-local factual propositions
- **Source-side evidence graph:** connections through discourse adjacency, entity continuity, number continuity and lexical overlap
- **Transformer encoders:** shared pretrained backbone for clauses and source spans
- **Learned evidence affinity:** clause-to-span softmax attention
- **One-hop graph propagation:** attention-normalized aggregation over evidence neighbors
- **Dual verification heads:** independent support and contradiction probabilities
- **Fact-aware priors:** higher importance for entity, numerical and relation-bearing clauses
- **Uncertainty-aware scoring:** normalized evidence-attention entropy downweights diffuse evidence
- **Multi-objective training:** document, clause-support, contradiction and calibration losses
- **Evaluation and diagnostics:** document/clause F1, AUROC, ECE, explanations and saved JSONL predictions
- **Ablations:** remove clause decomposition, graph context, contradiction modeling, uncertainty weighting or importance weighting

## Architecture

```text
Source document                         Generated summary
      |                                        |
Sentence + overlapping spans          Clause decomposition
      |                                        |
Transformer evidence encoder          Transformer clause encoder
      |                                        |
Source-side evidence graph                 Clause vectors
      |                                        |
One-hop neighbor aggregation <---- Learned clause-to-span affinity
      |                                        |
      +-------- Support head / Contradiction head ------+
                               |
                    Clause-level scores
                               |
                 Evidence-attention uncertainty
                               |
                    Weighted aggregation
                               |
              Document faithfulness probability
                               |
                   JSON evidence report
```

## Repository structure

```text
CLAUSEFACT/
├── src/clausefact/
│   ├── cli.py                 # inspect, train, evaluate, predict
│   ├── preprocessing.py       # clause splitter, spans, factual priors
│   ├── graph.py               # evidence graph construction
│   ├── model.py               # transformer, attention, heads and aggregation
│   ├── engine.py              # optimization, checkpoints, inference
│   ├── metrics.py             # F1, AUROC, ECE
│   └── data.py                # JSONL validation
├── data/example/              # tiny illustrative labeled data
├── scripts/
│   ├── check_dataset.py
│   ├── prepare_from_csv.py
│   └── run_ablations.py
├── docs/
│   ├── METHODOLOGY.md
│   └── DATASETS.md
├── tests/                     # automated tests
├── .github/workflows/tests.yml
├── examples/
│   ├── source.txt
│   └── summary.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Requirements

- Python **3.10+**
- CPU or CUDA-capable GPU (a GPU is recommended for transformer fine-tuning)
- Internet access at first model download, unless the selected Hugging Face backbone is already cached
- No paid LLM API key is needed

The default encoder is `distilbert-base-uncased`. Other compatible Hugging Face `AutoModel` backbones may be supplied through `--backbone`. **No trained CLAUSEFACT checkpoint is bundled.**

## Install

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CLAUSEFACT.git
cd CLAUSEFACT
```

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[all]"
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

For preprocessing and CLI inspection **without downloading torch/transformers**, install only the minimal package:

```bash
pip install -e .
```

## Try it without a model download

The `inspect` command previews clause splitting and source evidence construction. It **does not** produce factuality predictions:

```bash
clausefact inspect \
  --source-file examples/source.txt \
  --summary-file examples/summary.txt \
  --out outputs/inspect.json
```

On Windows PowerShell, use a single line or backtick continuation instead of `\`:

```powershell
clausefact inspect --source-file examples/source.txt --summary-file examples/summary.txt --out outputs/inspect.json
```

## Train a model

A tiny labeled dataset is included solely to verify the code paths. It is **not a scientific benchmark** and is far too small to justify factuality-performance claims.

```bash
clausefact train \
  --train data/example/train.jsonl \
  --val data/example/val.jsonl \
  --out outputs/demo_model \
  --backbone distilbert-base-uncased \
  --epochs 3 \
  --device auto
```

**Windows PowerShell (single line)**

```powershell
clausefact train --train data/example/train.jsonl --val data/example/val.jsonl --out outputs/demo_model --backbone distilbert-base-uncased --epochs 3 --device auto
```

Training downloads the backbone (if necessary), optimizes the complete encoder and dual heads, uses gradient clipping, and saves the **best checkpoint by validation F1**. It also uses early stopping. The training-loss weights and architecture switches are configurable with `clausefact train --help`.

Saved files (ignored by Git):

```text
outputs/demo_model/
├── backbone/             # fine-tuned transformer weights + config
├── tokenizer/            # tokenizer files
├── model.pt              # complete CLAUSEFACT state dict
├── config.json
├── training_history.json
└── training_meta.json
```

## Evaluate

```bash
clausefact evaluate --data data/example/test.jsonl --checkpoint outputs/demo_model --out outputs/demo_eval
```

This writes `metrics.json` and `predictions.jsonl`. Document-level metrics include F1, precision, recall, AUROC, accuracy and ECE. Clause-level metrics are computed when corresponding annotation arrays exist. AUROC is returned as `null` if a set contains only one label class.

## Verify a source document and summary

```bash
clausefact predict \
  --source-file examples/source.txt \
  --summary-file examples/summary.txt \
  --checkpoint outputs/demo_model \
  --out outputs/prediction.json
```

Or enter inline text:

```bash
clausefact predict --source-text "Orion Labs opened in 2018." --summary-text "Orion Labs opened in 2020." --checkpoint outputs/demo_model
```

Example **output structure** (numbers omitted because they require a trained model):

```json
{
  "document_faithfulness_probability": 0.0,
  "predicted_faithful": false,
  "num_evidence_spans": 1,
  "graph_edge_count": 0,
  "clauses": [
    {
      "clause": "Orion Labs opened in 2020.",
      "faithfulness_probability": 0.0,
      "support_score": 0.0,
      "contradiction_score": 0.0,
      "importance_prior": 0.0,
      "normalized_uncertainty": 0.0,
      "normalized_weight": 0.0,
      "best_support_evidence": "Orion Labs opened in 2018.",
      "best_contradicting_evidence": "Orion Labs opened in 2018.",
      "highest_attention_evidence": "Orion Labs opened in 2018."
    }
  ]
}
```

All `0.0` values above are **placeholders illustrating the JSON schema, not predictions or benchmark measurements**. The highest-scoring evidence for support and contradiction can be the same span; the two heads assign different scores to that span.

## Data format

Each line in a `.jsonl` file is an independent example:

```json
{"id":"example-1","source":"Ava founded Zen Labs in 2017.","summary":"Ava founded Zen Labs in 2019.","label":0,"clause_labels":[0],"contradiction_labels":[1]}
```

- `label`: **1** if the *entire* summary is faithful, **0** otherwise.
- `clause_labels`: **1** for a supported clause, **0** for an unsupported or contradicted clause.
- `contradiction_labels`: **1** for explicit contradiction; **0** for support or absence of evidence. An unsupported claim need not be contradicted.
- Both clause-label arrays are optional. If provided, they **must align with the exact output of the chosen clause splitter**. The current training pipeline validates lengths automatically.

Read [Dataset Preparation and Labeling](docs/DATASETS.md) before using this code for research evaluation. XSum and CNN/DailyMail article-summary pairs do not automatically supply human faithfulness labels.

Convert your labeled CSV to JSONL splits:

```bash
python scripts/prepare_from_csv.py --input path/to/annotations.csv --out data/processed
python scripts/check_dataset.py data/processed/train.jsonl
```

## Ablations

The paper motivates a comparison with and without major components. To independently train and measure six variants:

```bash
python scripts/run_ablations.py \
  --train data/example/train.jsonl \
  --val data/example/val.jsonl \
  --test data/example/test.jsonl \
  --out outputs/ablations \
  --epochs 3
```

Variants: `full`, `without_graph`, `without_contradiction`, `without_calibration`, `without_importance`, `without_clauses`. The no-clause variant uses **document-only** supervision so its labels remain meaningful.

Use a real labeled dataset, multiple seeds, and controlled training budgets before interpreting any differences as scientific conclusions. The output is `outputs/ablations/ablation_metrics.json`, generated from actual model runs.

## Tests

```bash
python -m pytest -q
```

The tests cover splitting, source windows, graph symmetry/ablation, metrics, JSONL validity, and a mocked-backbone forward/backward pass. The mocked model test requires PyTorch but downloads **no** transformer. GitHub Actions runs the dependency-light tests and CLI inspection.

## Method summary

For clause representation \(h_k\), source span \(t_j\), and learned affinity matrix \(W_a\), CLAUSEFACT estimates clause-to-evidence attention, performs one-hop graph propagation, and computes independent support and contradiction heads. Each clause is given an importance prior based on entities, numbers and relations. Evidence-attention entropy estimates uncertainty and controls the clause's contribution to the final document score.

The full equations, objective, design choices, and explicit departures from unspecified manuscript details are provided in [Methodology & Reproducibility](docs/METHODOLOGY.md).

## Limitations

1. **This is not a pretrained released research model.** You must train on suitable labeled data to obtain meaningful probabilities.
2. **Clause splitting is heuristic.** Implicit subjects, complex relative clauses and coreference may cause incorrect decomposition.
3. **Entity continuity is approximated by capitalization rules.** The prototype has no full named-entity disambiguation or neural coreference resolver.
4. **Long inputs are truncated** per span and bounded by `max_spans`; this can omit relevant evidence.
5. **Uncertainty weighting is not a guarantee of probability calibration.** Brier regularization and held-out ECE make calibration measurable, but substantial calibration testing may be necessary.
6. **Baseline methods are not shipped.** The paper references FactCC, QAGS, SummaC and DeBERTa-NLI; their reported scores are not re-created here.
7. **Manuscript hyperparameters are incomplete.** Encoder name, graph affinity parameterization, optimizer details and exact annotations are engineering choices, documented separately.

## Reference

Based on the user-provided manuscript:

*CLAUSEFACT: Clause-Level Evidence Graphs for Faithfulness Verification in Abstractive Summarization* (six-page manuscript). Its provided PDF contains placeholder author metadata; do not invent an author list or publication venue when citing it.

Related verification systems discussed in the manuscript include FactCC, QAGS, SummaC and DeBERTa-NLI. Their original implementations are not bundled in this repository.

## License

MIT — see [LICENSE](LICENSE). Review attribution and data/model license terms before distributing your own trained checkpoints or external evaluation datasets.
