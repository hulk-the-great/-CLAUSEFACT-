# Methodology and reproducibility notes

This document maps the supplied CLAUSEFACT manuscript (Section III, pages 3–4) to the accompanying Python implementation. **The manuscript is a high-level method description, not a complete reproducibility package.** Decisions below are documented to avoid presenting unspecified parameters as facts from the paper.

## 1. Problem

Given source document `D` and generated summary `Y`, the verifier estimates a document faithfulness score `F(D,Y)` in `[0,1]` and a score for each factual clause.

## 2. Clause decomposition

From the paper:

- split summary sentences on punctuation, conjunctions, dependency/attribution structures;
- preserve entities, numbers, negation and event triggers;
- encode clauses as `h_k = f_c(c_k)`;
- importance prior `q_k = 1 + α1 I_entity + α2 I_number + α3 I_relation`.

In code: `preprocessing.decompose_clauses` uses punctuation and conservative conjunction rules; `fact_signals` uses capitalization, number regexes and a small relation lexicon. Neural dependency parsing and coreference are **not implemented**. The encoder uses mean pooling over contextual token outputs from a Hugging Face `AutoModel`. `alpha_entity=0.5`, `alpha_number=0.5`, `alpha_relation=0.25` are **implementation choices**.

## 3. Graph-structured source evidence

Source is segmented into sentences and overlapping two-sentence windows. Each span `v_j` becomes `t_j = f_v(v_j)`. Graph edges capture (a) adjacent source spans, (b) shared approximate named entities, (c) shared numbers and (d) lexical Jaccard overlap above a threshold.

Clause–evidence affinity:

\[
a_{kj}=\operatorname{softmax}_j(h_k^{\top}W_a t_j).
\]

One-hop graph propagation:

\[
\tilde t_j=\sum_{r\in N(j)\cup\{j\}} \beta_{jr}t_r.
\]

The manuscript does not define `β`. Here `β` is masked softmax of scaled dot products between neighbor embeddings. Self-loops ensure every node is reachable from itself. `max_spans=80`, `window_size=2` and overlap threshold `0.27` are implementation choices; not measured optima.

## 4. Dual support–contradiction scoring

As specified in the manuscript:

\[
s_{kj}=\sigma(W_s[h_k;\tilde t_j;|h_k-\tilde t_j|]+b_s)
\]

\[
r_{kj}=\sigma(W_r[h_k;\tilde t_j;h_k\odot\tilde t_j]+b_r)
\]

For each clause, take the strongest support and contradiction over all evidence spans:

\[
s_k^*=\max_j s_{kj}, \quad r_k^*=\max_j r_{kj}
\]

The preliminary clause score is:

\[
f_k=\sigma(\gamma_1s_k^*-\gamma_2r_k^*+\gamma_3q_k).
\]

`γ1`, `γ2`, `γ3` are trainable scalars. `s_k*` is trained on `clause_labels`, `r_k*` on `contradiction_labels`; document labels additionally supervise the aggregated score. The paper does not list head initialization, exact backbone or optimal loss coefficients.

## 5. Uncertainty-aware aggregation

\[
u_k=-\sum_j a_{kj}\log a_{kj}, \quad
\hat u_k=u_k/\log M.
\]

For one evidence span, set normalized uncertainty to zero to avoid division by zero.

\[
w_k=q_k(1-\hat u_k), \quad
F(D,Y)={\sum_k w_k f_k\over\sum_k w_k}.
\]

The code floors `(1 - normalized_uncertainty)` at `1e-4` for numerical stability; no claim is made that entropy weighting alone calibrates probabilities.

## 6. Training loss

The manuscript proposes:

\[
\mathcal{L}=\lambda_1\mathcal{L}_{doc}+\lambda_2\mathcal{L}_{clause}+\lambda_3\mathcal{L}_{contra}+\lambda_4\mathcal{L}_{cal}.
\]

Implementation choices:

| Term | Implementation |
| --- | --- |
| `L_doc` | binary cross-entropy on document probability |
| `L_clause` | binary cross-entropy on strongest support score, if per-clause annotations exist |
| `L_contra` | binary cross-entropy on strongest contradiction score, if corresponding annotations exist |
| `L_cal` | Brier squared probability error as a differentiable calibration-related regularizer |
| default lambdas | `1.0`, `1.0`, `1.0`, `0.1` |

The exact definition of `L_cal`, specific weights, learning rate, dropout, model architecture name, early stopping patience and optimizer configuration are **not supplied in the manuscript**. Defaults here are engineering starting points and require validation.

## 7. Experimental protocol

The manuscript reports document F1 on XSum and CNN/DailyMail, clause F1, AUROC and ECE, with baselines FactCC, QAGS, SummaC and DeBERTa-NLI. The supplied PDF does **not** include the underlying annotations, train/test IDs, checkpoints, scripts or seed-level results. Accordingly:

- This repo does **not** claim to reproduce its numeric tables/figures.
- Example data are handcrafted toy records and cannot validate the manuscript claims.
- The `run_ablations.py` utility trains independent variants and computes **new** results, never hardcoded paper outcomes.
- Controlled studies need an appropriately labeled, license-compliant corpus, multiple random seeds, consistent splits, appropriate baseline implementations and hyperparameter selection on validation data only.
- Scoring with an untrained checkpoint should not be interpreted as calibrated or trustworthy.

## 8. Design caveats

- Long articles with more than 80 spans are truncated **before** neural retrieval. A production implementation should chunk and retrieve at corpus scale.
- Mean-pooled shared encoders are a practical substitute for separate trained clause/source encoders, whose exact design is unspecified.
- The graph captures simple span relationships; it does not resolve pronouns or deeper discourse identity.
- A clause-level binary label does not necessarily distinguish contradiction from absence of evidence. This is why `contradiction_labels` are separate.
- Evaluation uses positive-class F1 (faithful=1), document-level AUROC and confidence ECE. For hallucination detection reporting, you may also compute inverted-label F1; do not mix the two definitions.
