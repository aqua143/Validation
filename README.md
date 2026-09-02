
# RCT-Reviewer Validation Harness

This suite provides validation for the RCT-Reviewer. It bridges the gap between the 2017 original code (`robotreviewer-master`) and the 2026 refactored code (`RCT-Reviewer`), proving mathematical fidelity, predictive validity, and modern infrastructure robustness.

## Directory Structure
- `RCT-Reviewer/` : The refactored (2026) repository (Python 3.12) - the tool under test.
- `robotreviewer-master/` : The original (2017) repository - the reference implementation.
- `validation_shim.py` : The compatibility shim that runs the original 2017 code inside the modern Python 3.12 venv without touching either repo.
- `fetch_corpus.py` : Downloads recent open-access RCT PDFs from Europe PMC, filtered by RCT-Reviewer's own SVM.
- `evaluate.py` : Runs the 4 validation tiers and generates all statistics, CSVs, and figures.
- `corpus/` : The downloaded PDF corpus (1,000 PDFs, see below).
- `validation_results/` : All outputs - `report.md`, per-tier CSVs/JSONs, and figures in **PNG + SVG + PDF**.
- `requirements.txt` : Pinned environment (latest majors; `pandas<3` because streamlit 1.55 in RCT-Reviewer requires it).

## Environment
Tested with: Python 3.12.12, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0, spaCy 3.8.16 (+ `en_core_web_sm` 3.8.0), PyMuPDF 1.28.2, matplotlib 3.11.1, seaborn 0.13.2, statsmodels 0.15.0, pandas 2.3.3, pydantic 2.13.5. `provenance.json` records the exact versions of every run.

## How to Run (Step-by-Step)

1. Ensure you are in the `RCT-Reviewer-Validation` directory by cloning this repository:

```bash
git clone https://github.com/RCT-Reviewer/Validation.git
```

2. **Get the two upstream codebases (first time only).** This repo ships the harness, results, and corpus metadata - not the upstream code or the article PDFs. Place both repositories at these exact paths (the harness resolves `RCT-Reviewer/` and `robotreviewer-master/` relative to its own location):

```bash
# RCT-Reviewer (the refactored tool under test): model weights are stored with Git LFS, so install LFS once (git lfs install) and pull the weights:
git clone https://github.com/aurumz-rgb/RCT-Reviewer.git RCT-Reviewer
cd RCT-Reviewer && git lfs pull && cd ..
```

```bash
# RobotReviewer (the original 2017 reference implementation). Pull its LFS weights too, so both codebases hold their real model files and the original pipeline is fully runnable for independent verification:
git clone https://github.com/ijmarshall/robotreviewer.git robotreviewer-master
cd robotreviewer-master && git lfs pull && cd ..
```

3. Make a .venv environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

5. Corpus retrieval (three options)

The article PDFs are not redistributed in this repository (publisher licensing). Three ways to obtain the 1,000-PDF corpus:

-- **Zenodo archive (recommended, exact).** The full corpus used in the paper is deposited on Zenodo with restricted access - request access and download the byte-identical set: [https://doi.org/10.5281/zenodo.22260256](https://doi.org/10.5281/zenodo.22260256). Unzip into `corpus/`.

 -- **Exact 1:1 rebuild from the audited metadata.** `corpus/metadata.csv` records every accepted paper's PMCID. This re-downloads exactly those papers from Europe PMC (skipping cached files; verified 1,000/1,000):
   ```bash
   python fetch_corpus.py --pmcid-list corpus/metadata.csv
   ```
 Expect ~1 h (network-bound, ~0.5 s politeness delay per download). Failed downloads are retried on re-run.
 
-- **Fresh query (new comparable corpus, not identical).** `python fetch_corpus.py --target 1000` re-runs the Europe PMC search + SVM filter. Europe PMC result sets drift between searches, so this yields a comparable but not paper-identical corpus; Tier D statistics are descriptive and remain interpretable, while Tiers A/B/C do not use the corpus at all.

6. Execute the full validation suite (Tiers A, B, C, D):

```bash
python evaluate.py --tier all --corpus-dir corpus
```

   Useful flags: `--tier A|B|C|D` for a single tier

7. Check `validation_results/` for `report.md`, the CSV/JSON outputs, and the figures (`*.png`, `*.svg`, `*.pdf`).


## What reproducing takes (measured on an M1 Pro, 1,000-PDF corpus)

| Step | Time | Notes |
|---|---|---|
| Setup: venv + `requirements.txt` + spaCy model + upstream clones/LFS | 15–25 min | network-bound; RCT-Reviewer light LFS pull (bias + rct weights) ~2.6 GB full; RobotReviewer LFS pull adds its own weights |
| `fetch_corpus.py --target 1000` | 1–2 h | only needed if you don't reuse a corpus; results for Tiers A/B/C do not depend on it |
| `evaluate.py --tier all` - Tier A+B (751 benchmark records) | ~6 min | deterministic, seeded |
| Tier C (1,003 documents × 6 domains, both pipelines) | ~70 min | the dominant cost: runs the original 2017 pipeline over every corpus document |
| Tier D (1,000 PDFs, 12,060 pages) | ~35 min | parse + annotate each PDF |
| **Total** | **≈ 2 h** after setup | 6–8 GB disk (venv + models + corpus) |


## License

GPL-3.0 - the same licence as both upstream repositories (`RCT-Reviewer` and `robotreviewer-master`). See [`LICENSE`](LICENSE). RCT-Reviewer is an independent refactoring of RobotReviewer by a different team; the original authors' model weights are redistributed here under the same licence with attribution, and all published performance claims cited in the results belong to the original evaluations.


**Corpus:  The corpus is 1,000 recent (2025–2026) open-access PDFs from Europe PMC, all accepted by RCT-Reviewer's own SVM. Roughly a third of the titles mention "protocol" (trial protocols that the SVM also flags as RCT-like); they are retained as valid parse targets, and per-paper decisions are in `corpus/metadata.csv`. The corpus is intentionally self-selected, so Tier D measures robustness on tool-relevant PDFs, not a representative literature sample. Tier A/B/C do not depend on the corpus.


## How the Code Works (The Architecture)

### `validation_shim.py` (Compatibility shim)

The original RobotReviewer code relies on dead dependencies (Keras/TensorFlow 1.x, Python 3.6-era scikit-learn). The shim tricks it into running in the modern venv by applying **five compatibility shims**, none of which alter the numerics under test:

1. **Keras stubs** - the original `vectorizer.py`/`rct_robot.py` import Keras at module level. Fake `keras.*` modules are injected into `sys.modules` before import. The original CNN (Keras `.h5`) ensemble is *never executed*  its absence is exactly the ablation studied in Tier B.
2. **scikit-learn kwarg translation** - the original passes `non_negative=True` to `HashingVectorizer`; renamed to `alternate_sign` in scikit-learn 0.24 with identical semantics. `HashingVectorizer.__init__` is wrapped to translate the kwarg; hashing/tokenization behaviour is untouched.
3. **DATA_ROOT redirect** - the original repo's model weights are 132-byte git-LFS pointer stubs. `robotreviewer.DATA_ROOT` is repointed at `RCT-Reviewer/data`, which holds the real `.npz` weight files (the same artifacts the published tool loads; SHA-256 hashes are written to `provenance.json`). This redirect stays in place even when RobotReviewer's own LFS weights are pulled, so both pipelines are guaranteed to load byte-identical weight files - that guarantee is what makes the fidelity comparison in Tiers A and C meaningful. And it is *verified, not assumed*: on every run, `evaluate.py` hashes RCT-Reviewer's weight files and compares them against the original repository's Git LFS object hashes (or direct file hashes for non-LFS files) - all four files verified byte-identical as of the current run.
4. **Renamed-API aliases** - `np.int` (removed in numpy 2.0, used by the original `MiniClassifier.predict`) and `VectorizerMixin` (renamed to `_VectorizerMixin`) are re-aliased to their old names, plus the `sklearn.linear_model.logistic` module alias used by old pickles.
5. **Guarded unpickling** - the original `RCTRobot.__init__` unconditionally unpickles SVM+CNN calibration pickles (unused on the SVM-only path); load failures there are swapped for an inert dummy so construction completes.

On top of the shims it exposes four adapters used by `evaluate.py`:
- `original_bias_annotate(full_text)` - segments text with the *same* spaCy model the refactored tool uses, feeds it to the untouched original `BiasRobot.pdf_annotate`, and returns the original per-domain judgements + top-3 evidence sentences.
- `original_sentence_scores(sentences, domain)` - recomputes sentence-level decision scores with the original vectorizer+classifier, mirroring the original `pdf_annotate` lines verbatim.
- `original_rct_predict(title, abstract)` - original `RCTRobot.predict` in the configuration that matches the refactored tool (`ensemble_type='svm'`, `threshold_type='balanced'`, no publication-type features).
- `load_medline_records(path)` - parses the MEDLINE benchmark file with the original `ris.py` parser for input parity.

### `fetch_corpus.py` 

To guarantee the corpus contains strict RCTs, the fetcher:

1. Queries the Europe PMC REST API for open-access articles with a PDF, published strictly in 2025–2026 (`OPEN_ACCESS:y AND HAS_PDF:y AND (PUB_YEAR:2025 OR PUB_YEAR:2026)`), using `cursorMark` pagination (500/page) and a 0.5 s politeness delay.
2. For every candidate with a title + abstract, runs **RCT-Reviewer's own SVM** (`RCTRobot.predict`). Papers the model does not label as RCTs are skipped and logged.
3. Downloads accepted papers from `europepmc.org/articles/PMC…?pdf=render`, deduplicating by PMCID and skipping PDFs already on disk (resume-capable). Failed downloads are deleted so a re-run retries them.
4. Writes `corpus/metadata.csv` (pmcid, title, year, model score, decision, filename, status) so every inclusion decision is auditable.

### `evaluate.py` (The 4-Tier Validator)

- **Tier A (Predictive validity)** - Parses the 1,000-record Clinical Hedges MEDLINE benchmark (`pubmed_test.txt`) with the *original* parser, keeps records having both title and abstract (n = 751 after dedup), and scores each through the refactored `RCTRobot`. Metrics against the human `hedges_is_rct` ground truth: sensitivity, specificity, accuracy, PPV, NPV, F1, Cohen's kappa - each with a 95% Wilson score interval; F1/kappa CIs use 1,000-resample bootstrap (seed 42). Also computed: ROC AUC and Brier score. *Implementation fidelity* compares the new scores to (a) the stored original-model outputs (`pubmed_expected.json`) and (b) the original 2017 code executed live through the shim - the acceptance gate is max |Δscore| < 1e-9 vs the executed original (achieved: 0.0).

- **Tier B (CNN ablation)** - On the same 751 records, the refactored SVM-only decisions are compared against the stored decisions of the original full SVM+CNN(+ptyp) ensemble: decision agreement + kappa, both systems' metrics vs Hedges, plus **McNemar's exact test** on paired correct/incorrect counts (p < 0.0001). This quantifies the cost of dropping the dead CNN: agreement 92.5%, F1 0.925 → 0.969.

- **Tier C (RoB fidelity)** - Runs both the shimmed 2017 BiasRobot and the refactored BiasRobot over identical sentence lists from the bundled PDFs (`example.pdf`, `sample_bias.pdf`), the GROBID fixture abstract, and every corpus PDF. Per document × domain (6 RoB domains): judgement agreement, top-3 evidence-sentence Jaccard, and max |Δ| sentence decision score. A separate **vectorizer equivalence probe** pushes identical texts (plain + domain-interaction tuples) through the original and refactored `InteractionHashingVectorizer` stacks and asserts byte-identical sparse matrices. Result: 100% agreement, kappa 1.0, max |Δ| = 0, 14/14 probes identical. RoB accuracy therefore transfers by weight-identity (no new RoB ground truth was collected - the report's framing notes state this explicitly).

- **Tier D (Parser robustness - descriptive, not accuracy)** - Parses every corpus PDF with the new PyMuPDF-based `PDFParser`, recording parse success/failure mode, pages, chars, sentences, and per-document time (Wilson CI on success rate, median + IQR timing). It also runs the refactored BiasRobot on each PDF and reports the **keyword hit-rate** of the top-3 highlighted sentences per RoB domain against hand-written methodological-term regex lexicons - a lexical plausibility check of the extracted evidence. Finally, an n=1 **PyMuPDF-vs-GROBID case study** compares PyMuPDF extraction of `example.pdf` against the stored GROBID parse (`pdffile.json`) via title/abstract token coverage.

**Figures** - every figure is saved in **three formats (`.png`, `.svg`, `.pdf`)** by the `save_fig` helper.

| figure | what it shows | how to read it |
|---|---|---|
| `figure_tier_a_confusion_matrix` | The tool's verdicts vs the human "Clinical Hedges" label on all 751 benchmark records. Rows = human truth, columns = the tool's prediction; darker cells = more papers. | The diagonal cells are correct calls: 290 papers correctly rejected as not-RCT (top-left) and 397 RCTs correctly accepted (bottom-right). The off-diagonal cells are the errors: 39 not-RCTs wrongly accepted (top-right) and 25 RCTs missed (bottom-left). The parenthesized percentage in each cell is the count divided by its true-class total - exactly the specificity (88.1%, top row) and sensitivity (94.1%, bottom row) in the Tier A table. |
| `figure_tier_a_calibration` | Reliability diagram: how close the SVM's score is to a true probability. | The raw SVM score is an unbounded decision value, not a probability, so it is first Platt-scaled (logistic regression on the raw score, fit on this benchmark - in-sample, therefore optimistic). Papers are grouped by calibrated probability (8 quantile bins); each point plots mean predicted probability vs the fraction that actually were RCTs. Points near the dotted diagonal mean the calibrated score can be read as a probability. |
| `figure_tier_b_ablation` | What was lost by removing the dead TensorFlow CNN: SVM-only (refactored) vs the original full SVM+CNN ensemble, side by side. | Taller bars = better. The ensemble is slightly better on every metric (F1 0.969 vs 0.925). The report attributes the gap across all three arms: SVM-only agrees 94.5% with SVM+CNN (no publication-type features) but 92.5% with the full SVM+CNN+ptyp ensemble - so most of the loss comes from the publication-type features, with the CNN contributing the remainder; both are unrunnable in a maintained environment. McNemar's test (p < 0.0001) confirms the total gap is statistically significant. |
| `figure_tier_d_keywords` | Lexical plausibility check: of the top-3 evidence sentences the tool highlights per RoB domain per PDF (1,000 × 3 = 3,000 snippets per domain), the share containing an explicit methodological keyword (e.g., "randomly assigned", "sealed envelopes", "double-blind"). | Low rates are expected and are *not* an error rate. The model returns three highlighted sentences for every domain of every PDF - even when the paper never describes that domain - so many snippets are lexically generic (or off-topic) and contain no domain keyword. Randomization (40.5%) and participant blinding (28.6%) are most often described explicitly; allocation concealment (7.2%), selective reporting (4.5%) and outcome-assessor blinding (1.6%) almost never are. Descriptive only. |
| `figure_tier_d_scale` | Parse time vs document length across the corpus. | Points rising roughly along a line (Pearson r ≈ 0.93) show the parser scales linearly with document size and has no blow-ups on long PDFs. |

`report.md` (all tiers) and `master_summary.csv` (headline numbers) are regenerated on every `--tier all` run.

## Current Results Snapshot (1,000-PDF corpus)
- **Tier A** (n=751): Sensitivity 94.1 (91.4–96.0), Specificity 88.1 (84.2–91.2), F1 0.925, kappa 0.826, ROC AUC 0.966 (raw scores), Brier 0.067 (Platt-scaled scores, in-sample), prevalence 56.2%. Fidelity vs executed original code: max |Δ| = 0.0, agreement 100% (vs stored 2016 outputs: max |Δ| = 0.093 from scikit-learn version drift, decisions agree 99.6%).

- **Tier B**: agreement with full ensemble 92.5 (90.4–94.2), kappa 0.847; three-arm attribution - SVM-only agrees 94.5% with SVM+CNN (no ptyp) vs 92.5% with the full SVM+CNN+ptyp ensemble (F1 0.925 vs 0.956 vs 0.969), so the publication-type features account for most of the gap; McNemar p < 0.0001.

- **Tier C**: 6,018/6,018 judgements agree across all 1,003 documents × 6 domains (kappa 1.0, CI 99.9–100.0), sentence scores identical, vectorizer matrices identical (14/14 probes).

- **Tier D** (1,000 PDFs, 12,060 pages): parse success 100.0% (CI 99.6–100.0), median 1.66 s/PDF (IQR 1.32–2.06), longest PDF 7.0 s.


## Conclusion

1. **The refactored tool is mathematically the original.** Against the original 2017 code executed in this environment, the RCT classifier reproduces every score bit-identically (max |Δ| = 0.0, 751/751 records), and the refactored RoB pipeline reproduces the original's judgements in **6,018/6,018** document × domain comparisons (kappa 1.0) with zero sentence-score difference and byte-identical vectorizer matrices. Published accuracy therefore transfers by weight identity, the loaded weights are the original validated artifacts (SHA-256 in `provenance.json`).

2. **Predictive validity is preserved and strong.** On the human-labelled Clinical Hedges benchmark (n=751, 56.2% RCT prevalence): sensitivity 94.1 (91.4–96.0), specificity 88.1 (84.2–91.2), accuracy 91.5, PPV 91.1, NPV 92.1, F1 0.925, Cohen's kappa 0.826 ("almost perfect" agreement), ROC AUC 0.966. PPV/NPV are prevalence-dependent and will differ on streams with a different case mix.

3. **The cost of dropping the dead CNN is quantified and attributed.** The original full SVM+CNN+ptyp ensemble reaches F1 0.969 vs 0.925 for the SVM-only refactored tool (McNemar p < 0.0001). Decomposition across three arms shows the publication-type features, not the CNN, account for most of the gap (agreement 94.5% without ptyp vs 92.5% with it); both components cannot run in a maintained environment (TensorFlow 1.x).

4. **The new parser is robust at scale.** All **1,000/1,000** PDFs parsed successfully (95% CI 99.6–100.0), median 1.66 s per document (IQR 1.32–2.06), processing time scaling linearly with document length (Pearson r = 0.93) with no blow-ups on the longest documents (max 7.0 s across a 12,060-page corpus).

5. **Extracted evidence is lexically plausible where papers describe methods explicitly.** Of the 3,000 top-3 highlighted sentences per RoB domain, 40.5% contain an explicit randomization keyword, 28.6% a participant/personnel blinding keyword, and 14.2% an attrition keyword; allocation concealment (7.2%), selective reporting (4.5%) and outcome-assessor blinding (1.6%) are rarely described explicitly in modern open-access papers. Low rates reflect what papers actually say (and that the model highlights top-3 evidence for every domain, including undescribed ones), not a bug.

6. **Known limitations, stated up front.** The corpus is self-selected by the tool's own SVM (dogfooding) and includes trial protocols; the GROBID comparison is n=1 but documents a genuine PyMuPDF gap (that PDF's title font maps no Unicode); the Platt calibration is fit in-sample (optimistic); no new RoB ground truth was collected - RoB validity transfers by weight identity rather than being re-measured.

## Reproducibility notes
- `provenance.json` stores the run timestamp, package versions, and SHA-256 of every model weight file loaded by the tool.

- Bootstrap CIs are seeded (default 42); the SVM, vectorizers, and hash-based features are deterministic - re-running Tiers A–C reproduces the numbers exactly.

- Tier D sentence counts depend on the spaCy model version; upgrading spaCy can shift them slightly (results from a previous, older-package environment differed by a few sentences per PDF).

- `validation_shim.py` and `evaluate.py` never modify either repository - everything new lives in this directory.

## Troubleshooting
`libmupdf.dylib` Error: run these three commands, one by one:
```bash
RCT-Reviewer/.venv/bin/python -m pip uninstall -y pymupdf
RCT-Reviewer/.venv/bin/python -m pip cache purge
RCT-Reviewer/.venv/bin/python -m pip install "PyMuPDF==1.28.2" --no-cache-dir
```
Verify with `RCT-Reviewer/.venv/bin/python -c "import fitz; print('PyMuPDF works!')"`, then re-run `evaluate.py`.