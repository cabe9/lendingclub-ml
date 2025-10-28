# LendingClub-ML: Loan Repayment Prediction with TensorFlow and LLM-Enhanced Features

This repository contains a machine learning project that predicts whether borrowers will repay their loans using the publicly available LendingClub dataset. It showcases **LLM-powered feature engineering** — transforming messy text fields like job titles into standardized occupational categories — alongside baseline tabular models (logistic regression) and a TensorFlow/Keras neural network.

The goal is not just "can we predict default?" but: **does an LLM-generated feature materially improve future default prediction, and is that economically worth it?**

---

## Key Features

### LLM Feature Engineering (Core Signal)
We normalize borrower job titles into ~40 standardized occupation categories such as "Commercial Driving", "Healthcare Practitioner", "Executive Management", etc. That derived categorical feature is called `job_category_40`.

Pipeline details:
- Clean and normalize the raw free-text job title (`emp_title`) by lowercasing, stripping punctuation, collapsing aliases like `vp` → `vice president`, `rn` → `registered nurse`, etc.
- Send unique normalized titles to an LLM with a fixed allowed category list.
- Enforce strict JSON output, validate, and cache the mapping to disk.

That mapping is merged back into the loan dataset as a model input feature.

### Baseline Tabular Models
We train and evaluate:
- Logistic Regression
- A small fully connected TensorFlow/Keras model (2 hidden layers + dropout)

Both are evaluated on an **out-of-time / temporal split**: older loans for training, newer loans for testing. This simulates "train on history, predict on future vintages," which is closer to how real underwriting works than a random split.

### Clean Project Structure
- Project root includes MIT license and `.gitignore`
- Notebook documents the full pipeline end to end
- Processed parquet and cached mappings let you reproduce results without paying for the LLM again

### Kaggle Integration
The project can pull the LendingClub dataset from Kaggle. The repo also includes a small sample dataset so you can run the notebook without downloading the full 1M+ rows immediately.

---

## Project Structure

```text
Lendingclub-ml/
  data/
    data_dictionary.csv        # rewritten data dictionary
    samples/                   # small sample CSV for demo runs
    processed/                 # parquet / sample outputs (gitignored if large)
    cache/                     # cached job-title→category mapping
  Notebooks/
    Lendingclub-ml.ipynb       # main notebook with full pipeline
  src/
    preprocessing.py           # non-LLM feature engineering
    fe_openai.py               # LLM-based job title categorization
    model.py                   # model definition / training helpers
  LICENSE
  README.md
  .gitignore
```

---

## Getting Started

### 1) Clone this repository

```bash
git clone https://github.com/cabe9/lendingclub-ml.git
cd lendingclub-ml
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) (Optional) Download full dataset with KaggleHub

If you have a Kaggle account and API token:

```python
import kagglehub
path = kagglehub.dataset_download("wordsforthewise/lending-club")
print("Path to dataset files:", path)
```

You can then point the notebook at the large `accepted_*.csv` file under that path. The repo also ships a tiny sample under `data/samples/` so you can run quickly without downloading the full dataset.

### 4) Run the main notebook

```bash
jupyter notebook "Notebooks/Lendingclub-ml.ipynb"
```

### 5) Enable LLM-based feature engineering (optional at runtime, core to the method)

To regenerate the occupation category feature yourself, you’ll need an API key:

**macOS / Linux:**

```bash
export OPENAI_API_KEY="your_api_key_here"
```

**Windows (PowerShell):**

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

If you don’t set an API key, the notebook can fall back to the cached job-title mapping that’s already been generated. That lets you reproduce results without incurring new LLM cost.

---

## Results (Temporal Generalization)

We train on older loans and test on newer loans (temporal split). This is meant to approximate real deployment conditions: "you built a model today, can it predict next year’s repayment behavior."

Below are out-of-time test results:

| Model                 | Features Used                          | Test AUC |
|-----------------------|----------------------------------------|----------|
| Logistic Regression   | No occupation feature                  | ~0.669   |
| Logistic Regression   | + LLM `job_category_40`                | ~0.701   |
| TensorFlow MLP (3 ep) | + LLM `job_category_40`                | ~0.704   |

Interpretation:
- Adding a single LLM-derived categorical feature (standardized occupation) improves out-of-time AUC by roughly +0.03 to +0.035.
- Logistic regression and a small neural net land in a similar range once the LLM feature is present, which is useful: you don’t necessarily need a deep model to benefit from the enriched feature.

In credit modeling, an AUC lift of a few hundredths on future vintages is economically meaningful because it can support either higher approval rates at constant risk or lower expected losses at constant approval rates.

---

## LLM Cost / Practical Notes

Generating the occupation category feature (`job_category_40`) required classifying ~290k unique normalized job titles into one of ~40 allowed buckets.

This labeling step was done with `gpt-5-mini` in batched calls (with strict JSON validation and caching), and cost on the order of **$20 USD**.

Why this is still practical in a real setting:

- This is a **one-time enrichment** pass over historical data. After the mapping is created and cached, future model training and inference use only tabular features; no live LLM calls are required.
- In production, you would typically:
  - Cache the category for every job title string you’ve already seen.
  - For new applications, only classify titles you’ve never seen before.
  - Optionally distill a lightweight local classifier from this mapping so you never have to call the LLM online.

Open questions / future work (contributions welcome):
- Can a cheaper/smaller model (for example, a `...-nano` tier) produce categories that are "good enough" for similar AUC lift, at lower cost?
- Can we learn a small local text classifier from the cached mapping and remove the LLM from the loop entirely at inference time?
- Can we reduce prompt overhead and batching overhead to push the labeling cost even lower?

This approach is commercially relevant: a one-time ~$20 categorization pass produced a feature that improved out-of-time AUC by ~0.03+. In many lending and portfolio risk settings, that level of improvement is financially material.

---

## Reproducibility Notes

- **Caching:** The job-title → occupation mapping produced by the LLM is cached under `data/cache/`. This prevents re-spending money on repeated titles.
- **Processed parquet:** A preprocessed dataset (`df_model_ready.parquet`) is saved under `data/processed/` so you can jump straight into modeling.
- **Secrets:** API keys are environment variables; they are not stored in the repo.
- **Temporal split:** The notebook includes both a traditional random split and an explicit temporal split to show the difference.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- LendingClub dataset via Kaggle
- Occupation category ontology and mapping pipeline defined and implemented here
- TensorFlow / scikit-learn used for modeling and evaluation
