# LendingClub-ML

## Problem

Can LLM-based categorization of messy free-text job titles improve performance in a tabular loan default prediction task?

## Method

- Built a tabular ML pipeline on the LendingClub dataset using mostly origination-time features
- Normalized and mapped job titles into about 40 categories using an LLM
- Compared the LLM-derived category against simpler job-title baselines, including TF-IDF on normalized titles
- Ran the representation comparison under the same temporal split, same linear model, and same preprocessing so the only change was how job-title information was represented
- Compared logistic regression and a small neural network
- Evaluated using both random splits and a more realistic temporal split

## Results

- Logistic Regression: test AUC `~0.701`
- Small DNN: test AUC `~0.704`
- LightGBM with minimal tuning: test AUC `~0.710`
- Removing LendingClub pricing features reduced AUC to `~0.669`
- LLM-derived job category feature added a small but consistent gain of about `+0.003` to `+0.004` AUC
- In a controlled temporal comparison, a strong lexical baseline using TF-IDF on normalized job titles outperformed the coarser LLM category feature
- Adding `job_category_40` on top of TF-IDF changed test AUC by only about `+0.0001`, which suggests TF-IDF already captures most of the usable job-title signal in this setup

This repository is a notebook-first, end-to-end machine learning case study designed to explore feature engineering and evaluation choices in tabular models.

## Final Results

### Controlled Representation Comparison

Same temporal split, same linear model, and same preprocessing. The only change here is how job-title information is represented.

| Setup | Valid AUC | Test AUC | Takeaway |
|---|---:|---:|---|
| No job-title feature | 0.707 | 0.694 | baseline |
| LLM category (`job_category_40`) | 0.710 | 0.698 | small gain over no title |
| TF-IDF (`emp_title_clean`) | 0.715 | 0.704 | strongest job-title representation |
| TF-IDF + LLM | 0.715 | 0.704 | negligible extra lift |

### Other Temporal Benchmarks

| Model | Valid AUC | Test AUC | Takeaway |
|---|---:|---:|---|
| Logistic Regression (`job_category_40`) | 0.712 | 0.700 | simple structured baseline |
| Small DNN (`job_category_40`) | 0.711 | 0.704 | similar to logistic |
| Logistic Regression (no pricing features) | 0.666 | 0.669 | pricing and grade features dominate |
| LightGBM (`job_category_40`, minimal tuning) | 0.717 | 0.710 | strongest overall baseline tested |

Coefficient inspection of the linear TF-IDF model suggests that most of the predictive signal comes from credit-grade and underwriting variables such as `sub_grade`, `term_months`, `fico_mid`, and `annual_inc`, while job-title terms add smaller secondary effects.

## What This Project Shows

- Exploring whether LLM-based categorization of free-text job titles adds measurable value in a tabular ML problem
- Comparing LLM-based semantic grouping against a strong lexical baseline
- Preparing a real public dataset for modeling
- Thinking about target leakage and using mostly origination-time features
- Basic EDA and feature engineering for tabular data
- Comparing logistic regression with a small TensorFlow/Keras model
- Evaluating both random splits and a more realistic temporal split

## LLM Feature Engineering (What I Actually Did)

The raw `emp_title` field in the LendingClub dataset is messy free text, with values like `"VP Sales"`, `"rn"`, and `"owner/operator"`.

In this project, I:

- Normalized job titles by lowercasing, removing punctuation, and expanding common abbreviations
- Sent unique normalized titles to an LLM with a fixed list of about 40 occupation categories
- Enforced structured output and cached the mapping locally
- Added the resulting feature as `job_category_40`

This is a one-time preprocessing step. Once the mapping is created, no further LLM calls are needed to train or run the model.

## Project Overview

The goal is to predict whether a loan ends up `Fully Paid` or `Charged Off`.

The rough workflow in the notebook:

1. Download and explore the LendingClub dataset
2. Select a smaller set of mostly origination-time features
3. Build a modeling dataframe with numeric and categorical features
4. Optionally normalize and group job titles using the OpenAI API
5. Train a few baseline models
6. Re-evaluate using a temporal split to better reflect real-world performance

## Repository Structure

```text
Lendingclub-ml/
  Notebooks/
    Lendingclub-ml.ipynb      # main notebook
  data/
    data_dictionary.csv
    cache/                    # local, not committed
    processed/                # local, not committed
  README.md
  LICENSE
  requirements.txt
```

Notes:

- Some systems are case-sensitive, so `Notebooks` must be capitalized
- `data/cache` and `data/processed` are local working directories
- This is a notebook-first project; there is no full `src/` package yet

## Results (Approximate)

From the temporal split section:

- Logistic Regression: val AUC `~0.711`, test AUC `~0.701`
- Small DNN: test AUC `~0.704`
- LightGBM with minimal tuning: val AUC `~0.717`, test AUC `~0.710`
- Logistic Regression without LendingClub pricing features: test AUC `~0.669`

Results may vary depending on environment, random seeds, and whether job title processing is used.

- Adding the LLM-derived `job_category_40` feature produced a small but consistent improvement in the stricter temporal logistic-regression rerun, with about `+0.003` validation AUC and `+0.004` test AUC
- The gain was modest, so the feature was helpful but not transformative in this setting
- In a controlled representation comparison using the same temporal split, same linear model, and same preprocessing, TF-IDF on normalized job titles performed better than the coarser LLM category feature. That suggests TF-IDF retains more predictive signal, while the LLM feature provides a more compact, standardized, and interpretable representation.
- Combining TF-IDF with `job_category_40` produced only a negligible additional lift of about `+0.0001` test AUC, so TF-IDF appears to capture most of the usable job-title signal in this setup

The LLM-based feature required a one-time labeling pass over unique job titles, cached locally, so it does not require LLM calls during model training or inference.

## Setup

Clone the repo:

```bash
git clone https://github.com/cabe9/lendingclub-ml.git
cd lendingclub-ml
```

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`conda` or another environment manager is fine too.

Open the notebook:

```bash
jupyter notebook Notebooks/Lendingclub-ml.ipynb
```

## Running the Project

### Dataset

The notebook downloads the dataset using `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("wordsforthewise/lending-club")
print(path)
```

You’ll need Kaggle set up locally for this to work.

### Optional OpenAI Features

There is an optional section that maps job titles to categories using the OpenAI API.

If you want to use it, set your API key before starting Jupyter.

macOS/Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows PowerShell:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

Results are cached in `data/cache` so they don’t need to be recomputed.

### Paths

The notebook mostly uses repo-relative paths, but supports optional overrides:

- `LENDINGCLUB_ML_ROOT`
- `LENDINGCLUB_DATASET_DIR`

You likely won’t need these unless something breaks.

## Reproducibility Notes

- The dataset is not included
- Processed files are not committed
- The notebook is still somewhat exploratory in places
- The temporal split section is the most reliable evaluation

## What I Learned

- Temporal splits give a stricter and more realistic estimate than random splits on this dataset.
- Simple linear baselines are strong, especially when paired with TF-IDF on short text fields.
- LLM-derived categories make messy job titles more standardized and interpretable, but they compress some of the lexical signal that TF-IDF keeps.
- LendingClub pricing and grade features dominate performance in this setup.
- A standard tree-based baseline like LightGBM is competitive with minimal tuning.

## Limitations / Future Work

- The LLM-based job title categorization produced only a modest gain in this setup. As LLMs continue to improve in consistency and cost efficiency, similar approaches to semantic feature engineering may become more impactful or easier to deploy at scale.
- Compare job-title representations more systematically, including stronger text baselines and hybrid approaches that combine raw text with LLM-derived groupings
- Move more logic into reusable Python modules
- Improve experiment tracking and comparisons
- Expand the feature-importance and interpretability analysis beyond the linear-model coefficient view
- Introduce a more explicit train/val/test structure earlier
- Compare LightGBM with other tree-based tabular models such as CatBoost or XGBoost

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- LendingClub dataset via [Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)
- Built as a personal machine learning project
