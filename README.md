# LendingClub-ML: Loan Repayment Prediction with TensorFlow and OpenAI‑Enhanced Features

This repository contains a machine learning project that predicts whether borrowers will repay their loans using the publicly available LendingClub dataset. It showcases **OpenAI‑powered feature engineering**—transforming messy text fields like job titles into standardized categories—alongside a TensorFlow/Keras classification model.

---

## Key Features

* **OpenAI Feature Engineering**
  Optionally standardizes job titles via OpenAI’s API. Results are cached for reproducibility.

* **TensorFlow/Keras Model**
  Trains and evaluates a neural network on engineered features to predict loan repayment status.

* **Clean Project Structure**
  Includes `.gitignore`, MIT license, and a small sample dataset for quick testing.

* **Optional KaggleHub Integration**
  Download the full LendingClub dataset from Kaggle with one command.

---

## Project Structure

```text
Lendingclub-ml/
  data/
    data_dictionary.csv        # rewritten data dictionary
    samples/                   # small sample CSV for demo runs
  notebooks/
    Lendingclub-ml.ipynb       # main notebook
  src/
    preprocessing.py           # non-API feature engineering
    fe_openai.py               # OpenAI-based feature engineering
    model.py                   # TensorFlow model builder
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

Adjust your notebook to load the full CSV at that path. The repo also includes a tiny sample in `data/samples/` for quick runs.

### 4) Run the notebook

```bash
jupyter notebook notebooks/Lendingclub-ml.ipynb
```

### 5) Enable OpenAI feature engineering (optional)

**macOS/Linux:**

```bash
export OPENAI_API_KEY="your_api_key_here"
```

**Windows (PowerShell):**

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

The notebook caches labeled job titles in `data/cache/` (excluded from Git).

---

## Results

* Visualize loan status class balance
* Engineer features for job titles and ZIP codes
* Train a TensorFlow/Keras model and evaluate performance

*Update this section with your metrics and charts once finalized.*

---

## Reproducibility Notes

* **Caching:** OpenAI‑labeled features cached in `data/cache/` (gitignored)
* **No secrets:** API keys are never stored in the repository
* **Sample data:** Small CSV in `data/samples/` so the notebook runs offline

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

* LendingClub dataset via [Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)
* Inspired by coursework but implemented and extended with OpenAI feature engineering
