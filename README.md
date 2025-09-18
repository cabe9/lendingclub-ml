# LendingClub-ML: Loan Repayment Prediction with TensorFlow and OpenAI-Enhanced Features

This repository contains a machine learning project that predicts whether borrowers will repay their loans using the publicly available LendingClub dataset. It showcases **OpenAI-powered feature engineering** — transforming messy text fields like job titles into standardized categories — alongside a TensorFlow/Keras classification model.

---

## Key Features

- **OpenAI Feature Engineering**  
  Uses OpenAI’s API (optionally) to standardize job titles and other free-text fields. Results are cached for reproducibility.

- **TensorFlow/Keras Model**  
  Trains and evaluates a neural network on engineered features to predict loan repayment status.

- **Clean Project Structure**  
  Includes `.gitignore`, MIT license, and a small sample dataset for quick testing.

- **Optional KaggleHub Integration**  
  Download the full LendingClub dataset from Kaggle with one command.

---

## Project Structure

Lendingclub-ml/

data/

data_dictionary.csv — rewritten data dictionary

samples/ — small sample CSV for demo runs

notebooks/

Lendingclub-ml.ipynb — main notebook

src/

preprocessing.py — non-API feature engineering

fe_openai.py — OpenAI-based feature engineering

model.py — TensorFlow model builder

LICENSE

README.md

.gitignore




---

## Getting Started

### 1. Clone This Repository

bash
git clone https://github.com/cabe9/lendingclub-ml.git
cd lendingclub-ml

2. Install Dependencies

Create and activate a virtual environment, then install:

pip install -r requirements.txt

3. Download Full Dataset from KaggleHub

If you have a Kaggle account and API token:

import kagglehub
path = kagglehub.dataset_download("wordsforthewise/lending-club")
print("Path to dataset files:", path)


Adjust your notebook to load the full CSV at that path.

4. Run the Notebook

Open the Jupyter notebook:

jupyter notebook notebooks/Lendingclub-ml.ipynb

By default it loads the small sample dataset in data/samples/ for quick demos.

5. Enable OpenAI Feature Engineering

Set your OpenAI API key as an environment variable:

export OPENAI_API_KEY="your_api_key_here"

The notebook will then automatically cache labeled job titles in data/cache/.

Results

Visualize loan status class balance.

Engineer features for job titles and ZIP codes.

Train a TensorFlow/Keras model and evaluate performance.

(Update this section with your metrics and charts once finalized.)

Reproducibility Notes

Caching: All OpenAI-labeled features are cached in data/cache/ (excluded from Git).

No Secrets: API keys are never stored in the repository.

Sample Data: A small sample CSV is included so the notebook runs offline.

License

This project is licensed under the MIT License — see the LICENSE


