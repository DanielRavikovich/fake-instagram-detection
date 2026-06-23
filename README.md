# Detecting Fake Instagram Accounts Using Machine Learning

**Student:** Daniel Ravikovich  
**ID:** 324277060  
**Course:** Data Science in Cyber  
**Submitted:** July 2026

---

## Project Description

This project presents a critical reproduction study of the paper:

> Chelas, S., Routis, G., Roussaki, I. "Detection of Fake Instagram Accounts via Machine Learning Techniques." *Computers* 2024, 13(11), 296. DOI: 10.3390/computers13110296

The goal is not only to reproduce the proposed solution, but to critically evaluate whether the authors' claims are supported by the data and experiments.

Two public datasets are merged into a unified corpus of 1,855 records (after deduplication). Three classification models are trained and compared: Logistic Regression, Random Forest, and K-Nearest Neighbors. Random Forest achieves the best performance: Accuracy = 94.9%, F1 = 0.9140, AUC-ROC = 0.9836.

---

## Repository Contents

| File | Description |
|---|---|
| `Using_Data_Science_Methods_in_Cybersecurity_Project_Daniel_Ravikovich.ipynb` | Full Python notebook — data loading, EDA, feature engineering, model training, evaluation, and error analysis |
| `Project_final.pdf` | PDF report covering all 8 required sections |
| `README.md` | This file |

---

## Selected Article

**Title:** Detection of Fake Instagram Accounts via Machine Learning Techniques  
**Authors:** Chelas, Routis, Roussaki — National Technical University of Athens  
**Published:** MDPI Computers, November 2024  
**Link:** https://www.mdpi.com/2073-431X/13/11/296

---

## Original GitHub Repository

InstaFake Dataset (used in this project):  
https://github.com/fcakyon/instafake-dataset

---

## Dataset Sources

| Dataset | Source | Format |
|---|---|---|
| InstaFake Dataset | https://github.com/fcakyon/instafake-dataset/tree/master/data/fake-v1.0 | JSON |
| Instagram Fake Spammer Genuine Accounts | https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts | CSV |

---

## Execution Instructions

### Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Data Setup

Download the following four files and place them in the same directory as the notebook:

- `fakeAccountData.json` — from InstaFake GitHub
- `realAccountData.json` — from InstaFake GitHub
- `train.csv` — from Kaggle
- `test.csv` — from Kaggle

### Running the Notebook

Open the notebook in Google Colab or Jupyter and run all cells in order:

```
Runtime → Run all
```

All random operations use `random_state=42` for reproducibility.

---

## Key Results

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.9164 | 0.8531 | 0.9661 |
| **Random Forest** | **0.9488** | **0.9140** | **0.9836** |
| KNN (k=5) | 0.9137 | 0.8532 | 0.9594 |
