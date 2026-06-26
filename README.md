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

Two public datasets are merged into a unified corpus of 1,855 records (after deduplication). Seven classification models are trained and compared, matching the set evaluated in the reference paper. Random Forest achieves the best performance: Accuracy = 94.9%, F1 = 0.9140, AUC-ROC = 0.9836.

---

## Repository Structure

```
fake-instagram-detection/
├── Using_Data_Science_Methods_in_Cybersecurity_Project_Daniel_Ravikovich.ipynb
├── Project.pdf
├── requirements.txt
├── download_data.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── results/
│   └── README.md
├── figures/
│   └── README.md
└── README.md
```

| File/Folder | Description |
|---|---|
| `*.ipynb` | Full notebook - data loading, EDA, feature engineering, model training, evaluation, error analysis |
| `Project_final.pdf` | PDF report covering all 8 required sections |
| `requirements.txt` | Pinned Python dependencies |
| `download_data.py` | Script to download the InstaFake dataset automatically |
| `src/preprocessing.py` | Data loading, merging, cleaning, and feature engineering |
| `src/models.py` | Model definitions and training logic |
| `src/evaluation.py` | Metrics, confusion matrices, ROC curves, and figure saving |
| `results/` | Generated CSV files with model evaluation metrics |
| `figures/` | Generated plots (distributions, confusion matrices, ROC curves, feature importance) |

---

## Selected Article
**Title:** Detection of Fake Instagram Accounts via Machine Learning Techniques  
**Authors:** Chelas, Routis, Roussaki - National Technical University of Athens  
**Published:** MDPI Computers, November 2024  
**Link:** https://www.mdpi.com/2073-431X/13/11/296

---

## Dataset Sources
| Dataset | Source | Format |
|---|---|---|
| InstaFake Dataset | https://github.com/fcakyon/instafake-dataset/tree/master/data/fake-v1.0 | JSON |
| Instagram Fake Spammer Genuine Accounts | https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts | CSV |

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
Run the download script to get the InstaFake JSON files automatically:
```bash
python download_data.py
```
Then follow the printed instructions to download `train.csv` and `test.csv` from Kaggle.  
Place all four data files in the project root directory.

### 3. Run the Notebook

**Google Colab:**  
The notebook clones this repository automatically in the setup cell, so `src/` is available without any manual steps. Simply open the notebook and run:
```
Runtime → Run all
```

**Local (Jupyter):**  
Clone the repository and run from the project root:
```bash
git clone https://github.com/DanielRavikovich/fake-instagram-detection
cd fake-instagram-detection
jupyter notebook
```

All random operations use `random_state=42` for reproducibility.

---

## Key Results

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Gaussian Naive Bayes | 0.4744 | 0.5278 | 0.9032 |
| Decision Tree | 0.9111 | 0.8546 | 0.9049 |
| Logistic Regression | 0.9164 | 0.8531 | 0.9661 |
| **Random Forest** | **0.9488** | **0.9140** | **0.9836** |
| MLP | 0.9326 | 0.8869 | 0.9825 |
| KNN (k=5) | 0.9137 | 0.8532 | 0.9594 |
| SVM | 0.9191 | 0.8598 | 0.9614 |
