"""
src/models.py
-------------
Model definitions and training logic.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# Models that require feature scaling (distance or gradient-based)
SCALED_MODELS = {'Logistic Regression', 'KNN', 'SVM', 'MLP'}


def get_models():
    """
    Return the seven classifiers used in the study.
    Mirrors the set evaluated in the reference paper:
    Chelas, Routis & Roussaki (MDPI Computers, 2024).
    """
    models = {
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'MLP': MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(64, 32)),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(random_state=42, probability=True, kernel='rbf'),
    }
    return models


def train_all(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train all models and return results and predictions dicts.

    - scaled data is used for Logistic Regression, KNN, SVM, MLP
    - unscaled data is used for Random Forest, Decision Tree, Gaussian Naive Bayes
    - scaler must be fitted on X_train only (no data leakage)
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)
    from sklearn.metrics import fbeta_score, matthews_corrcoef

    results = {}
    predictions = {}

    for name, model in models.items():
        # scaled data for Logistic Regression and KNN
        if name in SCALED_MODELS:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'F2': fbeta_score(y_test, y_pred, beta=2, zero_division=0),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_prob)
        }
        predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob, 'y_test': y_test}

    return results, predictions
