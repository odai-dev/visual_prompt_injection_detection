"""
Supervised Learning Module.

Benchmarks Linear SVM (LinearSVC) vs. Random Forest (RandomForestClassifier)
with 5-fold cross-validation GridSearchCV on the training set, evaluates on
both in-distribution (test.csv) and out-of-distribution (cyberseceval3_test.csv)
held-out sets, and persists the best model artefacts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.nlp import build_vectorizer, clean_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "best_supervised_model.joblib"
METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"

TEXT_COLUMN = "cleaned_text"
LABEL_COLUMN = "label"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train_path: Path = PROCESSED_DIR / "train.csv",
    test_path: Path = PROCESSED_DIR / "test.csv",
    cyber_path: Optional[Path] = PROCESSED_DIR / "cyberseceval3_test.csv",
    models_dir: Path = MODELS_DIR,
    verbose: bool = True,
) -> Dict:
    """
    Train supervised models, evaluate, and save artefacts.

    Returns
    -------
    dict
        Full evaluation metrics for both test sets and model metadata.
    """
    for d in (models_dir, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Use cleaned_text if available, otherwise fall back to raw text
    text_col = TEXT_COLUMN if TEXT_COLUMN in train_df.columns else "text"
    train_texts = train_df[text_col].fillna("[EMPTY_IMAGE]").tolist()
    train_labels = train_df[LABEL_COLUMN].tolist()
    test_texts = test_df[text_col].fillna("[EMPTY_IMAGE]").tolist()
    test_labels = test_df[LABEL_COLUMN].tolist()

    if verbose:
        logger.info("Training set: %d samples.", len(train_texts))
        logger.info("Test set: %d samples.", len(test_texts))

    # Fit TF-IDF vectorizer on training data
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # --- Benchmark models ---
    results = {}

    svm_model, svm_metrics = _train_svm(X_train, train_labels, X_test, test_labels, verbose)
    results["linear_svm"] = svm_metrics

    rf_model, rf_metrics = _train_random_forest(X_train, train_labels, X_test, test_labels, verbose)
    results["random_forest"] = rf_metrics

    # --- Select best model by F1 score (recall-aware) ---
    best_name, best_model = (
        ("linear_svm", svm_model)
        if svm_metrics["f1"] >= rf_metrics["f1"]
        else ("random_forest", rf_model)
    )
    results["best_model"] = best_name
    if verbose:
        logger.info("Best model: %s (F1=%.4f)", best_name, results[best_name]["f1"])

    # Persist artefacts
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(best_model, models_dir / "best_supervised_model.joblib")

    # --- In-distribution prediction CSV ---
    _save_prediction_csv(
        best_model, vectorizer, test_df, test_texts, test_labels,
        PREDICTIONS_DIR / "synthetic_test_predictions.csv",
    )

    # --- Out-of-distribution: CyberSecEval 3 ---
    cyber_metrics = {}
    if cyber_path and cyber_path.exists():
        cyber_df = pd.read_csv(cyber_path)
        cyber_text_col = TEXT_COLUMN if TEXT_COLUMN in cyber_df.columns else "text"
        cyber_texts = cyber_df[cyber_text_col].fillna("[EMPTY_IMAGE]").tolist()
        cyber_labels = cyber_df[LABEL_COLUMN].tolist() if LABEL_COLUMN in cyber_df.columns else []

        if cyber_labels:
            X_cyber = vectorizer.transform(cyber_texts)
            cyber_metrics = _compute_metrics(best_model, X_cyber, cyber_labels)
            results["cyberseceval3"] = cyber_metrics
            if verbose:
                logger.info(
                    "CyberSecEval 3 – Accuracy=%.4f, Recall=%.4f, F1=%.4f",
                    cyber_metrics["accuracy"], cyber_metrics["recall"], cyber_metrics["f1"],
                )
            _save_prediction_csv(
                best_model, vectorizer, cyber_df, cyber_texts, cyber_labels,
                PREDICTIONS_DIR / "cyberseceval3_predictions.csv",
            )

    # Persist metrics JSON
    _save_metrics_json(results)

    return results


def load_model(
    model_path: Path = MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
) -> Tuple:
    """Load persisted model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_text(text: str, model, vectorizer) -> Dict:
    """
    Classify a single text string.

    Returns
    -------
    dict with keys: predicted_label (int), confidence (float), verdict (str)
    """
    cleaned = clean_text(text) if text.strip() else "[EMPTY_IMAGE]"
    X = vectorizer.transform([cleaned])
    label = int(model.predict(X)[0])
    try:
        prob = model.predict_proba(X)[0]
        confidence = float(prob[label])
    except AttributeError:
        confidence = 1.0  # LinearSVC without calibration
    verdict = "MALICIOUS / INJECTION" if label == 1 else "SAFE"
    return {"predicted_label": label, "confidence": confidence, "verdict": verdict}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _train_svm(X_train, y_train, X_test, y_test, verbose: bool):
    """Train and evaluate a calibrated LinearSVC."""
    import numpy as np  # noqa: PLC0415
    y_arr = np.asarray(y_train)
    min_class_count = int(np.min(np.unique(y_arr, return_counts=True)[1])) if len(y_arr) > 0 else 5
    # For nested CV, GridSearchCV uses (k-1)/k of data. The inner CalibratedClassifierCV
    # must have fewer folds than the number of samples in the training split.
    outer_cv = max(2, min(5, min_class_count))
    inner_cv = max(2, min(5, min_class_count - 1)) if min_class_count > 2 else 2

    if min_class_count < 3:
        # Fallback to uncalibrated or purely prefit calibration if dataset is impossibly small
        base = CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=2000), cv=2)
    else:
        base = CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=2000), cv=inner_cv)

    param_grid = {"estimator__C": [0.1, 1.0, 10.0]}
    gs = GridSearchCV(base, param_grid, cv=outer_cv, scoring="f1", n_jobs=-1, error_score="raise")
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    metrics = _compute_metrics(best, X_test, y_test)
    if verbose:
        logger.info(
            "LinearSVC – best C=%.2f | Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f AUC=%.4f",
            gs.best_params_.get("estimator__C", -1),
            metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["roc_auc"],
        )
    return best, metrics


def _train_random_forest(X_train, y_train, X_test, y_test, verbose: bool):
    """Train and evaluate a Random Forest classifier."""
    import numpy as np  # noqa: PLC0415
    y_arr = np.asarray(y_train)
    min_class_count = int(np.min(np.unique(y_arr, return_counts=True)[1])) if len(y_arr) > 0 else 5
    cv_folds = max(2, min(5, min_class_count))

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(rf, param_grid, cv=cv_folds, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    metrics = _compute_metrics(best, X_test, y_test)
    if verbose:
        logger.info(
            "RandomForest – best params=%s | Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f AUC=%.4f",
            gs.best_params_,
            metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["roc_auc"],
        )
    return best, metrics


def _compute_metrics(model, X, y_true) -> Dict:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y_true, y_prob))
    except (AttributeError, ValueError):
        auc = float(roc_auc_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def _save_prediction_csv(model, vectorizer, df: pd.DataFrame, texts, labels, out_path: Path):
    """Export per-image prediction records to a CSV file."""
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
    except AttributeError:
        probs = preds.astype(float)

    records = []
    for i, (pred, prob, gt) in enumerate(zip(preds, probs, labels)):
        row = {
            "id": df.iloc[i].get("filename", df.iloc[i].get("id", i)),
            "ground_truth_label": int(gt),
            "predicted_label": int(pred),
            "confidence": round(float(prob), 4),
            "verdict": "MALICIOUS / INJECTION" if pred == 1 else "SAFE",
        }
        records.append(row)

    pd.DataFrame(records).to_csv(out_path, index=False)
    logger.info("Saved prediction CSV → %s (%d rows).", out_path.name, len(records))


def _save_metrics_json(results: Dict):
    """Persist evaluation metrics to JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation metrics saved → %s", METRICS_PATH.name)


# ---------------------------------------------------------------------------
# Test-suite API aliases
# ---------------------------------------------------------------------------

def train_supervised_benchmark(
    train_csv: Path,
    models_dir: Path,
    test_csv: Optional[Path] = None,
    verbose: bool = False,
) -> Dict:
    """
    Alias for :func:`train_and_evaluate` using positional-style args expected by tests.

    If test_csv is not provided, creates a small internal validation split from train_csv.
    Returns the full metrics dict with 'best_model', 'linear_svm', 'random_forest' keys.
    """
    import pandas as pd  # noqa: PLC0415
    from sklearn.model_selection import train_test_split as _split  # noqa: PLC0415

    # If no test CSV provided, create one from a split of the training data
    if test_csv is None or not Path(test_csv).exists():
        train_df = pd.read_csv(train_csv)
        if len(train_df) >= 4 and len(train_df["label"].unique()) >= 2:
            tr, te = _split(train_df, test_size=0.25, random_state=42, stratify=train_df["label"])
        else:
            tr, te = train_df, train_df  # fall back to using same data for tiny corpora
        test_csv_path = Path(train_csv).parent / "_temp_test.csv"
        te.to_csv(test_csv_path, index=False)
    else:
        test_csv_path = Path(test_csv)

    return train_and_evaluate(
        train_path=Path(train_csv),
        test_path=test_csv_path,
        cyber_path=None,
        models_dir=models_dir,
        verbose=verbose,
    )


def evaluate_supervised_model(
    model_path: Path,
    vectorizer_path: Path,
    test_csv: Path,
    verbose: bool = False,
) -> Dict:
    """
    Load a persisted model+vectorizer and evaluate on a given test CSV.

    Returns a metrics dict with accuracy, precision, recall, f1, roc_auc,
    predictions, and probabilities keys.
    """
    import pandas as pd  # noqa: PLC0415

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    test_df = pd.read_csv(test_csv)
    text_col = TEXT_COLUMN if TEXT_COLUMN in test_df.columns else "text"
    texts = test_df[text_col].fillna("[EMPTY_IMAGE]").tolist()
    labels = test_df[LABEL_COLUMN].tolist()

    X = vectorizer.transform(texts)
    y_pred = model.predict(X).tolist()
    try:
        y_prob = model.predict_proba(X)[:, 1].tolist()
    except AttributeError:
        y_prob = [float(p) for p in y_pred]

    # Handle single-class edge case for ROC-AUC
    unique = set(labels)
    if len(unique) < 2:
        from sklearn.metrics import accuracy_score  # noqa: PLC0415
        return {
            "accuracy": float(accuracy_score(labels, y_pred)),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "confusion_matrix": [[len(labels), 0], [0, 0]],
            "predictions": y_pred,
            "probabilities": y_prob,
        }

    metrics = _compute_metrics(model, X, labels)
    metrics["predictions"] = y_pred
    metrics["probabilities"] = y_prob
    return metrics
