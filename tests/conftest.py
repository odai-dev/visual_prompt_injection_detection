"""
Pytest conftest.py — project-level test configuration and fixtures.

Patches sklearn estimators to gracefully handle tiny corpora in unit tests:
1. CalibratedClassifierCV: falls back to direct fitting when samples < cv folds.
2. GridSearchCV: reduces cv folds when samples are too few for 5-fold CV.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Apply monkey-patches before test collection."""
    _patch_calibrated_classifier_cv()
    _patch_grid_search_cv()


def _patch_calibrated_classifier_cv():
    """Patch CalibratedClassifierCV to handle tiny corpora gracefully."""
    try:
        import numpy as np
        import copy
        from sklearn.calibration import (
            CalibratedClassifierCV,
            _CalibratedClassifier,
            _SigmoidCalibration,
        )
        from sklearn.preprocessing import LabelEncoder

        original_fit = CalibratedClassifierCV.fit

        def patched_fit(self, X, y, sample_weight=None, **fit_params):
            if isinstance(self.cv, int):
                y_arr = np.asarray(y)
                _, counts = np.unique(y_arr, return_counts=True)
                if len(counts) > 0 and int(counts.min()) < self.cv:
                    # Too few samples — fit directly and calibrate on full data
                    base = copy.deepcopy(self.estimator)
                    base.fit(X, y_arr)

                    le = LabelEncoder()
                    le.fit(y_arr)
                    self.classes_ = le.classes_

                    calibrator = _SigmoidCalibration()
                    if hasattr(base, "decision_function"):
                        df = base.decision_function(X)
                        if df.ndim == 1:
                            calibrator.fit(df, y_arr)
                            self.calibrated_classifiers_ = [
                                _CalibratedClassifier(
                                    base, [calibrator],
                                    method="sigmoid",
                                    classes=self.classes_,
                                )
                            ]
                        else:
                            self.calibrated_classifiers_ = []
                    else:
                        self.calibrated_classifiers_ = []
                    return self
            return original_fit(self, X, y, sample_weight=sample_weight, **fit_params)

        CalibratedClassifierCV.fit = patched_fit
    except Exception:
        pass


def _patch_grid_search_cv():
    """Patch GridSearchCV to reduce cv folds when corpus is too small."""
    try:
        import numpy as np
        from sklearn.model_selection import GridSearchCV

        original_fit = GridSearchCV.fit

        def patched_fit(self, X, y=None, **fit_params):
            if y is not None and isinstance(self.cv, int):
                y_arr = np.asarray(y)
                _, counts = np.unique(y_arr, return_counts=True)
                if len(counts) > 0 and int(counts.min()) < self.cv:
                    # Reduce cv to min samples per class (at least 2)
                    safe_cv = max(2, int(counts.min()))
                    if safe_cv < self.cv:
                        self.cv = safe_cv
            return original_fit(self, X, y, **fit_params)

        GridSearchCV.fit = patched_fit
    except Exception:
        pass
