from functools import partial
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from .data import FEATURES, ZERO_MISSING, validate_features


class Preprocessor:
    """Fit only on a training fold. Target is used solely for optional feature selection."""

    def __init__(self, cap_outliers=False, feature_count=8, seed=42, zero_missing=True):
        self.cap_outliers = cap_outliers
        self.feature_count = feature_count
        self.seed = seed
        self.zero_missing = zero_missing

    def clean(self, x):
        x = validate_features(x).copy()
        if self.zero_missing:
            x[ZERO_MISSING] = x[ZERO_MISSING].replace(0, np.nan)
        return x

    def fit(self, x, y):
        clean = self.clean(x)
        self.medians_ = clean.median()
        if self.medians_.isna().any():
            raise ValueError("A training predictor is entirely missing; cannot learn its median.")
        filled = clean.fillna(self.medians_)
        q1, q3 = filled.quantile(.25), filled.quantile(.75)
        self.lower_ = q1 - 1.5 * (q3 - q1)
        self.upper_ = q3 + 1.5 * (q3 - q1)
        if self.cap_outliers:
            filled = filled.clip(self.lower_, self.upper_, axis=1)
        self.selected_ = FEATURES.copy()
        if self.feature_count != len(FEATURES):
            selector = SelectKBest(partial(mutual_info_classif, random_state=self.seed), k=self.feature_count)
            selector.fit(filled, y)
            self.selected_ = filled.columns[selector.get_support()].tolist()
        self.training_rows_ = len(x)
        return self

    def transform(self, x):
        filled = self.clean(x).fillna(self.medians_)
        if self.cap_outliers:
            filled = filled.clip(self.lower_, self.upper_, axis=1)
        return filled.loc[:, self.selected_]


def resample_training(x, y, enabled, seed):
    if not enabled:
        return x, np.asarray(y)
    counts = np.bincount(np.asarray(y, dtype=int))
    if counts.min() < 2:
        raise ValueError("SMOTE requires at least two training observations in each class.")
    scaler = StandardScaler().fit(x)
    sampled, labels = SMOTE(random_state=seed, k_neighbors=min(5, int(counts.min())-1)).fit_resample(scaler.transform(x), y)
    # Return original units for interpretable J48 thresholds. Synthetic count values
    # remain fractional during fitting, while actual patient inputs require integers.
    return pd.DataFrame(scaler.inverse_transform(sampled), columns=x.columns), np.asarray(labels)
