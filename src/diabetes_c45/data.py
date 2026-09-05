import hashlib
import numpy as np
import pandas as pd
from .paths import DATA

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
UNITS = dict(zip(FEATURES, ["count", "mg/dL (2-hour OGTT)", "mm Hg (diastolic)", "mm", "micro-units/mL (2-hour)", "kg/m²", "pedigree function", "years"]))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_features(frame):
    if set(frame.columns) != set(FEATURES) or len(frame.columns) != len(FEATURES):
        raise ValueError("Provide exactly the eight named predictors; Outcome and IDs are not inputs.")
    result = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite values are not supported.")
    if (result < 0).any().any():
        raise ValueError("Clinical inputs cannot be negative.")
    for col in ["Pregnancies", "Age"]:
        values = result[col].dropna()
        if not np.allclose(values, np.round(values), rtol=0, atol=0):
            raise ValueError(f"{col} must be a whole number.")
    if (result.Age.dropna() < 21).any():
        raise ValueError("This prototype supports the study population aged 21 or older.")
    return result


def load_data(path=DATA):
    frame = pd.read_csv(path)
    if frame.columns.tolist() != FEATURES + ["Outcome"]:
        raise ValueError("Dataset must have the verified Pima schema, including Outcome.")
    if frame.duplicated().any():
        raise ValueError("Duplicate records would contaminate cross-validation splits.")
    if frame.Outcome.isna().any() or set(frame.Outcome.unique()) != {0, 1}:
        raise ValueError("Both original binary classes 0 and 1 are required; missing labels are forbidden.")
    x = validate_features(frame.drop(columns="Outcome"))
    return x, frame.Outcome.astype(int)


def audit(x, y):
    return {
        "records": len(x), "predictors": list(x.columns),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "zero_missing_counts": {c: int(x[c].eq(0).sum()) for c in ZERO_MISSING},
        "explicit_missing": {c: int(x[c].isna().sum()) for c in FEATURES},
        "summary": x.describe().to_dict(),
    }
