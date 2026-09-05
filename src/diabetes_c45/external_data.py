"""Separate dataset contracts; never merges these populations or outcome labels."""
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from .paths import ROOT
from .data import sha256

DATASETS = {
    "sylhet": {"uci_id": 529, "title": "Sylhet symptom study", "target": "class",
               "label": "Original positive diabetes-study label", "negative": "Original negative study label",
               "scope": "Questionnaire participants at Sylhet Diabetes Hospital, Bangladesh; not a general-population screening sample."},
    "cdc": {"uci_id": 891, "title": "CDC health indicators", "target": "Diabetes_binary",
            "label": "Prediabetes or diabetes (UCI binary definition)", "negative": "No diabetes (UCI binary definition)",
            "scope": "US health-indicator survey data; age is a category, not years. Survey weights/design variables are not supplied in this processed file."},
}


def load_external(key):
    spec = DATASETS[key]
    folder = ROOT / f"data/external/uci_{spec['uci_id']}"
    metadata = json.loads((folder / "metadata.json").read_text())["data"]
    frame = pd.read_csv(folder / "data.csv")
    if len(frame) != metadata["num_instances"]:
        raise ValueError("Downloaded row count disagrees with source metadata")
    features = [v["name"] for v in metadata["variables"] if v["role"] == "Feature"]
    if set(frame.columns) != set(features + metadata["target_col"] + (metadata["index_col"] or [])):
        raise ValueError("Unexpected source schema")
    raw = frame[features].copy()
    y = frame[spec["target"]].copy()
    categories = {}
    if key == "sylhet":
        y = y.map({"Negative": 0, "Positive": 1})
        for col in features:
            if col != "age":
                categories[col] = ["Female", "Male"] if col == "gender" else ["No", "Yes"]
                observed = set(raw[col].dropna())
                if not observed <= set(categories[col]):
                    raise ValueError(f"Unexpected categories in {col}: {observed}")
                raw[col] = raw[col].map({v:i for i,v in enumerate(categories[col])})
    else:
        for v in metadata["variables"]:
            if v["role"] == "Feature" and v["type"] == "Binary":
                categories[v["name"]] = ["0", "1"]
                if not set(raw[v["name"]].dropna()) <= {0, 1}:
                    raise ValueError(f"Invalid binary values: {v['name']}")
    x = raw.apply(pd.to_numeric, errors="raise").astype(float)
    if y.isna().any() or set(y.unique()) != {0, 1} or np.isinf(x.to_numpy()).any():
        raise ValueError("Invalid labels or non-finite predictors")
    y = y.astype(int)
    # IDs and target are deliberately excluded. Identical observed profiles stay
    # in one partition even when they carry conflicting labels.
    groups = pd.util.hash_pandas_object(x, index=False).astype(str)
    profile_labels = pd.DataFrame({"group": groups, "label": y}).groupby("group").label.nunique()
    audit = {"rows": len(x), "features": features, "categories": categories,
             "class_counts": {str(k):int(v) for k,v in y.value_counts().sort_index().items()},
             "missing": x.isna().sum().to_dict(), "unique_profiles": int(groups.nunique()),
             "repeated_profile_rows": int(groups.duplicated().sum()),
             "duplicate_rows_excluding_id": int(frame.drop(columns=metadata["index_col"] or []).duplicated().sum()),
             "profiles_with_conflicting_labels": int(profile_labels.gt(1).sum()),
             "sha256": sha256(folder / "data.csv"), "metadata_sha256": sha256(folder / "metadata.json"),
             "source": metadata["repository_url"], "target_definition": spec["label"],
             "variables": metadata["variables"]}
    return x, y, groups, audit


def grouped_holdout(y, groups, seed=42):
    stats = pd.DataFrame({"group": groups, "y": y}).groupby("group").y.mean()
    strata = stats.ge(.5).astype(int)
    development, test = train_test_split(stats.index.to_numpy(), test_size=.2, random_state=seed, stratify=strata)
    train, validation = train_test_split(development, test_size=.25, random_state=seed+1, stratify=strata.loc[development])
    result = {name: np.flatnonzero(groups.isin(ids).to_numpy()) for name,ids in [("train",train),("validation",validation),("test",test)]}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert not set(groups.iloc[result[a]]) & set(groups.iloc[result[b]])
    assert sum(map(len,result.values())) == len(y)
    return result


class ExternalPreprocessor:
    def __init__(self, categories):
        self.categories = categories

    def validate(self, x):
        if set(x.columns) != set(self.features_):
            raise ValueError("Inputs must match this dataset's predictor schema exactly.")
        x = x[self.features_].apply(pd.to_numeric, errors="raise").astype(float)
        if np.isinf(x.to_numpy()).any():
            raise ValueError("Infinite values are not allowed")
        for c, levels in self.categories.items():
            if not set(x[c].dropna()) <= set(range(len(levels))):
                raise ValueError(f"Unknown category in {c}")
        return x

    def fit(self, x):
        self.features_ = list(x.columns)
        x = self.validate(x)
        self.fill_ = x.median()
        for c in self.categories:
            modes = x[c].mode()
            self.fill_[c] = modes.iloc[0] if len(modes) else np.nan
        if self.fill_.isna().any():
            raise ValueError("Entirely missing training column")
        return self

    def transform(self, x):
        return self.validate(x).fillna(self.fill_)
