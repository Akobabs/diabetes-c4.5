import json
from pathlib import Path
import joblib
import pandas as pd
from .data import FEATURES, validate_features, sha256
from .paths import ARTIFACTS
from .j48_adapter import J48Adapter, JVM_LOCK


class PredictionService:
    def __init__(self, directory=ARTIFACTS):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        for name, expected in self.manifest["files"].items():
            if sha256(self.directory / name) != expected:
                raise ValueError(f"Model artifact integrity check failed: {name}")
        self.preprocessor = joblib.load(self.directory / "preprocessing.joblib")
        self.model = J48Adapter.load(self.directory, self.manifest["selected_features"])

    def predict(self, values):
        with JVM_LOCK:
            x = validate_features(pd.DataFrame([values]))
            cleaned = self.preprocessor.clean(x)
            missing = cleaned.columns[cleaned.iloc[0].isna()].tolist()
            transformed = self.preprocessor.transform(x)
            score = float(self.model.predict_proba(transformed)[0, 1])
            label = int(score >= self.manifest["threshold"])
            explanation = self.model.explain(transformed)
            imputed = {c: float(self.preprocessor.medians_[c]) for c in missing}
            changed = {c: {"before": float(cleaned[c].iloc[0]), "after": float(transformed[c].iloc[0])}
                       for c in transformed.columns if c not in missing and float(cleaned[c].iloc[0]) != float(transformed[c].iloc[0])}
            outside = [c for c in FEATURES if pd.notna(x[c].iloc[0]) and
                       not self.manifest["training_ranges"][c][0] <= x[c].iloc[0] <= self.manifest["training_ranges"][c][1]]
            return {"class": label, "label": self.manifest["classes"][str(label)], "positive_class_score": score,
                    "threshold": self.manifest["threshold"], "imputed": imputed, "capped": changed,
                    "outside_training_range": outside, "selected_features": self.manifest["selected_features"],
                    "model_version": self.manifest["version"], **explanation}
