import json
import joblib
import pandas as pd
from .paths import ROOT
from .data import sha256
from .external_j48 import ExternalJ48
from .j48_adapter import JVM_LOCK


class ExternalPredictionService:
    def __init__(self, run_id, dataset):
        if dataset not in ['sylhet','cdc'] or not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
            raise ValueError('Invalid model identifier')
        folder=ROOT/'artifacts/external'/run_id/dataset
        self.manifest=json.loads((folder/'manifest.json').read_text())
        for name,digest in self.manifest['files'].items():
            if sha256(folder/name)!=digest:raise ValueError(f'Artifact checksum mismatch: {name}')
        self.prep=joblib.load(folder/'preprocessing.joblib')
        self.model=ExternalJ48.load(folder,self.manifest['features'])
        self.model.categories=self.manifest['categories']

    def predict(self, values):
        with JVM_LOCK:
            x=pd.DataFrame([values])
            checked=self.prep.validate(x)
            transformed=self.prep.transform(checked)
            score=float(self.model.predict_proba(transformed)[0,1])
            return dict(label=self.manifest['label'] if score>=.5 else self.manifest['negative'],
                        score=score,threshold=.5,imputed={c:float(self.prep.fill_[c]) for c in checked if checked[c].isna().any()},
                        **self.model.explain(transformed))
