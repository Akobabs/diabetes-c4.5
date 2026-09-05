from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/derived/pima_diabetes.csv"
CONFIG = ROOT / "config/experiment.json"
ARTIFACTS = ROOT / "artifacts/final"
RESULTS = ROOT / "results/full"
