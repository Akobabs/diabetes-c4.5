"""Launcher preflight; validates the actual runtime, data and optional model."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    import diabetes_c45
    if Path(diabetes_c45.__file__).resolve().parents[2] != root:
        raise ValueError("The Python environment points to a different project. Run FullRun.bat.")
    from diabetes_c45.j48_adapter import start_jvm
    from diabetes_c45.evaluate import validate_config
    from diabetes_c45.data import load_data, sha256
    from diabetes_c45.paths import ARTIFACTS, RESULTS, DATA

    if platform.python_version_tuple()[:2] != ("3", "12") or struct.calcsize("P") != 8:
        raise ValueError("The Windows launchers require 64-bit Python 3.12.")
    for line in (root / "requirements.lock").read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"{name}: installed {actual}, expected {expected}. Run FullRun.bat.")
    source = root / "data/raw/pima_diabetes.arff"
    metadata = json.loads((root / "data/raw/pima_openml_metadata.json").read_text())
    if hashlib.md5(source.read_bytes()).hexdigest() != metadata["data_set_description"]["md5_checksum"].lower():
        raise ValueError("The original ARFF checksum does not match OpenML metadata.")
    config = json.loads((root / "config/experiment.json").read_text())
    validate_config(config)
    if (config["outer_folds"], config["inner_folds"]) != (10, 5):
        raise ValueError("FullRun requires the complete 10-outer / 5-inner fold protocol.")
    start_jvm()
    import jpype
    print(f"Python {platform.python_version()} (64-bit), Java {jpype.JClass('java.lang.System').getProperty('java.version')}, WEKA {jpype.JClass('weka.core.Version').VERSION}", flush=True)
    print("Locked dependencies, original data and full research configuration verified.", flush=True)
    if args.model:
        from diabetes_c45.predict import PredictionService
        model = PredictionService(ARTIFACTS)
        summary = json.loads((RESULTS / "summary.json").read_text())
        if (summary.get("status") != "complete"
                or sha256(DATA) != model.manifest["dataset_sha256"]
                or sha256(RESULTS / "summary.json") != model.manifest["evaluation"]["summary_sha256"]
                or sha256(RESULTS / "config.json") != model.manifest["evaluation"]["config_sha256"]):
            raise ValueError("Saved model, data and completed evaluation are not a matching set. Run FullRun.bat.")
        x, _ = load_data()
        result = model.predict(x.iloc[0].to_dict())
        print(f"Saved model verified; sample inference succeeded ({result['label']}, score {result['positive_class_score']:.4f}).", flush=True)


if __name__ == "__main__":
    main()
