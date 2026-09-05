"""Retune J48 on all eligible records and save a demonstration model."""
import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
from .data import load_data, sha256, FEATURES, UNITS
from .paths import CONFIG, ARTIFACTS, RESULTS, DATA
from .evaluate import tune, fit_pipeline, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--tuning-cache", type=Path, help="Reuse a locally computed full-data search with matching configuration and dataset")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    summary = json.loads((args.results / "summary.json").read_text())
    if summary["status"] != "complete" or summary["dataset_sha256"] != sha256(DATA):
        raise ValueError("Complete the full evaluation on this dataset before final training.")
    if json.loads((args.results / "config.json").read_text()) != config:
        raise ValueError("Training configuration must match the completed evaluation.")
    x, y = load_data()
    print("Tuning final J48 on all eligible data (five-fold model selection)", flush=True)
    if args.tuning_cache:
        cached = json.loads(args.tuning_cache.read_text())
        if cached["config"] != config or cached["dataset_sha256"] != sha256(DATA):
            raise ValueError("Final tuning cache does not match this experiment")
        params, search = cached["params"], cached["search"]
    else:
        params, search = tune(x, y, "J48", config, config["seed"])
    prep, model = fit_pipeline(x, y, "J48", params, config["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    joblib.dump(prep, args.output / "preprocessing.joblib")
    write_json(args.output / "tree.json", model.export_tree())
    write_json(args.output / "final_search.json", search)
    import jpype
    metadata = {
        "model_type": "WEKA J48 (C4.5)", "version": "0.1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256(DATA), "records": len(x), "features": FEATURES,
        "selected_features": prep.selected_, "units": UNITS, "classes": {"0": "Negative", "1": "Positive"},
        "threshold": .5, "params": params, "seed": config["seed"], "medians": prep.medians_.to_dict(),
        "tree_size": model.tree_size, "leaf_count": model.leaf_count,
        "training_ranges": {c: [float(x[c].min()), float(x[c].max())] for c in FEATURES},
        "python": platform.python_version(), "java": str(jpype.JClass("java.lang.System").getProperty("java.version")),
        "weka": str(jpype.JClass("weka.core.Version").VERSION),
        "packages": {p: importlib.metadata.version(p) for p in ["python-weka-wrapper3", "scikit-learn", "imbalanced-learn", "pandas", "numpy", "streamlit", "joblib"]},
        "evaluation": {"outer_folds": config["outer_folds"], "inner_folds": config["inner_folds"],
                       "j48": summary["models"]["J48"], "results_directory": str(args.results.resolve()),
                       "summary_sha256": sha256(args.results / "summary.json"),
                       "config_sha256": sha256(args.results / "config.json")},
        "files": {name: sha256(args.output / name) for name in ["j48.model", "preprocessing.joblib", "tree.json", "tree.txt", "tree.dot"]},
    }
    write_json(args.output / "manifest.json", metadata)
    print(f"Saved final model: {params}; {model.tree_size} nodes, {model.leaf_count} leaves", flush=True)


if __name__ == "__main__":
    main()
