"""Run nested validation with fold-local preprocessing and saved out-of-fold scores."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold
from .paths import CONFIG, RESULTS, DATA
from .data import load_data, audit, sha256
from .models import make_model
from .preprocessing import Preprocessor, resample_training


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def validate_config(config):
    if config.get("threshold") != .5 or config.get("selection_metric") != "roc_auc":
        raise ValueError("This research protocol fixes the decision threshold at 0.5 and selection metric at roc_auc.")
    if config["outer_folds"] < 2 or config["inner_folds"] < 2:
        raise ValueError("At least two outer and inner folds are required.")


def metrics(y, scores):
    labels = (np.asarray(scores) >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, labels)),
            "precision": float(precision_score(y, labels, zero_division=0)),
            "sensitivity": float(recall_score(y, labels, zero_division=0)),
            "specificity": float(tn / (tn + fp)),
            "f1": float(f1_score(y, labels, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def candidates(config, name):
    if name == "J48":
        model_options = [dict(confidence=c, min_leaf=m) for c, m in itertools.product(config["confidence_factors"], config["minimum_leaf_sizes"])]
    elif name == "Naive Bayes":
        model_options = [dict(smoothing=v) for v in config["naive_bayes_smoothing"]]
    else:
        model_options = [dict(c=v) for v in config["logistic_c"]]
    for smote, cap, count in itertools.product(config["smote"], config["cap_outliers"], config["feature_counts"]):
        for params in model_options:
            yield dict(smote=smote, cap_outliers=cap, feature_count=count, **params)


def prepare(x, y, indices, params, seed):
    train, valid = indices
    prep = Preprocessor(params["cap_outliers"], params["feature_count"], seed).fit(x.iloc[train], y.iloc[train])
    xt, yt = resample_training(prep.transform(x.iloc[train]), y.iloc[train], params["smote"], seed)
    return xt, yt, prep.transform(x.iloc[valid]), y.iloc[valid]


def tune(x, y, name, config, seed):
    validate_config(config)
    splits = list(StratifiedKFold(config["inner_folds"], shuffle=True, random_state=seed).split(x, y))
    cache, records = {}, []
    for params in candidates(config, name):
        key = (params["smote"], params["cap_outliers"], params["feature_count"])
        if key not in cache:
            cache[key] = [prepare(x, y, split, params, seed+i) for i, split in enumerate(splits)]
        scores, sizes = [], []
        for i, (xt, yt, xv, yv) in enumerate(cache[key]):
            model = make_model(name, params, seed+i).fit(xt, yt)
            scores.append(roc_auc_score(yv, model.predict_proba(xv)[:, 1]))
            sizes.append(model.tree_size if name == "J48" else 0)
        records.append({"params": params, "mean_auc": float(np.mean(scores)), "std_auc": float(np.std(scores, ddof=1)), "mean_tree_size": float(np.mean(sizes))})
    # Exact ties favour the smaller tree, then stable candidate order.
    best = min(records, key=lambda r: (-r["mean_auc"], r["mean_tree_size"]))
    return best["params"], records


def fit_pipeline(x, y, name, params, seed):
    prep = Preprocessor(params.get("cap_outliers", False), params.get("feature_count", 8), seed,
                        zero_missing=params.get("zero_missing", True)).fit(x, y)
    xt, yt = resample_training(prep.transform(x), y, params.get("smote", False), seed)
    return prep, make_model(name, params, seed).fit(xt, yt)


def plots(predictions, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows in predictions.groupby("model"):
        if name not in ["J48", "Naive Bayes", "Logistic Regression"]:
            continue
        fpr, tpr, _ = roc_curve(rows.outcome, rows.score)
        ax.plot(fpr, tpr, label=f"{name} (AUC {summary[name]['pooled']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="Sensitivity", title="Outer-fold predictions: ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=180)
    plt.close(fig)
    for name, rows in predictions.groupby("model"):
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(confusion_matrix(rows.outcome, rows.score.ge(.5), labels=[0, 1]), annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                    xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        ax.set(xlabel="Predicted", ylabel="Observed", title=name)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=180)
        plt.close(fig)


def run(config, out, smoke=False):
    validate_config(config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x, y = load_data()
    # Invalidate an older completion marker before overwriting any experiment files.
    write_json(out / "summary.json", {"status": "running", "dataset_sha256": sha256(DATA)})
    write_json(out / "config.json", config)
    write_json(out / "data_audit.json", audit(x, y))
    x.describe().to_csv(out / "descriptive_statistics.csv")
    outer = list(StratifiedKFold(config["outer_folds"], shuffle=True, random_state=config["seed"]).split(x, y))
    write_json(out / "folds.json", [{"fold": i+1, "train": tr.tolist(), "validation": va.tolist()} for i, (tr, va) in enumerate(outer)])
    predictions, fold_metrics = [], []
    for fold, (train, valid) in enumerate(outer, 1):
        for name in ["J48", "Naive Bayes", "Logistic Regression"]:
            print(f"Outer fold {fold}/{len(outer)}: tuning {name}", flush=True)
            params, search = tune(x.iloc[train], y.iloc[train], name, config, config["seed"]+fold)
            write_json(out / f"search_{fold}_{name.lower().replace(' ', '_')}.json", search)
            prep, model = fit_pipeline(x.iloc[train], y.iloc[train], name, params, config["seed"]+fold)
            scores = model.predict_proba(prep.transform(x.iloc[valid]))[:, 1]
            fold_metrics.append(dict(model=name, fold=fold, params=json.dumps(params), **metrics(y.iloc[valid], scores)))
            predictions.extend(dict(model=name, fold=fold, row_id=int(row), outcome=int(y.iloc[row]), score=float(score)) for row, score in zip(valid, scores))
        # Fixed configurations isolate the effects of imputation, pruning, SMOTE,
        # capping and feature selection without selecting on outer validation.
        ablations = {
            "J48 reference": {},
            "J48 zeros untreated": {"zero_missing": False},
            "J48 unpruned": {"unpruned": True},
            "J48 SMOTE": {"smote": True},
            "J48 capped": {"cap_outliers": True},
            "J48 five features": {"feature_count": 5},
        }
        for name, params in ablations.items():
            prep, model = fit_pipeline(x.iloc[train], y.iloc[train], "J48", params, config["seed"]+fold)
            scores = model.predict_proba(prep.transform(x.iloc[valid]))[:, 1]
            fold_metrics.append(dict(model=name, fold=fold, params=json.dumps(params), **metrics(y.iloc[valid], scores)))
            predictions.extend(dict(model=name, fold=fold, row_id=int(row), outcome=int(y.iloc[row]), score=float(score)) for row, score in zip(valid, scores))
        pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
        pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
    predictions, folds = pd.DataFrame(predictions), pd.DataFrame(fold_metrics)
    summary = {}
    metric_names = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]
    for name, rows in predictions.groupby("model"):
        fs = folds[folds.model.eq(name)]
        summary[name] = {"pooled": metrics(rows.outcome, rows.score),
                         "fold_mean": fs[metric_names].mean().to_dict(), "fold_std": fs[metric_names].std().to_dict()}
    write_json(out / "summary.json", {"status": "smoke_only" if smoke else "complete", "dataset_sha256": sha256(DATA),
                                      "protocol": "Nested stratified CV; fixed ablations use outer folds only", "models": summary})
    pd.DataFrame({k: v["pooled"] for k, v in summary.items()}).T.to_csv(out / "comparison.csv", index_label="model")
    plots(predictions, summary, out)
    print(f"Saved evaluation to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(outer_folds=3, inner_folds=2, confidence_factors=[.25], minimum_leaf_sizes=[2],
                      smote=[False], cap_outliers=[False], feature_counts=[8], naive_bayes_smoothing=[1e-9], logistic_c=[1.0])
        if args.output == RESULTS:
            args.output = RESULTS.parent / "smoke"
    run(config, args.output, args.smoke)


if __name__ == "__main__":
    main()
