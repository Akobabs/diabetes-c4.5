"""Produce exportable research figures and a factual results draft from saved scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from diabetes_c45.data import load_data, ZERO_MISSING
from diabetes_c45.paths import ROOT, RESULTS, ARTIFACTS
from diabetes_c45.evaluate import plots

summary = json.loads((RESULTS / "summary.json").read_text())
if summary["status"] != "complete":
    raise ValueError("A completed full evaluation is required")
manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
predictions = pd.read_csv(RESULTS / "predictions.csv")
plots(predictions, summary["models"], RESULTS)
x, y = load_data()
clean = x.copy()
clean[ZERO_MISSING] = clean[ZERO_MISSING].replace(0, np.nan)
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for col, ax in zip(clean.columns, axes.flat):
    sns.histplot(clean[col], bins=20, ax=ax, color="#16765b")
    ax.set_title(col)
    ax.set_xlabel("")
fig.suptitle("Pima predictors (missing-value indicators excluded)")
fig.tight_layout()
fig.savefig(RESULTS / "feature_distributions.png", dpi=180)
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(clean.corr(), cmap="BrBG", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Exploratory predictor correlations (pairwise observed values)")
fig.tight_layout()
fig.savefig(RESULTS / "correlations.png", dpi=180)
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4))
clean.isna().sum().plot.bar(ax=ax, color="#16765b")
ax.set(title="Missing entries after zero-indicator conversion", ylabel="Records")
fig.tight_layout()
fig.savefig(RESULTS / "missingness.png", dpi=180)
plt.close(fig)
names = ["J48", "Naive Bayes", "Logistic Regression"]
metrics = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]
lines = ["# Implementation and results draft", "", "Generated from the completed experiment; these are observed results, not target performance figures.", "",
         "## Implementation", "", f"Python {manifest['python']} controls preprocessing, nested validation and WEKA {manifest['weka']} J48 through python-weka-wrapper3. Java {manifest['java']} supplies the runtime. Streamlit provides the local prototype. The dataset contains 768 records, with 500 negative and 268 positive original outcomes.", "",
         "The original proposal's class-specific imputation was replaced with training-fold predictor medians. Ten stratified outer folds estimate generalisation, with five inner folds selecting preprocessing and model settings. Positive predictions use score >= 0.5. SMOTE, when selected, affects training records only. Five-feature selection uses training-only mutual information; optional outlier capping uses training-derived 1.5-IQR bounds.", "",
         "## Main results: pooled outer-fold predictions", "", "| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|"]
for name in names:
    row = summary["models"][name]["pooled"]
    lines.append("| " + name + " | " + " | ".join(f"{row[m]:.4f}" for m in metrics) + " |")
lines += ["", "## Fold means and standard deviations", "", "| Model | Accuracy | Sensitivity | ROC-AUC |", "|---|---:|---:|---:|"]
for name in names:
    row = summary["models"][name]
    lines.append("| " + name + " | " + " | ".join(f"{row['fold_mean'][m]:.4f} ± {row['fold_std'][m]:.4f}" for m in ["accuracy", "sensitivity", "roc_auc"]) + " |")
j48 = summary["models"]["J48"]["pooled"]
best = max(names, key=lambda n: summary["models"][n]["pooled"]["roc_auc"])
lines += ["", "## Interpretation", "",
          f"Across held-out records, tuned J48 produced {j48['tp']} true positives, {j48['fn']} false negatives, {j48['tn']} true negatives and {j48['fp']} false positives. {best} had the highest pooled ROC-AUC among the three main classifiers in this run. This is a descriptive comparison, not evidence of statistical significance.", "",
          "The majority-class reference accuracy is 500/768 = 65.10%, but always predicting negative would have zero sensitivity. Accuracy should therefore be read together with sensitivity, specificity and AUC.", "",
          "## Controlled preprocessing and pruning comparisons", "", "| Fixed J48 configuration | Accuracy | Sensitivity | ROC-AUC |", "|---|---:|---:|---:|"]
for name, row in summary["models"].items():
    if name not in names:
        lines.append("| " + name + " | " + " | ".join(f"{row['pooled'][m]:.4f}" for m in ["accuracy", "sensitivity", "roc_auc"]) + " |")
lines += ["", "These use default confidence 0.25 and minimum leaf size 2 unless pruning is disabled. The reference uses median imputation, all eight predictors, no capping and no resampling. Each other row changes just the named component. They share the outer folds and do not select settings using outer validation scores.", "",
          "## Final demonstration model", "", f"After evaluation, J48 was retuned on all records and fitted with {manifest['params']}. The saved tree has {manifest['tree_size']} nodes and {manifest['leaf_count']} leaves. Retained predictors: {', '.join(manifest['selected_features'])}.", "",
          "The prototype follows actual J48 split objects and reports imputation or clipping. Its model score is not presented as a calibrated clinical risk. The full-data tree is different from the outer-fold models; the reported generalisation metrics evaluate the training procedure, not an independent test of this exact final tree.", "",
          "## Limitations and conclusion", "",
          "The data are small and restricted to the source population. Missing indicators are extensive in insulin and skin thickness. Random cross-validation is internal validation, not evidence of performance in Nigerian clinics or other populations. Synthetic count features can be fractional after SMOTE. Scores were not calibrated, and fold standard deviations are not confidence intervals. The label does not define a future forecast horizon.", "",
          "The implementation demonstrates a reproducible and interpretable C4.5 classification workflow and a working research interface. External validation, prospective evaluation and probability calibration are appropriate future work. Do not claim clinical readiness or that preprocessing always improves every metric.", "",
          "## Evidence for Chapters 4 and 5", "",
          "Use results/full/comparison.csv, fold_metrics.csv, predictions.csv, folds.json and search logs as the numeric evidence. Figures include roc_curves.png, confusion matrices, feature_distributions.png, correlations.png and missingness.png. The final tree is in artifacts/final/tree.dot and tree.txt. Dataset attribution and measurement definitions are in data/README.md. Update Chapter 3 to match this implemented protocol before incorporating these results.", ""]
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/IMPLEMENTATION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("Generated research figures and docs/IMPLEMENTATION_RESULTS.md")
