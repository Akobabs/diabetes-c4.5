from pathlib import Path
import json
import pandas as pd
import streamlit as st
from diabetes_c45.data import FEATURES, UNITS, sha256
from diabetes_c45.paths import ARTIFACTS, RESULTS
from diabetes_c45.predict import PredictionService

st.set_page_config(page_title="Pima | Diabetes research", page_icon="🌿", layout="wide")
st.markdown("""<style>
.stApp {background: #f7f9f8;}
h1,h2,h3 {color: #123d35;}
div[data-testid="stMetric"] {background: white; border: 1px solid #dce6e1; padding: 16px; border-radius: 12px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_service(manifest_content):
    return PredictionService(ARTIFACTS)


st.sidebar.title("Pima research")
st.sidebar.caption("C4.5 · Interpretable classification")
page = st.sidebar.radio("Explore", ["Predict", "Research results", "Decision tree", "About the study"])
st.sidebar.divider()
st.sidebar.caption("Research prototype based on 768 Pima records. The study population consists of women aged 21 or older of Pima Indian heritage. This tool does not establish a clinical diagnosis.")

manifest_path = ARTIFACTS / "manifest.json"
if not manifest_path.exists():
    st.title("Model preparation")
    st.info("The research model has not been built yet. Run evaluation and training using the README commands, then refresh this page.")
    st.stop()

try:
    service = load_service(manifest_path.read_text())
except Exception as exc:
    st.error(f"Unable to load the saved research model: {exc}")
    st.stop()
manifest = service.manifest

if page == "Predict":
    st.caption("PATIENT INPUTS")
    st.title("Understand the prediction.")
    st.write("Enter the study measurements to see the model's classification and the decision path behind it.")
    st.caption("Glucose and insulin refer to two-hour test measurements; blood pressure is diastolic. Unknown inputs are filled using saved training medians.")
    labels = {"Pregnancies": "Pregnancies", "Glucose": "2-hour plasma glucose", "BloodPressure": "Diastolic blood pressure",
              "SkinThickness": "Triceps skinfold thickness", "Insulin": "2-hour serum insulin", "BMI": "Body mass index",
              "DiabetesPedigreeFunction": "Diabetes pedigree function", "Age": "Age"}
    with st.form("patient"):
        cols = st.columns(2)
        values = {}
        for i, c in enumerate(FEATURES):
            with cols[i % 2]:
                integer = c in ["Pregnancies", "Age"]
                minimum = 21 if c == "Age" else (0 if integer else 0.0)
                values[c] = st.number_input(f"{labels[c]} ({UNITS[c]})", min_value=minimum, value=None,
                                             step=1 if integer else (.001 if c == "DiabetesPedigreeFunction" else .1),
                                             format="%d" if integer else ("%.3f" if c == "DiabetesPedigreeFunction" else "%.1f"),
                                             placeholder="Enter value or leave unknown", key=c)
        submitted = st.form_submit_button("Show prediction", type="primary", width="stretch")
    if submitted:
        if all(v is None for v in values.values()):
            st.warning("Enter at least one measurement before requesting a prediction.")
            st.session_state.pop("prediction", None)
        else:
            try:
                st.session_state.prediction = service.predict(values)
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.pop("prediction", None)
    if "prediction" in st.session_state:
        result = st.session_state.prediction
        st.divider()
        st.subheader("Last submitted prediction")
        a, b, c = st.columns(3)
        a.metric("Model classification", result["label"])
        b.metric("Positive-class model score", f"{result['positive_class_score']:.3f}")
        c.metric("Decision threshold", f"{result['threshold']:.2f}")
        st.caption("The score is the tree's class estimate, not a calibrated medical risk probability. Editing the form does not change this result until you submit again.")
        if result["imputed"]:
            st.info("Training medians used for: " + ", ".join(result["imputed"]))
        if result["capped"]:
            st.info("Training-derived outlier bounds applied to: " + ", ".join(result["capped"]))
        if result["outside_training_range"]:
            st.warning("Outside the observed training range: " + ", ".join(result["outside_training_range"]))
        st.subheader("Decision path")
        if result["steps"]:
            for i, step in enumerate(result["steps"], 1):
                st.write(f"{i}. **{labels[step['feature']]}**: {step['value']:.6g} {step['operator']} {step['threshold']:.12g}")
        else:
            st.write("The fitted tree is a single leaf; no split conditions apply.")
        st.caption("Conditions use the transformed values actually passed to J48. Full-precision values are available in the download.")
        st.download_button("Download this result", json.dumps(result, indent=2), "prediction.json", "application/json")

elif page == "Research results":
    st.title("Research results")
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        st.info("The full evaluation is not available yet.")
        st.stop()
    summary = json.loads(summary_path.read_text())
    if (summary["status"] != "complete" or summary["dataset_sha256"] != manifest["dataset_sha256"]
            or sha256(summary_path) != manifest["evaluation"]["summary_sha256"]):
        st.error("The evaluation does not match the completed model dataset.")
        st.stop()
    st.write(f"Stratified {manifest['evaluation']['outer_folds']}-fold outer validation with {manifest['evaluation']['inner_folds']}-fold inner model selection. Each patient was evaluated once per model outside its training fold.")
    names = ["J48", "Naive Bayes", "Logistic Regression"]
    metric_names = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]
    table = pd.DataFrame({n: summary["models"][n]["pooled"] for n in names}).T
    st.subheader("Pooled out-of-fold metrics")
    st.dataframe(table[metric_names].style.format("{:.3f}"), width="stretch")
    st.caption("These estimate the training procedure's performance. The final demonstration tree is subsequently fitted on all 768 records.")
    with st.expander("Fold means and standard deviations"):
        st.dataframe(pd.DataFrame({n: {m: f"{summary['models'][n]['fold_mean'][m]:.3f} ± {summary['models'][n]['fold_std'][m]:.3f}" for m in metric_names} for n in names}).T)
    st.image(str(RESULTS / "roc_curves.png"))
    columns = st.columns(3)
    for column, name in zip(columns, names):
        column.image(str(RESULTS / f"confusion_{name.lower().replace(' ', '_')}.png"))
    with st.expander("Controlled J48 comparisons"):
        st.dataframe(pd.DataFrame({n: v["pooled"] for n, v in summary["models"].items() if n not in names}).T[metric_names].style.format("{:.3f}"))
        st.caption("Fixed default J48 settings isolate one change at a time; these rows are not tuned competitors.")
    st.download_button("Download comparison CSV", (RESULTS / "comparison.csv").read_bytes(), "comparison.csv", "text/csv")

elif page == "Decision tree":
    st.title("The fitted C4.5 tree")
    a, b, c = st.columns(3)
    a.metric("Nodes", manifest["tree_size"])
    b.metric("Leaves", manifest["leaf_count"])
    c.metric("Retained predictors", len(manifest["selected_features"]))
    st.write("Selected predictors: " + ", ".join(manifest["selected_features"]))
    st.graphviz_chart((ARTIFACTS / "tree.dot").read_text())
    with st.expander("Text rules and training configuration"):
        st.code((ARTIFACTS / "tree.txt").read_text())
        st.json(manifest["params"])
    st.download_button("Download tree rules", (ARTIFACTS / "tree.txt").read_bytes(), "tree.txt", "text/plain")

else:
    st.title("About the study")
    st.write("Prediction of Diabetes Using C4.5 — a Python research pipeline and Streamlit decision-support prototype.")
    st.markdown("The model uses **WEKA J48**, the C4.5 implementation, through Python. Naive Bayes and Logistic Regression provide baseline comparisons.")
    st.markdown("Data: [Pima Indians Diabetes, OpenML 37](https://www.openml.org/d/37). There are 768 records, with 500 negative and 268 positive original labels.")
    st.write("Training-fold medians handle missing measurements. Inner validation selects pruning, optional SMOTE, outlier capping and feature count. The target is the existing dataset label; the study does not establish a future time-to-diabetes forecast.")
    st.caption("Generalisability to other populations requires external validation. Clinical deployment is outside this project's scope.")
    st.write(f"Model version {manifest['version']} · created {manifest['created_utc']}")
