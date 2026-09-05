"""Application integration tests run after the full model is available."""
import json
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from diabetes_c45.paths import ARTIFACTS, ROOT, RESULTS
from diabetes_c45.data import load_data
from diabetes_c45.predict import PredictionService

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="Run full evaluation and final training first")


def test_saved_prediction_service():
    service = PredictionService()
    x, _ = load_data()
    row = x.iloc[0].to_dict()
    result = service.predict(row)
    transformed = service.preprocessor.transform(x.iloc[[0]])
    np.testing.assert_allclose(result["positive_class_score"], service.model.predict_proba(transformed)[0, 1])
    result_missing = service.predict({**row, "Glucose": None})
    assert result_missing["imputed"]["Glucose"] == service.manifest["medians"]["Glucose"]
    with pytest.raises(ValueError):
        service.predict({**row, "Outcome": 1})


def test_streamlit_prediction_and_navigation():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.number_input) == 8
    # No patient data defaults silently fill the form.
    next(b for b in app.button if b.label == 'Show prediction').click().run()
    assert app.warning
    x, _ = load_data()
    for widget in app.number_input:
        widget.set_value(float(x.iloc[0][widget.key]) if widget.key not in ["Age", "Pregnancies"] else int(x.iloc[0][widget.key]))
    next(b for b in app.button if b.label == 'Show prediction').click().run()
    assert not app.exception
    assert len(app.metric) == 3
    for page in ["Research results", "Decision tree", "About the study", "Predict"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception


def test_outer_predictions_are_complete_and_disjoint():
    import pandas as pd
    folds = json.loads((RESULTS / "folds.json").read_text())
    seen = []
    for fold in folds:
        assert not set(fold["train"]) & set(fold["validation"])
        seen.extend(fold["validation"])
    assert sorted(seen) == list(range(768))
    predictions = pd.read_csv(RESULTS / "predictions.csv")
    for _, rows in predictions.groupby("model"):
        assert len(rows) == 768
        assert rows.row_id.nunique() == 768
        assert rows.score.between(0, 1).all()


def test_reproduces_saved_outer_fold():
    import pandas as pd
    from diabetes_c45.evaluate import fit_pipeline
    config = json.loads((RESULTS / "config.json").read_text())
    fold = json.loads((RESULTS / "folds.json").read_text())[0]
    metrics = pd.read_csv(RESULTS / "fold_metrics.csv")
    params = json.loads(metrics.loc[metrics.model.eq("J48") & metrics.fold.eq(1), "params"].iloc[0])
    x, y = load_data()
    train, valid = fold["train"], fold["validation"]
    prep, model = fit_pipeline(x.iloc[train], y.iloc[train], "J48", params, config["seed"] + 1)
    observed = model.predict_proba(prep.transform(x.iloc[valid]))[:, 1]
    saved = pd.read_csv(RESULTS / "predictions.csv")
    expected = saved.loc[saved.model.eq("J48") & saved.fold.eq(1)].set_index("row_id").loc[valid, "score"]
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)
