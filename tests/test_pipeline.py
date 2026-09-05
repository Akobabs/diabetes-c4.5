import json
import numpy as np
import pandas as pd
import pytest
from diabetes_c45.data import load_data, validate_features, FEATURES
from diabetes_c45.preprocessing import Preprocessor, resample_training
from diabetes_c45.j48_adapter import J48Adapter
from diabetes_c45.evaluate import metrics


def test_dataset_contract():
    x, y = load_data()
    assert x.shape == (768, 8)
    assert y.value_counts().to_dict() == {0: 500, 1: 268}
    with pytest.raises(ValueError, match="exactly"):
        validate_features(x.assign(Outcome=y))
    with pytest.raises(ValueError, match="negative"):
        validate_features(x.iloc[:1].assign(Glucose=-1))
    with pytest.raises(ValueError, match="Infinite"):
        validate_features(x.iloc[:1].assign(BMI=np.inf))
    with pytest.raises(ValueError, match="whole"):
        validate_features(x.iloc[:1].assign(Pregnancies=1.5))


def test_imputation_uses_training_statistics_without_class_labels():
    x, y = load_data()
    train = x.iloc[:100].copy()
    prep = Preprocessor().fit(train, y.iloc[:100])
    reversed_labels = Preprocessor().fit(train, 1-y.iloc[:100])
    pd.testing.assert_series_equal(prep.medians_, reversed_labels.medians_)
    expected = train.Glucose.replace(0, np.nan).median()
    assert prep.medians_.Glucose == expected
    held_out = x.iloc[100:102].copy()
    held_out.iloc[0, held_out.columns.get_loc("Glucose")] = 0
    held_out.iloc[1, held_out.columns.get_loc("Glucose")] = 10000
    result = prep.transform(held_out)
    assert result.iloc[0].Glucose == expected
    assert prep.medians_.Glucose == expected
    assert prep.transform(x.iloc[:1].assign(Pregnancies=0)).iloc[0].Pregnancies == 0
    # Column order is canonicalised before prediction.
    pd.testing.assert_frame_equal(prep.transform(x.iloc[:2]), prep.transform(x.iloc[:2, ::-1]))


def test_smote_only_returns_training_rows_and_preserves_units():
    x, y = load_data()
    train = x.iloc[:200]
    prep = Preprocessor().fit(train, y.iloc[:200])
    original = prep.transform(train)
    sampled, labels = resample_training(original, y.iloc[:200], True, 42)
    np.testing.assert_allclose(sampled.iloc[:200], original, atol=1e-10)
    assert np.bincount(labels)[0] == np.bincount(labels)[1]
    assert len(train) == 200


def test_metric_orientation():
    result = metrics([0, 0, 1, 1], [.1, .8, .2, .9])
    assert result["tp"] == result["tn"] == result["fp"] == result["fn"] == 1
    assert result["specificity"] == result["sensitivity"] == .5
    assert result["roc_auc"] == .75


def test_j48_serialization_and_exact_tree_paths(tmp_path):
    x, y = load_data()
    prep = Preprocessor().fit(x, y)
    transformed = prep.transform(x)
    model = J48Adapter().fit(transformed, y)
    score = model.predict_proba(transformed)
    assert score.shape == (768, 2)
    np.testing.assert_allclose(score.sum(axis=1), 1)
    tree = model.export_tree()
    assert tree["leaf"] is False
    model.save(tmp_path)
    loaded = J48Adapter.load(tmp_path, FEATURES)
    np.testing.assert_allclose(loaded.predict_proba(transformed), score, atol=0, rtol=0)
    for i in range(len(x)):
        explanation = loaded.explain(transformed.iloc[[i]])
        leaf = np.asarray(explanation["leaf_training_weight"])
        assert leaf.sum() > 0
        np.testing.assert_allclose(leaf / leaf.sum(), score[i], atol=1e-12)
    # Check the exact split and its two floating-point neighbours through J48.
    for value in [tree["threshold"], np.nextafter(tree["threshold"], -np.inf), np.nextafter(tree["threshold"], np.inf)]:
        row = transformed.iloc[[0]].copy()
        row[tree["feature"]] = value
        path = loaded.explain(row)
        leaf = np.asarray(path["leaf_training_weight"])
        np.testing.assert_allclose(leaf / leaf.sum(), loaded.predict_proba(row)[0], atol=1e-12)
