from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .j48_adapter import J48Adapter


def make_model(name, params, seed=42):
    if name == "J48":
        return J48Adapter(params.get("confidence", .25), params.get("min_leaf", 2), params.get("unpruned", False))
    if name == "Naive Bayes":
        return GaussianNB(var_smoothing=params["smoothing"])
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["c"], solver="lbfgs", max_iter=2000, random_state=seed))
    raise ValueError(f"Unknown model: {name}")
