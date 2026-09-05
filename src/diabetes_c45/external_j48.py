"""J48 with genuine nominal splits and optional training-only class weights."""
import numpy as np
from .j48_adapter import J48Adapter, JVM_LOCK, field


class ExternalJ48(J48Adapter):
    def __init__(self, categories=None, confidence=.25, min_leaf=20, balanced=False):
        super().__init__(confidence, min_leaf)
        self.categories = categories or {}
        self.balanced = balanced

    def fit(self, x, y):
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        self.class_weights_ = len(y)/(2*counts) if self.balanced else np.ones(2)
        return super().fit(x, y)

    def instances(self, x, y=None):
        from weka.core.dataset import Attribute, Instances
        import jpype
        if list(x.columns) != self.features_:
            raise ValueError("Feature order does not match the saved header")
        attrs = [Attribute.create_nominal(c, self.categories[c]) if c in self.categories else Attribute.create_numeric(c) for c in x.columns]
        attrs.append(Attribute.create_nominal("Outcome", ["0", "1"]))
        data = Instances.create_instances("external_diabetes", attrs, len(x))
        data.class_is_last()
        values = np.asarray(x, dtype=float)
        labels = np.full(len(x), np.nan) if y is None else np.asarray(y, dtype=float)
        dense = jpype.JClass("weka.core.DenseInstance")
        # Direct Java calls avoid per-row Python wrapper introspection on CDC's
        # large dataset. Attribute types and instance content remain WEKA-native.
        for row, label in zip(values, labels):
            weight = 1.0 if y is None else float(self.class_weights_[int(label)])
            data.jobject.add(dense(weight, np.append(row, label)))
        return data

    def predict_proba(self, x):
        with JVM_LOCK:
            data = self.instances(x)
            return np.asarray([self.classifier_.jobject.distributionForInstance(data.jobject.instance(i)) for i in range(len(x))])

    def explain(self, x):
        with JVM_LOCK:
            instance = self.instances(x).jobject.instance(0)
            node = field(self.classifier_.jobject, "m_root")
            steps = []
            while not bool(field(node, "m_isLeaf")):
                split = field(node, "m_localModel")
                branch = int(split.whichSubset(instance))
                if branch < 0: raise ValueError("Unimputed split value")
                col = self.features_[int(field(split, "m_attIndex"))]
                value = float(x[col].iloc[0])
                if col in self.categories:
                    steps.append({"feature":col,"value":self.categories[col][int(value)],"operator":"=", "threshold":self.categories[col][branch]})
                else:
                    steps.append({"feature":col,"value":value,"operator":"<=" if branch == 0 else ">", "threshold":float(field(split,"m_splitPoint"))})
                node = field(node, "m_sons")[branch]
            dist = field(node,"m_localModel").distribution()
            return {"steps":steps,"leaf_training_weight":[float(dist.perClass(i)) for i in range(2)]}

    def export_tree(self):
        def visit(node):
            split=field(node,"m_localModel")
            dist=split.distribution()
            result={"leaf":bool(field(node,"m_isLeaf")),"class_weights":[float(dist.perClass(i)) for i in range(2)]}
            if not result["leaf"]:
                col=self.features_[int(field(split,"m_attIndex"))]
                result.update(feature=col, categories=self.categories.get(col),
                              threshold=None if col in self.categories else float(field(split,"m_splitPoint")),
                              children=[visit(child) for child in field(node,"m_sons")])
            return result
        with JVM_LOCK:
            return visit(field(self.classifier_.jobject,"m_root"))
