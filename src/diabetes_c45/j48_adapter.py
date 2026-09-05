"""Genuine WEKA J48, with exact decision traversal through its fitted split objects."""
import os
import threading
import numpy as np
from .paths import ROOT

JVM_LOCK = threading.RLock()


def start_jvm():
    with JVM_LOCK:
        import jpype
        if not jpype.isJVMStarted():
            local_java = sorted((ROOT / ".runtime/java").glob("*/bin/server/jvm.dll"))
            if local_java and not os.environ.get("JAVA_HOME"):
                os.environ["JAVA_HOME"] = str(local_java[0].parents[2])
            os.environ.setdefault("WEKA_HOME", str(ROOT / ".runtime/weka"))
            import weka.core.jvm as jvm
            jvm.start(packages=False, max_heap_size="1024m")


def field(obj, name):
    """Read pinned WEKA fields; fail loudly if its internal tree layout changes."""
    cls = obj.getClass()
    while cls is not None:
        try:
            member = cls.getDeclaredField(name)
            member.setAccessible(True)
            return member.get(obj)
        except Exception:
            cls = cls.getSuperclass()
    raise RuntimeError(f"Cannot inspect WEKA tree field {name}; check the pinned version.")


class J48Adapter:
    def __init__(self, confidence=.25, min_leaf=2, unpruned=False):
        self.confidence = confidence
        self.min_leaf = min_leaf
        self.unpruned = unpruned

    def instances(self, x, y=None):
        from weka.core.dataset import Attribute, Instance, Instances
        names = list(x.columns)
        if hasattr(self, "features_") and names != self.features_:
            raise ValueError("Predictor order must match the trained J48 header.")
        attrs = [Attribute.create_numeric(c) for c in names]
        attrs.append(Attribute.create_nominal("Outcome", ["0", "1"]))
        data = Instances.create_instances("pima_diabetes", attrs, len(x))
        data.class_is_last()
        values = np.asarray(x, dtype=float)
        labels = np.full(len(x), np.nan) if y is None else np.asarray(y, dtype=float)
        for row, label in zip(values, labels):
            data.add_instance(Instance.create_instance(np.append(row, label)))
        return data

    def fit(self, x, y):
        start_jvm()
        from weka.classifiers import Classifier
        from weka.core.dataset import Instances
        with JVM_LOCK:
            self.features_ = list(x.columns)
            options = ["-M", str(self.min_leaf)]
            options += ["-U"] if self.unpruned else ["-C", str(self.confidence)]
            self.classifier_ = Classifier(classname="weka.classifiers.trees.J48", options=options)
            data = self.instances(x, y)
            self.classifier_.build_classifier(data)
            self.header_ = Instances.template_instances(data, 0)
        return self

    def predict_proba(self, x):
        with JVM_LOCK:
            return np.asarray([self.classifier_.distribution_for_instance(i) for i in self.instances(x)])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= .5).astype(int)

    @property
    def tree_size(self):
        return int(self.classifier_.jobject.measureTreeSize())

    @property
    def leaf_count(self):
        return int(self.classifier_.jobject.measureNumLeaves())

    def explain(self, x):
        if len(x) != 1:
            raise ValueError("Explanation requires a single record.")
        with JVM_LOCK:
            instance = self.instances(x).get_instance(0)
            node = field(self.classifier_.jobject, "m_root")
            steps = []
            while not bool(field(node, "m_isLeaf")):
                split = field(node, "m_localModel")
                branch = int(split.whichSubset(instance.jobject))
                if branch < 0:
                    raise ValueError("A missing split value survived preprocessing.")
                index = int(field(split, "m_attIndex"))
                threshold = float(field(split, "m_splitPoint"))
                steps.append({"feature": self.features_[index], "value": float(x.iloc[0, index]),
                              "operator": "<=" if branch == 0 else ">", "threshold": threshold})
                node = field(node, "m_sons")[branch]
            distribution = field(node, "m_localModel").distribution()
            counts = [float(distribution.perClass(i)) for i in range(2)]
            return {"steps": steps, "leaf_training_weight": counts}

    def export_tree(self):
        def visit(node):
            split = field(node, "m_localModel")
            dist = split.distribution()
            counts = [float(dist.perClass(i)) for i in range(2)]
            if bool(field(node, "m_isLeaf")):
                return {"leaf": True, "class_weights": counts}
            return {"leaf": False, "feature": self.features_[int(field(split, "m_attIndex"))],
                    "threshold": float(field(split, "m_splitPoint")), "class_weights": counts,
                    "children": [visit(child) for child in field(node, "m_sons")]}
        with JVM_LOCK:
            return visit(field(self.classifier_.jobject, "m_root"))

    def save(self, directory):
        from weka.core import serialization
        with JVM_LOCK:
            serialization.write_all(str(directory / "j48.model"), [self.classifier_.jobject, self.header_.jobject])
            (directory / "tree.txt").write_text(str(self.classifier_), encoding="utf-8")
            (directory / "tree.dot").write_text(self.classifier_.graph, encoding="utf-8")

    @classmethod
    def load(cls, directory, features):
        start_jvm()
        from weka.core import serialization
        from weka.classifiers import Classifier
        from weka.core.dataset import Instances
        with JVM_LOCK:
            objects = serialization.read_all(str(directory / "j48.model"))
            model = cls()
            model.classifier_ = Classifier(jobject=objects[0])
            model.header_ = Instances(jobject=objects[1])
            model.features_ = features
            if model.header_.attribute_names() != features + ["Outcome"]:
                raise ValueError("Saved WEKA header does not match the model manifest.")
            return model
