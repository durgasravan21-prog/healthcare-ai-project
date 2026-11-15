"""Introductory supervised and unsupervised learning experiments."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def supervised_demo(df):
    """Predict synthetic category labels from demographics and visit type."""
    features = ["age", "age_group", "sex", "region", "visit_type"]
    target = "category"
    data = df[features + [target]].dropna().copy()
    if len(data) < 20 or data[target].nunique() < 2:
        raise ValueError("Not enough labelled records for this demonstration.")
    x, y = data[features], data[target]
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), ["age"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         ["age_group", "sex", "region", "visit_type"]),
    ])
    model = Pipeline([("prepare", prep),
                      ("tree", DecisionTreeClassifier(max_depth=4, random_state=42,
                                                       class_weight="balanced"))])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, prediction),
        "report": classification_report(y_test, prediction, output_dict=True, zero_division=0),
        "train_size": len(y_train), "test_size": len(y_test),
        "predictions": pd.DataFrame({"Actual synthetic category": y_test.values,
                                     "Predicted synthetic category": prediction}),
    }


def unsupervised_demo(df, clusters=3):
    """Cluster by age and severity score; cluster IDs are not diagnoses."""
    severity = {"Mild": 1, "Moderate": 2, "Severe": 3}
    data = df[["age", "reported_severity"]].copy()
    data["severity_score"] = data["reported_severity"].map(severity)
    data = data.dropna()
    if len(data) <= clusters:
        raise ValueError("Not enough records for the selected cluster count.")
    x = StandardScaler().fit_transform(data[["age", "severity_score"]])
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(x)
    result = data.copy()
    result["cluster"] = labels.astype(str)
    score = silhouette_score(x, labels) if len(set(labels)) > 1 else 0.0
    return result, score
