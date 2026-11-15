"""Decision Tree supervised learning and K-Means clustering."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier


def supervised(df):
    fields=["age","age_group","sex","region","visit_type"]
    data=df[fields+["category"]].dropna()
    xtr,xte,ytr,yte=train_test_split(data[fields],data.category,test_size=.25,
                                      random_state=42,stratify=data.category)
    prep=ColumnTransformer([("age",StandardScaler(),["age"]),
        ("labels",OneHotEncoder(handle_unknown="ignore"),fields[1:])])
    model=make_pipeline(prep,DecisionTreeClassifier(max_depth=4,random_state=42,
                                                     class_weight="balanced"))
    model.fit(xtr,ytr); predicted=model.predict(xte)
    return accuracy_score(yte,predicted),pd.DataFrame({
        "Actual synthetic category":yte.values,"Predicted category":predicted})


def unsupervised(df,k=3):
    data=df[["age","reported_severity"]].copy()
    data["severity"]=data.pop("reported_severity").map({"Mild":1,"Moderate":2,"Severe":3})
    data=data.dropna(); x=StandardScaler().fit_transform(data)
    labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(x)
    data["cluster"]=labels.astype(str)
    return data,silhouette_score(x,labels)
