import base64
import os
import pickle

import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler


def load_data():
    """Load the Wine dataset and return serialized DataFrame."""
    wine = load_wine(as_frame=True)
    df = wine.frame.drop(columns=['target'])  # unsupervised clustering, drop label
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """Deserialize, select clustering features, apply MinMax scaling."""
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    df = df.dropna()
    clustering_data = df[["alcohol", "ash", "alcalinity_of_ash"]]
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
    serialized = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """Fit KMeans for k=1..15, save best model, return SSE list for elbow method."""
    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(1, 16):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)
    return sse


def load_model_elbow(filename: str, sse: list):
    """Load saved model, find optimal k via elbow, predict on a sample."""
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))
    kl = KneeLocator(range(1, 16), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # Predict cluster for the first wine sample
    wine = load_wine(as_frame=True)
    df = wine.frame.drop(columns=['target'])
    sample = df[["alcohol", "ash", "alcalinity_of_ash"]].head(1)
    scaler = MinMaxScaler()
    scaler.fit(df[["alcohol", "ash", "alcalinity_of_ash"]])
    sample_scaled = scaler.transform(sample)
    pred = loaded_model.predict(sample_scaled)[0]
    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred
