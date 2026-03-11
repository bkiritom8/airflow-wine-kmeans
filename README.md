# Airflow Lab — Wine Dataset Clustering with KMeans

An Apache Airflow DAG that orchestrates a 4-step ML pipeline to cluster the **Wine dataset** using **KMeans** and determine the optimal number of clusters via the elbow method.

## What Changed from the Original

The original lab loaded a credit card CSV file (`data/file.csv`) with financial features (BALANCE, PURCHASES, CREDIT_LIMIT). This version replaces it with the **Wine dataset** from `sklearn`, which requires no external CSV files.

| | Original | This Lab |
|---|---|---|
| Dataset | Credit card CSV (BALANCE, PURCHASES, CREDIT_LIMIT) | Wine dataset (alcohol, ash, alcalinity_of_ash) |
| Source | Local CSV file | `sklearn.datasets.load_wine` |
| Algorithm | KMeans | KMeans (unchanged) |

## Dataset

**Wine** from `sklearn.datasets.load_wine` — 178 samples, 13 chemical measurements of wines from 3 cultivars. We cluster on 3 features: `alcohol`, `ash`, `alcalinity_of_ash`.

## DAG Pipeline

```
load_data_task → data_preprocessing_task → build_save_model_task → load_model_task
```

| Task | Function | Description |
|---|---|---|
| load_data_task | `load_data()` | Loads Wine dataset, serializes to base64 |
| data_preprocessing_task | `data_preprocessing()` | Selects 3 features, applies MinMaxScaler |
| build_save_model_task | `build_save_model()` | Fits KMeans k=1..15, saves model, returns SSE |
| load_model_task | `load_model_elbow()` | Finds optimal k via KneeLocator, predicts on a sample |

## Project Structure

```
Airflow_Lab/
├── dags/
│   ├── airflow.py         # DAG definition
│   ├── model/             # Saved model.sav written here at runtime
│   └── src/
│       ├── __init__.py
│       └── lab.py         # 4 pipeline functions
├── docker-compose.yml
├── setup.sh
├── requirements.txt
└── README.md
```

## How to Run

### Step 1 — Prepare the environment

```bash
chmod +x setup.sh
./setup.sh
```

This stops any existing containers, sets `AIRFLOW_UID`, and enables XCom pickling.

### Step 2 — Start Airflow

```bash
docker compose up
```

Wait for all services to be healthy (~1–2 min), then open `http://localhost:8080`.

Login: **admin / admin**

### Step 3 — Trigger the DAG

In the Airflow UI:
1. Find `Airflow_WineClustering` in the DAGs list
2. Toggle it **on**
3. Click the play button to trigger a run

Or trigger from the CLI:
```bash
docker compose exec airflow-scheduler airflow dags trigger Airflow_WineClustering
```

### Step 4 — Check logs

Click into the DAG run → select any task → **Logs** to see output, including the optimal cluster count printed by `load_model_elbow`.

### Tear down

```bash
docker compose down -v
```

## Dependencies

```
apache-airflow
scikit-learn
kneed
pandas
numpy
```

> **Note:** XCom pickling is required to pass numpy arrays between tasks. It is enabled automatically via `setup.sh` and the `docker-compose.yml` environment variable `AIRFLOW__CORE__ENABLE_XCOM_PICKLING=true`.
