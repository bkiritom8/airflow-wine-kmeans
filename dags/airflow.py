from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# NOTE: In Airflow 3.x, enable XCom pickling via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

default_args = {
    'owner': 'mlops-lab',
    'start_date': datetime(2026, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'Airflow_WineClustering',
    default_args=default_args,
    description='KMeans clustering DAG using the Wine dataset',
    catchup=False,
) as dag:

    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

if __name__ == "__main__":
    dag.test()
