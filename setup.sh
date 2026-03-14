set -e
rm -f .env
docker compose down -v
mkdir -p ./logs ./plugins ./config
echo "AIRFLOW_UID=$(id -u)" > .env
echo "AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True" >> .env
docker compose up airflow-init
