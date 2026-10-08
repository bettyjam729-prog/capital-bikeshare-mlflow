# Rebuild the experiment and MLflow model registry when the web image is built.
# The UI is served with MLflow basic authentication (configure secrets in Render).
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY download_data.py train_bike_demand.py verify_bike_registry.py ./
COPY tests ./tests
RUN python -m unittest discover -s tests -v \
    && python download_data.py \
    && python train_bike_demand.py \
    && python verify_bike_registry.py
EXPOSE 10000
CMD ["sh", "-c", "mlflow server --app-name basic-auth --backend-store-uri sqlite:////app/mlflow.db --artifacts-destination /app/mlartifacts --host 0.0.0.0 --port ${PORT:-10000} --allowed-hosts '*.onrender.com'"]
