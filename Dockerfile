FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY artifacts/baseline_model.pkl artifacts/metrics.json artifacts/run_manifest.json ./artifacts/
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "denial_appeal.serve_stub:app", "--host", "0.0.0.0", "--port", "8000"]
