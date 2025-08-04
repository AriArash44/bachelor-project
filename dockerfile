# dockerfile for AI model
FROM python:3.10

WORKDIR /app

COPY ./2-sub-models ./2-sub-models
COPY ./3-model-ensembling ./3-model-ensembling

WORKDIR /app/3-model-ensembling

ENV PIP_REQUIRE_HASHES=false

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
