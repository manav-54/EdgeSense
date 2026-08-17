# Analytics store service: Kafka -> ClickHouse writer, plus the portal read API.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/sink/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY packages/ /app/packages/
RUN pip install --no-cache-dir -e /app/packages

COPY services/sink/ /app/services/sink/
# The API's /api/policies endpoint reads the same catalog the worker uses, so
# the portal can render a violation's full text.
COPY services/worker/worker/policies.py /app/services/worker/worker/policies.py
COPY services/worker/worker/obs.py /app/services/worker/worker/obs.py
COPY services/worker/worker/__init__.py /app/services/worker/worker/__init__.py
COPY tools/corpus/policies.yaml /app/tools/corpus/policies.yaml
COPY deploy/clickhouse/schema.sql /app/deploy/clickhouse/schema.sql

ENV PYTHONPATH=/app/services/sink:/app/services/worker:/app \
    POLICY_CATALOG=/app/tools/corpus/policies.yaml

RUN useradd --create-home --uid 10003 sink && chown -R sink:sink /app
USER sink

EXPOSE 8000 9104
ENTRYPOINT ["python", "-m", "sink.main"]
