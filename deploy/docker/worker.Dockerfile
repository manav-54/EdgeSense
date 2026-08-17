# Intelligence worker: Kafka consumer, agent loop, live + post-call analysis.
#
# No speech model and no audio libraries: by the time data reaches this
# process it is already redacted text, and keeping the image free of anything
# that could consume audio is part of how the boundary stays legible.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY packages/ /app/packages/
RUN pip install --no-cache-dir -e /app/packages

COPY services/worker/ /app/services/worker/
# The policy catalog is the corpus of record for both the lookup_policy tool
# and the golden labels; shipping the same file avoids a desync between what
# the agent is told the rule is and what the eval scores against.
COPY tools/corpus/policies.yaml /app/tools/corpus/policies.yaml

ENV PYTHONPATH=/app/services/worker:/app \
    POLICY_CATALOG=/app/tools/corpus/policies.yaml

RUN useradd --create-home --uid 10002 worker && chown -R worker:worker /app
USER worker

EXPOSE 9103
ENTRYPOINT ["python", "-m", "worker.main"]
