# Edge agent: local ASR, PII redaction, and the corpus/audio generation tools.
#
# This image is the privacy boundary. It is the only one that ever holds raw
# audio or an unredacted transcript, and it is deliberately the only one with
# a speech model or a TTS engine in it -- nothing downstream has any use for
# either, and giving them one would blur where the boundary is.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# espeak-ng gives the container a TTS backend so the demo generates real audio
# on Linux instead of shipping canned WAVs. ffmpeg normalises whatever it
# produces to the 16 kHz mono the ASR expects.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/edge-agent/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m spacy download en_core_web_sm

# Pre-fetch the whisper weights at build time. Downloading on first call would
# put a 30 s model fetch inside the latency budget of the first real
# conversation, and would make the container fail closed without a network.
ENV WHISPER_MODEL=tiny.en \
    WHISPER_MODEL_DIR=/opt/models
RUN python - <<'PY'
from faster_whisper import WhisperModel
WhisperModel("tiny.en", device="cpu", compute_type="int8", download_root="/opt/models")
print("whisper weights cached")
PY

COPY packages/ /app/packages/
RUN pip install --no-cache-dir -e /app/packages

COPY services/edge-agent/ /app/services/edge-agent/
COPY tools/ /app/tools/
COPY eval/ /app/eval/
COPY scripts/ /app/scripts/

ENV PYTHONPATH=/app/services/edge-agent:/app

# Unprivileged: this process handles customer audio, so a container escape
# should not also be a root escape.
RUN useradd --create-home --uid 10001 edge \
    && mkdir -p /app/data/audio /opt/models \
    && chown -R edge:edge /app /opt/models
USER edge

EXPOSE 9101
ENTRYPOINT ["python", "-m", "edge_agent.main"]
CMD ["--help"]
