# EdgeSense developer entrypoints.
#
# Two ways to run everything:
#   make up / make demo      the docker-compose stack
#   make dev-*               native, on this machine, no containers
#
# The native targets exist because the interesting work -- redaction, the
# agent loop, the eval -- needs no infrastructure at all, and iterating on it
# through a container rebuild is a waste of a minute per change.

SHELL := /bin/bash
PY := .venv/bin/python
PIP := .venv/bin/pip
VENV_PY := python3.12

export PYTHONPATH := services/edge-agent:services/worker:services/sink:eval:.
export POLICY_CATALOG ?= tools/corpus/policies.yaml

.DEFAULT_GOAL := help

# ---------------------------------------------------------------- setup ---

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.venv:
	$(VENV_PY) -m venv .venv
	$(PIP) install -q --upgrade pip

.PHONY: install
install: .venv ## Create the venv and install every Python service
	$(PIP) install -q -e packages/
	$(PIP) install -q -r services/edge-agent/requirements.txt
	$(PIP) install -q -r services/worker/requirements.txt
	$(PIP) install -q -r services/sink/requirements.txt
	$(PIP) install -q pytest pytest-timeout pytest-asyncio
	$(PY) -m spacy download en_core_web_sm
	@echo "installed"

.PHONY: portal-install
portal-install: ## Install portal dependencies
	cd portal && npm install

# ------------------------------------------------------------- corpus ---

.PHONY: corpus
corpus: ## Generate the golden corpus (48 labelled calls)
	$(PY) -m tools.corpus.generate

.PHONY: audio
audio: corpus ## Synthesise real call audio from the corpus
	$(PY) -m tools.audio.synthesize

.PHONY: fixtures
fixtures: corpus ## Regenerate the Go cross-language contract fixtures
	$(PY) -m tools.fixtures.make_segments

# ---------------------------------------------------------------- test ---

.PHONY: test
test: test-edge test-go ## Run every test suite

.PHONY: test-edge
test-edge: ## Egress tests: prove no raw PII crosses the boundary
	$(PY) -m pytest services/edge-agent/tests/ -q

.PHONY: test-go
test-go: ## Go contract and server tests
	cd services/ingest && go vet ./... && go test ./...

.PHONY: typecheck
typecheck: ## Typecheck the portal
	cd portal && npx tsc --noEmit

# ------------------------------------------------------------------ eval ---

.PHONY: eval
eval: ## Full text-mode evaluation over the golden corpus
	$(PY) -m harness

.PHONY: eval-audio
eval-audio: ## Full evaluation through real audio and real ASR
	$(PY) -m harness --mode audio --out eval/reports/run-audio.json

.PHONY: eval-adversarial
eval-adversarial: ## Evaluate only the adversarial (obfuscated PII) set
	$(PY) -m harness --categories adversarial

.PHONY: eval-compare
eval-compare: ## Prompt regression: make eval-compare BEFORE=v1 AFTER=v2
	$(PY) -m harness --compare $(or $(BEFORE),v1) $(or $(AFTER),v2)

.PHONY: loadtest
loadtest: ## Ramp concurrency until p95 breaches the 2s budget
	$(PY) scripts/loadtest.py --realtime

# ------------------------------------------------------------ analytics ---

.PHONY: seed
seed: ## Populate ClickHouse with pipeline output for the dashboard
	$(PY) scripts/seed_pipeline.py --apply-schema --truncate

.PHONY: explain
explain: ## Regenerate docs/clickhouse-explain.md from a live server
	$(PY) scripts/explain_report.py

# --------------------------------------------------------------- native ---

.PHONY: dev-api
dev-api: ## Run the read API against a local ClickHouse
	$(PY) -m uvicorn sink.api:app --port 8099 --reload

.PHONY: dev-portal
dev-portal: ## Run the portal dev server (proxies to :8099)
	cd portal && npm run dev

.PHONY: dev-call
dev-call: ## Stream one real call through the edge agent to JSONL
	$(PY) -m edge_agent.main \
		--audio data/audio/gold-pii_fraud_report_full_profile-v0.wav \
		--no-realtime --out /tmp/edgesense-segments.jsonl
	@echo "--- redacted segments ---"
	@$(PY) -c "import json;[print(json.loads(l)['text']) for l in open('/tmp/edgesense-segments.jsonl')]"

# ---------------------------------------------------------------- stack ---

.PHONY: up
up: ## Start the full docker-compose stack
	docker compose up -d --build
	@echo "portal   http://localhost:5173"
	@echo "grafana  http://localhost:3000"
	@echo "jaeger   http://localhost:16686"

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: clean-stack
clean-stack: ## Stop the stack and delete its volumes
	docker compose down -v

.PHONY: logs
logs: ## Tail service logs
	docker compose logs -f ingest worker sink

.PHONY: demo
demo: ## Seeded demo: generate audio, stream a call, open the portal
	docker compose up -d --build
	@echo "waiting for ingest..."
	@until curl -sf http://localhost:8080/healthz >/dev/null 2>&1; do sleep 2; done
	docker compose exec -T edge-agent python -m tools.corpus.generate
	docker compose exec -T edge-agent python -m tools.audio.synthesize --limit 6
	docker compose exec -T edge-agent python -m edge_agent.main \
		--audio data/audio/gold-pii_fraud_report_full_profile-v0.wav \
		--agent-name Ray --no-realtime
	@echo ""
	@echo "call streamed. open http://localhost:5173"

.PHONY: demo-call
demo-call: ## Stream one more call through the running stack
	docker compose exec -T edge-agent python -m edge_agent.main \
		--audio data/audio/$(or $(CALL),gold-escalation_repeat_failure-v0).wav \
		--no-realtime

.PHONY: clean
clean: ## Remove generated artefacts
	rm -rf data/audio/*.wav data/audio/*.turns.json eval/reports/*.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
