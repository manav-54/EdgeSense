"""Intelligence worker entrypoint.

    python -m worker.main
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal

from worker.analysis.live import LiveAnalyzer, LiveConfig
from worker.analysis.postcall import PostCallAnalyzer
from worker.consumer import ConsumerConfig, InsightWorker
from worker.llm import build_provider
from worker.obs import get_logger, setup_logging, setup_tracing, start_metrics
from worker.policies import load_policy_store

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EdgeSense intelligence worker")
    ap.add_argument("--brokers", default=os.environ.get("KAFKA_BROKERS", "localhost:9092"))
    ap.add_argument("--segments-topic", default=os.environ.get("TOPIC_SEGMENTS", "transcript.segments"))
    ap.add_argument("--insights-topic", default=os.environ.get("TOPIC_INSIGHTS", "call.insights"))
    ap.add_argument("--group-id", default=os.environ.get("WORKER_GROUP_ID", "edgesense-worker"))
    ap.add_argument("--provider", default=None, help="azure | offline | auto")
    ap.add_argument("--prompt-version", default=os.environ.get("PROMPT_VERSION", "latest"))
    ap.add_argument("--no-agent", action="store_true",
                    help="Run the deterministic fast path only (no LLM calls).")
    ap.add_argument("--metrics-port", type=int,
                    default=int(os.environ.get("WORKER_PROMETHEUS_PORT", "9103")))
    ap.add_argument("--from-beginning", action="store_true",
                    help="Consume the topic from the earliest offset.")
    return ap


async def amain(args: argparse.Namespace) -> int:
    provider = build_provider(args.provider)
    policies = load_policy_store()

    live = LiveAnalyzer(
        provider, policies,
        LiveConfig(prompt_version=args.prompt_version, use_agent=not args.no_agent),
    )
    post_call = PostCallAnalyzer(provider, policies, prompt_version="latest")

    worker = InsightWorker(
        ConsumerConfig(
            brokers=args.brokers,
            segments_topic=args.segments_topic,
            insights_topic=args.insights_topic,
            group_id=args.group_id,
            auto_offset_reset="earliest" if args.from_beginning else "latest",
        ),
        live,
        post_call,
    )

    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopping.set)

    await worker.start()
    log.info("worker ready", provider=provider.name, model=provider.model,
             policies=policies.name, prompt_version=args.prompt_version,
             agent_path=not args.no_agent)

    runner = asyncio.create_task(worker.run())
    stopper = asyncio.create_task(stopping.wait())
    done, pending = await asyncio.wait({runner, stopper}, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()
    await worker.stop()

    if runner in done and runner.exception():
        log.error("consumer loop failed", error=str(runner.exception()))
        return 1
    return 0


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(os.environ.get("LOG_LEVEL", "info"))
    setup_tracing("worker", os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""))
    start_metrics(args.metrics_port)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
