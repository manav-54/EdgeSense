// Command ingest is the cloud-facing entry point for redacted transcript segments.
//
// It terminates the edge WebSocket, validates every payload against the shared
// contract, dedupes on (call_id, seq, kind) via Redis, and publishes to Kafka
// with bounded-queue backpressure.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/manav-54/edgesense/ingest/internal/obs"
	"github.com/manav-54/edgesense/ingest/internal/publish"
	"github.com/manav-54/edgesense/ingest/internal/server"
	"github.com/manav-54/edgesense/ingest/internal/session"
)

type config struct {
	addr         string
	metricsAddr  string
	brokers      []string
	topic        string
	redisAddr    string
	redisPass    string
	redisDB      int
	otlpEndpoint string
	logLevel     string
	queueSize    int
	workers      int
	// drainGrace is how long in-flight calls get to finish on SIGTERM before
	// the process exits. Sized to outlast one segment round-trip, not a call:
	// a rolling deploy should not wait for a twenty-minute conversation.
	drainGrace time.Duration
}

func load() config {
	return config{
		addr:         env("INGEST_HTTP_ADDR", ":8080"),
		metricsAddr:  env("INGEST_METRICS_ADDR", ":9102"),
		brokers:      strings.Split(env("KAFKA_BROKERS", "localhost:9092"), ","),
		topic:        env("TOPIC_SEGMENTS", "transcript.segments"),
		redisAddr:    env("REDIS_ADDR", "localhost:6379"),
		redisPass:    env("REDIS_PASSWORD", ""),
		redisDB:      envInt("REDIS_DB", 0),
		otlpEndpoint: env("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
		logLevel:     env("LOG_LEVEL", "info"),
		queueSize:    envInt("INGEST_QUEUE_SIZE", 4096),
		workers:      envInt("INGEST_PUBLISH_WORKERS", 4),
		drainGrace:   time.Duration(envInt("INGEST_DRAIN_SECONDS", 15)) * time.Second,
	}
}

func main() {
	cfg := load()
	log := obs.SetupLogging(cfg.logLevel)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	shutdownTracing, err := obs.SetupTracing(ctx, cfg.otlpEndpoint)
	if err != nil {
		log.Warn("tracing setup failed", slog.String("error", err.Error()))
	}

	sessions := session.New(cfg.redisAddr, cfg.redisPass, cfg.redisDB, log)
	defer func() { _ = sessions.Close() }()

	pingCtx, cancelPing := context.WithTimeout(ctx, 3*time.Second)
	if err := sessions.Ping(pingCtx); err != nil {
		// Not fatal. Dedupe fails open, and refusing to start because a cache
		// is cold would make Redis a hard dependency of accepting live calls.
		log.Warn("redis unreachable at startup; dedupe will fail open",
			slog.String("addr", cfg.redisAddr), slog.String("error", err.Error()))
	}
	cancelPing()

	pub := publish.New(publish.Config{
		Brokers:   cfg.brokers,
		Topic:     cfg.topic,
		QueueSize: cfg.queueSize,
		Workers:   cfg.workers,
	}, log)

	srv := server.New(pub, sessions, log)

	mux := http.NewServeMux()
	srv.Routes(mux)
	httpSrv := &http.Server{
		Addr:              cfg.addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		// No WriteTimeout: a WebSocket connection is long-lived by design and
		// a write deadline on the server would sever live calls mid-stream.
		IdleTimeout: 120 * time.Second,
	}

	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", promhttp.Handler())
	metricsSrv := &http.Server{
		Addr:              cfg.metricsAddr,
		Handler:           metricsMux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Info("metrics listening", slog.String("addr", cfg.metricsAddr))
		if err := metricsSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("metrics server failed", slog.String("error", err.Error()))
		}
	}()

	go func() {
		log.Info("ingest listening",
			slog.String("addr", cfg.addr),
			slog.String("topic", cfg.topic),
			slog.String("brokers", strings.Join(cfg.brokers, ",")),
		)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("http server failed", slog.String("error", err.Error()))
			stop()
		}
	}()

	<-ctx.Done()
	log.Info("shutdown signal received; draining", slog.Duration("grace", cfg.drainGrace))

	// Fail readiness first so a load balancer stops sending new calls here
	// while existing ones finish.
	srv.SetReady(false)

	drainCtx, cancelDrain := context.WithTimeout(context.Background(), cfg.drainGrace)
	defer cancelDrain()

	if err := httpSrv.Shutdown(drainCtx); err != nil {
		log.Warn("http shutdown incomplete", slog.String("error", err.Error()))
	}
	// Close the publisher after the server, so segments already read off a
	// socket are still written rather than discarded with the queue.
	if err := pub.Close(); err != nil {
		log.Warn("publisher shutdown incomplete", slog.String("error", err.Error()))
	}
	_ = metricsSrv.Shutdown(drainCtx)
	if shutdownTracing != nil {
		_ = shutdownTracing(drainCtx)
	}
	log.Info("shutdown complete")
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
