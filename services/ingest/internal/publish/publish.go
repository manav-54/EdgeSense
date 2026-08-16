// Package publish writes validated segments to Kafka with explicit backpressure.
package publish

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/segmentio/kafka-go"

	"github.com/manav-54/edgesense/ingest/internal/contracts"
	"github.com/manav-54/edgesense/ingest/internal/obs"
)

// Publisher owns a bounded queue in front of a Kafka writer.
//
// The queue is bounded because the alternative is worse in every direction. An
// unbounded queue converts a Kafka slowdown into unbounded memory growth, and
// the process dies holding every segment it had not yet written -- losing more
// data than it would have shed, and taking every other live call with it.
//
// Bounded means we must decide what to do when it fills. The policy here is
// asymmetric on purpose:
//
//   - Partials are dropped. A partial is superseded by its final within a
//     second, so dropping it costs a moment of portal jitter and nothing else.
//   - Finals block the reader, up to BlockTimeout. Blocking propagates through
//     the WebSocket read loop into TCP backpressure and eventually into the
//     edge agent's send queue, which is exactly where the pressure should
//     land: the producer slows down instead of the buffer growing.
//   - Finals that exceed BlockTimeout are counted and dropped, so one wedged
//     broker cannot hold every connection open forever.
type Publisher struct {
	writer       *kafka.Writer
	topic        string
	queue        chan job
	wg           sync.WaitGroup
	log          *slog.Logger
	blockTimeout time.Duration
	closeOnce    sync.Once
}

type job struct {
	key     []byte
	value   []byte
	isFinal bool
}

type Config struct {
	Brokers      []string
	Topic        string
	QueueSize    int
	Workers      int
	BatchSize    int
	BatchTimeout time.Duration
	BlockTimeout time.Duration
	RequiredAcks int
}

func (c *Config) withDefaults() {
	if c.QueueSize <= 0 {
		c.QueueSize = 4096
	}
	if c.Workers <= 0 {
		c.Workers = 4
	}
	if c.BatchSize <= 0 {
		c.BatchSize = 64
	}
	if c.BatchTimeout <= 0 {
		// Low, because this sits inside a sub-2s end-to-end latency budget.
		// Kafka's default of 1s would spend half the budget in the batcher.
		c.BatchTimeout = 20 * time.Millisecond
	}
	if c.BlockTimeout <= 0 {
		c.BlockTimeout = 2 * time.Second
	}
	if c.RequiredAcks == 0 {
		c.RequiredAcks = int(kafka.RequireOne)
	}
}

func New(cfg Config, log *slog.Logger) *Publisher {
	cfg.withDefaults()

	p := &Publisher{
		writer: &kafka.Writer{
			Addr:  kafka.TCP(cfg.Brokers...),
			Topic: cfg.Topic,
			// Partition by call_id so every segment of one call lands on one
			// partition and is consumed in order. Sliding-window analysis is
			// order-dependent; round-robin partitioning would silently
			// reorder a conversation.
			Balancer:     &kafka.Hash{},
			BatchSize:    cfg.BatchSize,
			BatchTimeout: cfg.BatchTimeout,
			RequiredAcks: kafka.RequiredAcks(cfg.RequiredAcks),
			Async:        false,
			Compression:  kafka.Snappy,
			MaxAttempts:  3,
			ErrorLogger: kafka.LoggerFunc(func(msg string, args ...any) {
				log.Warn("kafka writer", slog.String("detail", obs.Scrub(sprintf(msg, args...))))
			}),
		},
		topic:        cfg.Topic,
		queue:        make(chan job, cfg.QueueSize),
		log:          log,
		blockTimeout: cfg.BlockTimeout,
	}

	for i := 0; i < cfg.Workers; i++ {
		p.wg.Add(1)
		go p.worker()
	}
	return p
}

// Publish enqueues a segment. Returns an error only when the segment was shed.
func (p *Publisher) Publish(ctx context.Context, seg *contracts.TranscriptSegment) error {
	value, err := json.Marshal(seg)
	if err != nil {
		return err
	}
	j := job{key: []byte(seg.CallID), value: value, isFinal: seg.IsFinal}

	select {
	case p.queue <- j:
		obs.QueueDepth.Set(float64(len(p.queue)))
		return nil
	default:
	}

	// Queue is full. Shed partials immediately; make finals wait.
	if !seg.IsFinal {
		obs.BackpressureEvents.WithLabelValues("dropped_partial").Inc()
		return nil
	}

	obs.BackpressureEvents.WithLabelValues("blocked_final").Inc()
	timer := time.NewTimer(p.blockTimeout)
	defer timer.Stop()

	select {
	case p.queue <- j:
		obs.QueueDepth.Set(float64(len(p.queue)))
		return nil
	case <-timer.C:
		obs.BackpressureEvents.WithLabelValues("dropped_final").Inc()
		return errors.New("publish queue full: shed a final segment after blocking")
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *Publisher) worker() {
	defer p.wg.Done()
	for j := range p.queue {
		obs.QueueDepth.Set(float64(len(p.queue)))
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err := p.writer.WriteMessages(ctx, kafka.Message{Key: j.key, Value: j.value})
		cancel()
		obs.PublishLatency.Observe(time.Since(start).Seconds())

		if err != nil {
			kind := "write"
			if errors.Is(err, context.DeadlineExceeded) {
				kind = "timeout"
			}
			obs.PublishErrors.WithLabelValues(kind).Inc()
			p.log.Error("kafka publish failed",
				slog.String("topic", p.topic), slog.String("error", err.Error()))
			continue
		}
		obs.SegmentsPublished.WithLabelValues(p.topic).Inc()
	}
}

// Close drains the queue and shuts the writer down.
func (p *Publisher) Close() error {
	var err error
	p.closeOnce.Do(func() {
		close(p.queue)
		p.wg.Wait()
		err = p.writer.Close()
	})
	return err
}

func (p *Publisher) QueueDepth() int { return len(p.queue) }

func sprintf(format string, args ...any) string {
	if len(args) == 0 {
		return format
	}
	return format + " " + jsonish(args)
}

func jsonish(args []any) string {
	b, err := json.Marshal(args)
	if err != nil {
		return ""
	}
	return string(b)
}
