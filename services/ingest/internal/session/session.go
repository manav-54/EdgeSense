// Package session holds per-call state and idempotency keys in Redis.
package session

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/manav-54/edgesense/ingest/internal/obs"
)

// IdempotencyTTL bounds how long a seen-key is remembered.
//
// Sized to comfortably exceed the longest plausible call plus a client's
// retry window. Longer costs memory; shorter risks a genuine duplicate after
// a reconnect being republished, which would double-count the segment in
// every downstream aggregate.
const IdempotencyTTL = 6 * time.Hour

// SessionTTL expires call state after the call can no longer be live.
const SessionTTL = 12 * time.Hour

type Store struct {
	rdb    *redis.Client
	log    *slog.Logger
	failed bool // sticky: logged once, then we stop shouting about it
}

type State struct {
	CallID     string `redis:"call_id"`
	AgentID    string `redis:"agent_id"`
	FirstSeen  string `redis:"first_seen"`
	LastSeen   string `redis:"last_seen"`
	LastSeq    int64  `redis:"last_seq"`
	Segments   int64  `redis:"segments"`
	Duplicates int64  `redis:"duplicates"`
}

func New(addr, password string, db int, log *slog.Logger) *Store {
	return &Store{
		rdb: redis.NewClient(&redis.Options{
			Addr:         addr,
			Password:     password,
			DB:           db,
			DialTimeout:  2 * time.Second,
			ReadTimeout:  500 * time.Millisecond,
			WriteTimeout: 500 * time.Millisecond,
			PoolSize:     32,
		}),
		log: log,
	}
}

func (s *Store) Ping(ctx context.Context) error {
	return s.rdb.Ping(ctx).Err()
}

func (s *Store) Close() error { return s.rdb.Close() }

// SeenBefore reports whether this idempotency key has already been accepted.
//
// **Fails open.** If Redis is unavailable the segment is treated as new and
// allowed through. That is the deliberate choice: a duplicate segment
// double-counts one row in an aggregate, whereas dropping segments because a
// cache is down silently loses live call content that cannot be recovered.
// The failure is counted so a Redis outage is visible as a dedupe-coverage
// gap rather than as silent data loss.
func (s *Store) SeenBefore(ctx context.Context, key string) bool {
	ok, err := s.rdb.SetNX(ctx, "es:idem:"+key, 1, IdempotencyTTL).Result()
	if err != nil {
		obs.SessionErrors.WithLabelValues("setnx").Inc()
		if !s.failed {
			s.log.WarnContext(ctx, "idempotency check unavailable; failing open",
				slog.String("error", err.Error()))
			s.failed = true
		}
		return false
	}
	s.failed = false
	return !ok // SetNX returns true when the key was newly set
}

// Touch records activity for a call. Best-effort: a failure here loses a
// dashboard row, not call data, so it is logged at debug and moved past.
func (s *Store) Touch(ctx context.Context, callID, agentID string, seq int64, duplicate bool) {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	key := "es:call:" + callID

	pipe := s.rdb.TxPipeline()
	pipe.HSetNX(ctx, key, "call_id", callID)
	pipe.HSetNX(ctx, key, "first_seen", now)
	pipe.HSet(ctx, key, "last_seen", now)
	if agentID != "" {
		pipe.HSet(ctx, key, "agent_id", agentID)
	}
	if duplicate {
		pipe.HIncrBy(ctx, key, "duplicates", 1)
	} else {
		pipe.HIncrBy(ctx, key, "segments", 1)
		pipe.HSet(ctx, key, "last_seq", seq)
	}
	pipe.Expire(ctx, key, SessionTTL)
	pipe.SAdd(ctx, "es:calls:active", callID)
	pipe.Expire(ctx, "es:calls:active", SessionTTL)

	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		obs.SessionErrors.WithLabelValues("touch").Inc()
		s.log.DebugContext(ctx, "session touch failed",
			slog.String("call_id", callID), slog.String("error", err.Error()))
	}
}

func (s *Store) Get(ctx context.Context, callID string) (*State, error) {
	vals, err := s.rdb.HGetAll(ctx, "es:call:"+callID).Result()
	if err != nil {
		return nil, err
	}
	if len(vals) == 0 {
		return nil, fmt.Errorf("no session for call %q", callID)
	}
	st := &State{
		CallID:    vals["call_id"],
		AgentID:   vals["agent_id"],
		FirstSeen: vals["first_seen"],
		LastSeen:  vals["last_seen"],
	}
	fmt.Sscanf(vals["last_seq"], "%d", &st.LastSeq)
	fmt.Sscanf(vals["segments"], "%d", &st.Segments)
	fmt.Sscanf(vals["duplicates"], "%d", &st.Duplicates)
	return st, nil
}

// EndCall marks a call finished so the worker knows to run post-call analysis.
func (s *Store) EndCall(ctx context.Context, callID string) {
	pipe := s.rdb.TxPipeline()
	pipe.HSet(ctx, "es:call:"+callID, "ended_at", time.Now().UTC().Format(time.RFC3339Nano))
	pipe.SRem(ctx, "es:calls:active", callID)
	if _, err := pipe.Exec(ctx); err != nil {
		obs.SessionErrors.WithLabelValues("end").Inc()
	}
}
