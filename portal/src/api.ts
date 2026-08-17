import { useEffect, useRef, useState } from 'react'
import type { LiveEvent, Signal, TranscriptSegment } from './types'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`)
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

/** Fetch-on-mount with loading and error states, refreshed when deps change. */
export function useQuery<T>(path: string, deps: unknown[] = []): {
  data: T | null
  error: string | null
  loading: boolean
} {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    get<T>(path)
      .then((result) => {
        if (!cancelled) setData(result)
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, error, loading }
}

export interface LiveState {
  connected: boolean
  kafkaConnected: boolean
  segments: TranscriptSegment[]
  signals: Signal[]
}

const MAX_SEGMENTS = 400
const MAX_SIGNALS = 200

/**
 * Subscribe to the live pipeline feed.
 *
 * Reconnects with backoff, because a supervisor leaving the tab open
 * overnight should find a working feed in the morning rather than a silently
 * dead socket. Only final segments are displayed: a partial is superseded
 * within a second, and rendering them makes the transcript flicker.
 */
export function useLiveFeed(): LiveState {
  const [state, setState] = useState<LiveState>({
    connected: false,
    kafkaConnected: false,
    segments: [],
    signals: [],
  })
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let closed = false
    let backoff = 500
    let timer: number | undefined

    const apply = (event: LiveEvent) => {
      setState((prev) => {
        if (event.type === 'segment') {
          if (!event.data.is_final) return prev
          const segments = [...prev.segments, event.data].slice(-MAX_SEGMENTS)
          return { ...prev, segments }
        }
        if (event.type === 'signal') {
          const signals = [...prev.signals, event.data].slice(-MAX_SIGNALS)
          return { ...prev, signals }
        }
        return prev
      })
    }

    const connect = () => {
      if (closed) return
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
      const url = BASE
        ? `${BASE.replace(/^http/, 'ws')}/api/live`
        : `${protocol}://${location.host}/api/live`
      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.onopen = () => {
        backoff = 500
        setState((prev) => ({ ...prev, connected: true }))
      }

      socket.onmessage = (message) => {
        let event: LiveEvent
        try {
          event = JSON.parse(message.data as string) as LiveEvent
        } catch {
          return
        }
        if (event.type === 'hello') {
          setState((prev) => ({ ...prev, kafkaConnected: event.kafka_connected }))
          event.replay.forEach(apply)
          return
        }
        apply(event)
      }

      socket.onclose = () => {
        setState((prev) => ({ ...prev, connected: false }))
        if (closed) return
        timer = window.setTimeout(connect, backoff)
        backoff = Math.min(backoff * 2, 10_000)
      }

      socket.onerror = () => socket.close()
    }

    connect()
    return () => {
      closed = true
      if (timer) window.clearTimeout(timer)
      socketRef.current?.close()
    }
  }, [])

  return state
}
