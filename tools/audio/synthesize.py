"""Synthesise real call audio from the golden corpus.

Run: ``python -m tools.audio.synthesize --calls eval/golden/calls --out data/audio``

Produces, per call, a 16 kHz mono 16-bit PCM WAV -- the format faster-whisper
wants, so nothing has to resample at stream time -- plus a manifest of turn
boundaries. The manifest is the diarisation ground truth: the edge agent
derives speaker labels from it rather than pretending to run a diariser, which
is an explicit non-goal (see DESIGN.md).

Backends, tried in order: macOS ``say``, then ``piper``, then ``espeak-ng``.
The container image installs piper so the pipeline produces genuine audio on
Linux too; no stage of the happy path reads canned audio from disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from tools.audio.speech import to_speech

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1
GAP_MS = 260       # pause between turns
LEAD_IN_MS = 300   # silence before the first word
TAIL_MS = 400


@dataclass(frozen=True)
class Voice:
    agent: str
    customer: str


class TTSBackend:
    name = "abstract"

    def available(self) -> bool:
        raise NotImplementedError

    def voices_for(self, call_id: str) -> Voice:
        raise NotImplementedError

    def synth(self, text: str, voice: str, dest: Path) -> None:
        """Write a 16 kHz mono 16-bit WAV of ``text`` to ``dest``."""
        raise NotImplementedError


class MacSayBackend(TTSBackend):
    """macOS ``say``. Emits the target format directly, no resampling step."""

    name = "macos-say"
    AGENT_VOICES = ("Daniel", "Fred")
    CUSTOMER_VOICES = ("Samantha", "Moira", "Karen")

    def available(self) -> bool:
        return sys.platform == "darwin" and shutil.which("say") is not None

    def voices_for(self, call_id: str) -> Voice:
        h = int(hashlib.sha256(call_id.encode()).hexdigest(), 16)
        return Voice(
            agent=self.AGENT_VOICES[h % len(self.AGENT_VOICES)],
            customer=self.CUSTOMER_VOICES[(h >> 8) % len(self.CUSTOMER_VOICES)],
        )

    def synth(self, text: str, voice: str, dest: Path) -> None:
        subprocess.run(
            ["say", "-v", voice, "-o", str(dest),
             "--data-format=LEI16@16000", "--file-format=WAVE", text],
            check=True, capture_output=True,
        )


class PiperBackend(TTSBackend):
    """piper-tts. The Linux/container path; ONNX voices, CPU only."""

    name = "piper"
    AGENT_VOICES = ("en_US-ryan-medium",)
    CUSTOMER_VOICES = ("en_US-amy-medium", "en_US-lessac-medium")

    def available(self) -> bool:
        return shutil.which("piper") is not None

    def voices_for(self, call_id: str) -> Voice:
        h = int(hashlib.sha256(call_id.encode()).hexdigest(), 16)
        return Voice(
            agent=self.AGENT_VOICES[h % len(self.AGENT_VOICES)],
            customer=self.CUSTOMER_VOICES[(h >> 8) % len(self.CUSTOMER_VOICES)],
        )

    def synth(self, text: str, voice: str, dest: Path) -> None:
        model = Path(f"/opt/piper/voices/{voice}.onnx")
        subprocess.run(
            ["piper", "--model", str(model), "--output_file", str(dest)],
            input=text.encode(), check=True, capture_output=True,
        )
        _force_format(dest)


class EspeakBackend(TTSBackend):
    """espeak-ng. Last resort: intelligible, robotic, and a genuinely harder
    ASR target -- useful as a lower bound rather than as the demo voice."""

    name = "espeak-ng"

    def available(self) -> bool:
        return shutil.which("espeak-ng") is not None

    def voices_for(self, call_id: str) -> Voice:
        return Voice(agent="en-gb", customer="en-us")

    def synth(self, text: str, voice: str, dest: Path) -> None:
        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", "155", "-w", str(dest), text],
            check=True, capture_output=True,
        )
        _force_format(dest)


def _force_format(path: Path) -> None:
    """Resample/convert in place to 16 kHz mono 16-bit if the backend drifted."""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (
            SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH
        ):
            return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{path} is not 16kHz mono 16-bit and ffmpeg is unavailable to convert it"
        )
    tmp = path.with_suffix(".conv.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-sample_fmt", "s16", str(tmp)],
        check=True,
    )
    tmp.replace(path)


def _silence(ms: int) -> bytes:
    return b"\x00" * int(SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS * ms / 1000)


def _read_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != CHANNELS:
            raise RuntimeError(f"unexpected format in {path}")
        return w.readframes(w.getnframes())


def _peak_dbfs(pcm: bytes) -> float:
    """Peak level, as a cheap guard against a backend writing silence."""
    if not pcm:
        return -120.0
    count = len(pcm) // 2
    peak = max(abs(v) for v in struct.unpack(f"<{count}h", pcm[: count * 2]))
    if peak == 0:
        return -120.0
    import math

    return 20 * math.log10(peak / 32768.0)


def pick_backend(preferred: str | None = None) -> TTSBackend:
    backends = [MacSayBackend(), PiperBackend(), EspeakBackend()]
    if preferred:
        for b in backends:
            if b.name == preferred:
                if not b.available():
                    raise RuntimeError(f"requested TTS backend {preferred!r} is unavailable")
                return b
        raise RuntimeError(f"unknown TTS backend {preferred!r}")
    for b in backends:
        if b.available():
            return b
    raise RuntimeError(
        "no TTS backend available; install piper or espeak-ng, or run on macOS"
    )


def synthesize_call(call: dict, out_dir: Path, backend: TTSBackend) -> dict:
    call_id = call["call_id"]
    voices = backend.voices_for(call_id)
    pcm = bytearray(_silence(LEAD_IN_MS))
    turns: list[dict] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for turn in call["turns"]:
            spoken = to_speech(turn["text"])
            voice = voices.agent if turn["speaker"] == "agent" else voices.customer
            piece = tmp / f"t{turn['idx']:03d}.wav"
            backend.synth(spoken, voice, piece)
            frames = _read_frames(piece)

            start_ms = int(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000)
            pcm += frames
            end_ms = int(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000)
            pcm += _silence(GAP_MS)

            turns.append({
                "idx": turn["idx"],
                "speaker": turn["speaker"],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": turn["text"],
                "spoken_text": spoken,
                "voice": voice,
            })

    pcm += _silence(TAIL_MS)

    wav_path = out_dir / f"{call_id}.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(pcm))

    peak = _peak_dbfs(bytes(pcm))
    if peak < -50.0:
        raise RuntimeError(f"{call_id}: synthesised audio is effectively silent ({peak:.1f} dBFS)")

    manifest = {
        "call_id": call_id,
        "audio": wav_path.name,
        "sample_rate": SAMPLE_RATE,
        "duration_ms": int(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000),
        "backend": backend.name,
        "peak_dbfs": round(peak, 1),
        "agent_id": call["agent_id"],
        "turns": turns,
    }
    (out_dir / f"{call_id}.turns.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesise call audio from the golden corpus.")
    ap.add_argument("--calls", type=Path, default=Path("eval/golden/calls"))
    ap.add_argument("--out", type=Path, default=Path("data/audio"))
    ap.add_argument("--backend", default=None, help="macos-say | piper | espeak-ng")
    ap.add_argument("--limit", type=int, default=0, help="synthesise only the first N calls")
    ap.add_argument("--only", default=None, help="synthesise one call_id")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    backend = pick_backend(args.backend)

    paths = sorted(args.calls.glob("*.json"))
    if args.only:
        paths = [p for p in paths if p.stem == args.only]
    if args.limit:
        paths = paths[: args.limit]

    total_ms = 0
    for i, p in enumerate(paths, 1):
        call = json.loads(p.read_text())
        m = synthesize_call(call, args.out, backend)
        total_ms += m["duration_ms"]
        print(f"[{i}/{len(paths)}] {m['call_id']:52s} {m['duration_ms']/1000:6.1f}s "
              f"peak {m['peak_dbfs']:6.1f} dBFS")

    print(f"\nbackend={backend.name}  calls={len(paths)}  total={total_ms/1000/60:.1f} min")


if __name__ == "__main__":
    main()
