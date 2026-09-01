"""Small, privacy-conscious diagnostics for one robot interaction at a time."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class TurnSnapshot:
    sequence: int = 0
    heard: str = ""
    route: str = "idle"
    action: str = "none"
    reaction: str = "idle"
    reply: str = ""
    provider_error: str | None = None
    mic_peak: float = 0.0
    mic_average: float = 0.0
    transcript_source: str = "none"


class TurnDiagnostics:
    """Expose the latest decision and optionally append text-only JSONL records."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path
        self._snapshot = TurnSnapshot()
        self._lock = threading.Lock()

    def begin(self) -> None:
        with self._lock:
            self._snapshot = TurnSnapshot(sequence=self._snapshot.sequence + 1, route="listening")

    def heard(
        self,
        text: str,
        *,
        mic_peak: float = 0.0,
        mic_average: float = 0.0,
        transcript_source: str = "unknown",
    ) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                heard=text,
                route="transcribing",
                mic_peak=mic_peak,
                mic_average=mic_average,
                transcript_source=transcript_source,
            )

    def complete(
        self,
        *,
        route: str,
        action: str,
        reaction: str,
        reply: str,
        provider_error: str | None = None,
    ) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                route=route,
                action=action,
                reaction=reaction,
                reply=reply,
                provider_error=provider_error,
            )
            snapshot = self._snapshot
        self._append(snapshot)

    def no_input(self, *, mic_peak: float = 0.0, mic_average: float = 0.0) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                route="no_input",
                mic_peak=mic_peak,
                mic_average=mic_average,
                transcript_source="none",
            )
            snapshot = self._snapshot
        self._append(snapshot)

    def current(self) -> TurnSnapshot:
        with self._lock:
            return self._snapshot

    def _append(self, snapshot: TurnSnapshot) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": time.time(), **asdict(snapshot)}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
