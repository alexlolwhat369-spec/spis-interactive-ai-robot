"""Choose local music from explicit requests and robot events only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Track:
    title: str
    category: str
    path: str


class MusicSelector:
    """Deterministic selector. It never receives face or identity information."""

    def __init__(self, tracks: Iterable[Track]) -> None:
        self._tracks = tuple(tracks)
        if not self._tracks:
            raise ValueError("At least one track is required.")

    @classmethod
    def from_file(cls, path: Path) -> "MusicSelector":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(Track(**track) for track in data["tracks"])

    def choose(
        self,
        *,
        requested_category: str | None = None,
        gesture: str | None = None,
        game_won: bool = False,
    ) -> Track:
        if requested_category:
            category = requested_category
        elif game_won or gesture == "thumbs_up":
            category = "celebration"
        elif gesture == "heart":
            category = "warm"
        else:
            category = "calm"
        for track in self._tracks:
            if track.category == category:
                return track
        return self._tracks[0]

