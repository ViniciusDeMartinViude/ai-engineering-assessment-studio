from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIContextBuffer:
    kind: str = "selected_text"
    text: str = ""
    source: str = ""

    def set(self, kind: str, text: str, source: str) -> None:
        self.kind = kind
        self.text = text
        self.source = source

    def clear(self) -> None:
        self.kind = "selected_text"
        self.text = ""
        self.source = ""
