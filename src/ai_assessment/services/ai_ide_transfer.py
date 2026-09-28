from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AIAskRequest:
    action: str
    question: str
    source_path: str | None
    code: str
    file_hash: str
    selection_hash: str
    has_selection: bool


@dataclass(frozen=True)
class CodeBlock:
    block_id: str
    language: str
    code: str
    code_hash: str
    suggested_path: str | None = None


@dataclass(frozen=True)
class IDEHandoff:
    response_id: str
    block: CodeBlock
    source_path: str | None = None
    source_file_hash: str | None = None
    source_selection_hash: str | None = None


_FENCED_BLOCK = re.compile(r"```(?P<language>[^\r\n`]*)\r?\n(?P<code>.*?)(?:\r?\n)?```", re.DOTALL)


def extract_code_blocks(response: str) -> tuple[CodeBlock, ...]:
    blocks: list[CodeBlock] = []
    for index, match in enumerate(_FENCED_BLOCK.finditer(response), start=1):
        language = match.group("language").strip().lower() or "text"
        code = match.group("code")
        blocks.append(CodeBlock(f"block-{index}", language, code, sha256_text(code)))
    return tuple(blocks)


class AIIdeCoordinator(QObject):
    """Application-level typed bridge between the IDE and AI pages."""

    ask_requested = Signal(object)
    handoff_requested = Signal(object)

    def request_ai(self, request: AIAskRequest) -> None:
        self.ask_requested.emit(request)

    def request_ide_handoff(self, handoff: IDEHandoff) -> None:
        self.handoff_requested.emit(handoff)
