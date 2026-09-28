from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


API_VERSION = "m8.gateway.v1"
DEFAULT_MODEL = "practice-mock"
ALLOWED_CONTEXT_TYPES = {"selected_text", "error_traceback", "training_metrics"}


class GatewayError(RuntimeError):
    code = "gateway_error"
    status_code = 400

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id


class AuthenticationError(GatewayError):
    code = "authentication_required"
    status_code = 401


class AuthorizationError(GatewayError):
    code = "assessment_session_not_trusted"
    status_code = 403


class InputLimitError(GatewayError):
    code = "input_limit_exceeded"
    status_code = 413


class BudgetError(GatewayError):
    code = "budget_exhausted"
    status_code = 429


class ConcurrentRequestError(GatewayError):
    code = "concurrent_request_limit"
    status_code = 429


class RateLimitError(GatewayError):
    code = "request_rate_limit"
    status_code = 429


class IdempotencyError(GatewayError):
    code = "idempotency_in_flight"
    status_code = 409


class ProviderTimeout(GatewayError):
    code = "provider_timeout_uncertain"
    status_code = 504


class ProviderFailure(GatewayError):
    code = "provider_failure"
    status_code = 502


@dataclass(frozen=True)
class ProviderResult:
    text: str
    provider_request_id: str
    prompt_tokens: int
    completion_tokens: int


class ProviderAdapter(Protocol):
    def complete(self, *, model: str, message: str, context: dict[str, str], max_output_tokens: int) -> ProviderResult: ...


class FakeProvider:
    """Deterministic practice provider. It never contacts the network."""

    def complete(self, *, model: str, message: str, context: dict[str, str], max_output_tokens: int) -> ProviderResult:
        context_type = context.get("type", "none")
        answer = f"Practice response for {context_type}: {message[:240]}"
        answer = answer[: max_output_tokens * 4]
        return ProviderResult(
            text=answer,
            provider_request_id=f"fake-{uuid.uuid4().hex[:12]}",
            prompt_tokens=max(1, (len(message) + len(context.get("text", ""))) // 4),
            completion_tokens=max(1, len(answer) // 4),
        )


class OpenAIProvider:
    """Server-only Responses API adapter; the key is read from server environment."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY must be configured on the organizer gateway")
        self._timeout = timeout

    def complete(self, *, model: str, message: str, context: dict[str, str], max_output_tokens: int) -> ProviderResult:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key, timeout=self._timeout)
            prompt = message
            if context.get("text"):
                prompt += f"\n\nAttached {context.get('type', 'context')}:\n{context['text']}"
            response = client.responses.create(model=model, input=prompt, max_output_tokens=max_output_tokens)
            usage = getattr(response, "usage", None)
            return ProviderResult(
                text=str(getattr(response, "output_text", "")),
                provider_request_id=str(getattr(response, "id", "unknown")),
                prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            )
        except GatewayError:
            raise
        except TimeoutError as exc:
            raise ProviderTimeout(str(exc)) from exc
        except Exception as exc:
            if "timeout" in type(exc).__name__.lower():
                raise ProviderTimeout(str(exc)) from exc
            raise ProviderFailure(str(exc)) from exc


@dataclass(frozen=True)
class GatewayConfig:
    approved_models: dict[str, tuple[float, float]]
    session_allowance_cents: int = 500
    global_allowance_cents: int = 10_000
    max_input_chars: int = 8_000
    max_context_chars: int = 12_000
    max_output_tokens: int = 800
    max_concurrent_per_session: int = 1
    max_concurrent_global: int = 8
    max_requests_per_minute: int = 12
    reservation_ttl_seconds: int = 300
    transcript_retention_days: int = 30

    @classmethod
    def practice(cls) -> "GatewayConfig":
        return cls(approved_models={DEFAULT_MODEL: (0.0, 0.0)})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class GatewayService:
    def __init__(self, database: str | Path = ":memory:", *, provider: ProviderAdapter | None = None, config: GatewayConfig | None = None) -> None:
        self.config = config or GatewayConfig.practice()
        self.provider = provider or FakeProvider()
        self._lock = threading.RLock()
        self._database = str(database)
        if self._database != ":memory:":
            Path(self._database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._database, timeout=10.0, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 10000")
        if self._database != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._init_database()

    def _init_database(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    credential_hash TEXT NOT NULL UNIQUE,
                    mode TEXT NOT NULL,
                    trusted INTEGER NOT NULL,
                    allowance_cents INTEGER NOT NULL,
                    reserved_cents INTEGER NOT NULL DEFAULT 0,
                    used_cents INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model TEXT NOT NULL,
                    context_type TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    message_hash TEXT NOT NULL,
                    message_text TEXT NOT NULL DEFAULT '',
                    message_chars INTEGER NOT NULL,
                    reserved_cents INTEGER NOT NULL,
                    actual_cents INTEGER NOT NULL DEFAULT 0,
                    provider_request_id TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    response_text TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    UNIQUE(session_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS global_budget (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    allowance_cents INTEGER NOT NULL,
                    reserved_cents INTEGER NOT NULL DEFAULT 0,
                    used_cents INTEGER NOT NULL DEFAULT 0
                );
                INSERT OR IGNORE INTO global_budget(id, allowance_cents) VALUES (1, 10000);
                """
            )
            columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(requests)").fetchall()}
            if "message_text" not in columns:
                self._connection.execute("ALTER TABLE requests ADD COLUMN message_text TEXT NOT NULL DEFAULT ''")
            self._connection.execute("UPDATE global_budget SET allowance_cents = ? WHERE id = 1", (self.config.global_allowance_cents,))

    def provision_session(self, session_id: str, credential: str, *, mode: str = "practice", trusted: bool = False, allowance_cents: int | None = None) -> None:
        if mode == "assessment" and not trusted:
            raise AuthorizationError("Assessment sessions require an organizer-trusted credential")
        if not credential or len(credential) < 8:
            raise AuthenticationError("A server-provisioned session credential is required")
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO sessions(session_id, credential_hash, mode, trusted, allowance_cents) VALUES(?,?,?,?,?)",
                (session_id, sha256_text(credential), mode, int(trusted), allowance_cents if allowance_cents is not None else self.config.session_allowance_cents),
            )

    def create_practice_session(self, session_id: str, *, credential: str | None = None) -> str:
        token = credential or f"practice-{secrets.token_urlsafe(18)}"
        self.provision_session(session_id, token, mode="practice", trusted=False)
        return token

    def _session(self, credential: str, request_id: str | None = None) -> sqlite3.Row:
        if not credential:
            raise AuthenticationError("Bearer session credential required", request_id=request_id)
        row = self._connection.execute("SELECT * FROM sessions WHERE credential_hash = ?", (sha256_text(credential),)).fetchone()
        if row is None:
            raise AuthenticationError("Unknown or expired session credential", request_id=request_id)
        if row["mode"] == "assessment" and not row["trusted"]:
            raise AuthorizationError("Assessment mode requires an organizer-issued trusted credential", request_id=request_id)
        return row

    def allowance(self, credential: str) -> dict[str, Any]:
        row = self._session(credential)
        global_row = self._connection.execute("SELECT * FROM global_budget WHERE id = 1").fetchone()
        return {
            "api_version": API_VERSION,
            "session_id": row["session_id"],
            "mode": row["mode"],
            "model": next(iter(self.config.approved_models)),
            "remaining_cents": max(0, row["allowance_cents"] - row["reserved_cents"] - row["used_cents"]),
            "global_remaining_cents": max(0, global_row["allowance_cents"] - global_row["reserved_cents"] - global_row["used_cents"]),
            "max_output_tokens": self.config.max_output_tokens,
        }

    def history(self, credential: str, limit: int = 50) -> list[dict[str, Any]]:
        row = self._session(credential)
        limit = max(1, min(limit, 100))
        with self._lock:
            self._purge_expired_transcripts_locked()
        records = self._connection.execute(
            "SELECT request_id, model, context_type, context_hash, status, message_text, response_text, actual_cents, prompt_tokens, completion_tokens, created_at, updated_at FROM requests WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (row["session_id"], limit),
        ).fetchall()
        return [dict(record) for record in records]

    def send_message(self, credential: str, *, message: str, context: dict[str, str] | None = None, model: str | None = None, idempotency_key: str | None = None, request_id: str | None = None) -> dict[str, Any]:
        request_id = request_id or f"ai-{uuid.uuid4().hex}"
        context = context or {"type": "selected_text", "text": ""}
        self._validate_input(message, context, request_id)
        session = self._session(credential, request_id)
        model = model or next(iter(self.config.approved_models))
        if model not in self.config.approved_models:
            raise GatewayError("Model is not approved by the organizer", request_id=request_id)
        idempotency_key = idempotency_key or request_id
        context_hash = sha256_text(context.get("text", ""))
        message_hash = sha256_text(message)

        with self._lock:
            self._reap_stale_locked()
            existing = self._connection.execute(
                "SELECT * FROM requests WHERE session_id = ? AND idempotency_key = ?",
                (session["session_id"], idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["status"] == "completed":
                    return self._response_from_row(existing, session)
                raise IdempotencyError("This idempotency key is already in flight or uncertain", request_id=existing["request_id"])
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                active_session = self._connection.execute("SELECT COUNT(*) AS count FROM requests WHERE session_id = ? AND status IN ('reserved','running')", (session["session_id"],)).fetchone()["count"]
                active_global = self._connection.execute("SELECT COUNT(*) AS count FROM requests WHERE status IN ('reserved','running')").fetchone()["count"]
                if active_session >= self.config.max_concurrent_per_session or active_global >= self.config.max_concurrent_global:
                    raise ConcurrentRequestError("Concurrent AI request limit reached", request_id=request_id)
                recent = self._connection.execute("SELECT COUNT(*) AS count FROM requests WHERE session_id = ? AND created_at >= ?", (session["session_id"], datetime.fromtimestamp(time.time() - 60, timezone.utc).isoformat())).fetchone()["count"]
                if recent >= self.config.max_requests_per_minute:
                    raise RateLimitError("AI request rate limit reached", request_id=request_id)
                estimated = self._estimate_cost(model)
                global_row = self._connection.execute("SELECT * FROM global_budget WHERE id = 1").fetchone()
                if session["allowance_cents"] - session["reserved_cents"] - session["used_cents"] < estimated:
                    raise BudgetError("Session AI allowance exhausted", request_id=request_id)
                if global_row["allowance_cents"] - global_row["reserved_cents"] - global_row["used_cents"] < estimated:
                    raise BudgetError("Global AI event budget exhausted", request_id=request_id)
                now = utc_now()
                self._connection.execute(
                    "INSERT INTO requests(request_id, session_id, idempotency_key, status, model, context_type, context_hash, message_hash, message_text, message_chars, reserved_cents, created_at, updated_at, expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (request_id, session["session_id"], idempotency_key, "reserved", model, context["type"], context_hash, message_hash, message, len(message), estimated, now, now, time.time() + self.config.reservation_ttl_seconds),
                )
                self._connection.execute("UPDATE sessions SET reserved_cents = reserved_cents + ? WHERE session_id = ?", (estimated, session["session_id"]))
                self._connection.execute("UPDATE global_budget SET reserved_cents = reserved_cents + ? WHERE id = 1", (estimated,))
                self._connection.execute("UPDATE requests SET status = 'running', updated_at = ? WHERE request_id = ?", (utc_now(), request_id))
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

        try:
            result = self.provider.complete(model=model, message=message, context=context, max_output_tokens=self.config.max_output_tokens)
        except ProviderTimeout as exc:
            with self._lock:
                self._connection.execute("UPDATE requests SET status='uncertain', error_code=?, updated_at=? WHERE request_id=?", (exc.code, utc_now(), request_id))
            raise ProviderTimeout("Provider timed out; reservation remains until organizer cleanup", request_id=request_id) from exc
        except GatewayError as exc:
            self._settle(request_id, status="failed", actual_cents=0, error_code=exc.code)
            raise
        except Exception as exc:
            self._settle(request_id, status="failed", actual_cents=0, error_code="provider_failure")
            raise ProviderFailure(str(exc), request_id=request_id) from exc

        actual = self._actual_cost(model, result.prompt_tokens, result.completion_tokens)
        self._settle(request_id, status="completed", actual_cents=actual, result=result)
        row = self._connection.execute("SELECT * FROM requests WHERE request_id = ?", (request_id,)).fetchone()
        session = self._session(credential, request_id)
        return self._response_from_row(row, session)

    def _validate_input(self, message: str, context: dict[str, str], request_id: str) -> None:
        if not isinstance(context, dict):
            raise InputLimitError("Context must be an object", request_id=request_id)
        if not isinstance(message, str) or not message.strip():
            raise InputLimitError("Message is required", request_id=request_id)
        if len(message) > self.config.max_input_chars:
            raise InputLimitError("Message exceeds the gateway input limit", request_id=request_id)
        if context.get("type") not in ALLOWED_CONTEXT_TYPES:
            raise InputLimitError("Context type is not approved", request_id=request_id)
        if not isinstance(context.get("text", ""), str) or len(context.get("text", "")) > self.config.max_context_chars:
            raise InputLimitError("Attached context exceeds the gateway limit", request_id=request_id)

    def _estimate_cost(self, model: str) -> int:
        input_price, output_price = self.config.approved_models[model]
        return max(1, int((self.config.max_input_chars / 1000) * input_price + (self.config.max_output_tokens / 1000) * output_price))

    def _actual_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> int:
        input_price, output_price = self.config.approved_models[model]
        return max(0, int((prompt_tokens / 1000) * input_price + (completion_tokens / 1000) * output_price))

    def _settle(self, request_id: str, *, status: str, actual_cents: int, error_code: str | None = None, result: ProviderResult | None = None) -> None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM requests WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                return
            reserved = row["reserved_cents"]
            response_text = result.text if result else None
            self._connection.execute(
                "UPDATE requests SET status=?, actual_cents=?, provider_request_id=?, prompt_tokens=?, completion_tokens=?, response_text=?, error_code=?, updated_at=? WHERE request_id=?",
                (status, actual_cents, result.provider_request_id if result else None, result.prompt_tokens if result else 0, result.completion_tokens if result else 0, response_text, error_code, utc_now(), request_id),
            )
            self._connection.execute("UPDATE sessions SET reserved_cents = reserved_cents - ?, used_cents = used_cents + ? WHERE session_id = ?", (reserved, actual_cents, row["session_id"]))
            self._connection.execute("UPDATE global_budget SET reserved_cents = reserved_cents - ?, used_cents = used_cents + ? WHERE id = 1", (reserved, actual_cents))

    def _reap_stale_locked(self) -> None:
        stale = self._connection.execute("SELECT request_id FROM requests WHERE status IN ('reserved','running','uncertain') AND expires_at <= ?", (time.time(),)).fetchall()
        for row in stale:
            self._settle(row["request_id"], status="expired", actual_cents=0, error_code="reservation_expired")

    def _purge_expired_transcripts_locked(self) -> None:
        cutoff = datetime.fromtimestamp(time.time() - self.config.transcript_retention_days * 86400, timezone.utc).isoformat()
        self._connection.execute(
            "UPDATE requests SET message_text = '', response_text = NULL WHERE updated_at < ? AND status IN ('completed','failed','expired')",
            (cutoff,),
        )

    def purge_expired_transcripts(self) -> None:
        with self._lock:
            self._purge_expired_transcripts_locked()

    def reap_stale(self) -> int:
        with self._lock:
            before = self._connection.execute("SELECT COUNT(*) AS count FROM requests WHERE status IN ('reserved','running','uncertain') AND expires_at <= ?", (time.time(),)).fetchone()["count"]
            self._reap_stale_locked()
            return int(before)

    def _response_from_row(self, row: sqlite3.Row, session: sqlite3.Row) -> dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "request_id": row["request_id"],
            "session_id": session["session_id"],
            "status": row["status"],
            "model": row["model"],
            "answer": row["response_text"],
            "usage": {"prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"], "actual_cents": row["actual_cents"]},
            "remaining_cents": max(0, session["allowance_cents"] - session["reserved_cents"] - session["used_cents"]),
            "provider_request_id": row["provider_request_id"],
            "server_timestamp": row["updated_at"],
        }


def create_app(service: GatewayService | None = None):
    try:
        from fastapi import FastAPI, Header, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised by deployment setup
        raise RuntimeError("Install the GUI/server dependencies to run the gateway") from exc

    gateway = service or GatewayService(os.environ.get("AI_GATEWAY_DB", "gateway.sqlite3"), provider=FakeProvider())
    app = FastAPI(title="Assessment Organizer AI Gateway", version=API_VERSION)

    def fail(exc: GatewayError) -> None:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message, "request_id": exc.request_id})

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "api_version": API_VERSION}

    @app.get("/api/v1/allowance")
    def allowance(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        try:
            return gateway.allowance(_bearer(authorization))
        except GatewayError as exc:
            fail(exc)

    @app.get("/api/v1/history")
    def history(authorization: str | None = Header(default=None), limit: int = 50) -> dict[str, Any]:
        try:
            return {"api_version": API_VERSION, "items": gateway.history(_bearer(authorization), limit)}
        except GatewayError as exc:
            fail(exc)

    @app.post("/api/v1/messages")
    def message(body: dict[str, Any], authorization: str | None = Header(default=None), idempotency_key: str | None = Header(default=None), x_request_id: str | None = Header(default=None)) -> dict[str, Any]:
        try:
            return gateway.send_message(_bearer(authorization), message=body.get("message", ""), context=body.get("context"), model=body.get("model"), idempotency_key=idempotency_key, request_id=x_request_id)
        except GatewayError as exc:
            fail(exc)

    return app


def _bearer(value: str | None) -> str:
    if not value or not value.startswith("Bearer "):
        raise AuthenticationError("Bearer session credential required")
    return value[7:].strip()
