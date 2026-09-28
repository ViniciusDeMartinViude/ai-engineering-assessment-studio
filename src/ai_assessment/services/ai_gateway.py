from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class GatewayClientError(RuntimeError):
    def __init__(self, message: str, *, code: str = "gateway_client_error", status: int | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.request_id = request_id


@dataclass(frozen=True)
class SelectedContext:
    kind: str
    text: str
    sha256: str

    @classmethod
    def create(cls, kind: str, text: str) -> "SelectedContext":
        if kind not in {"selected_text", "error_traceback", "training_metrics"}:
            raise ValueError("Unsupported AI context type")
        return cls(kind=kind, text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


class GatewayClient:
    """Candidate-side HTTP client. It has no provider SDK or provider secret."""

    def __init__(self, base_url: str, credential: str, *, timeout: float = 20.0, transport: Callable[..., dict[str, Any]] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.credential = credential
        self.timeout = timeout
        self.transport = transport

    def send_message(self, message: str, context: SelectedContext, *, idempotency_key: str | None = None, request_id: str | None = None) -> dict[str, Any]:
        request_id = request_id or f"client-{uuid.uuid4().hex}"
        payload = {"message": message, "context": {"type": context.kind, "text": context.text}}
        return self._request("POST", "/api/v1/messages", payload, request_id=request_id, idempotency_key=idempotency_key or request_id)

    def allowance(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/allowance")

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._request("GET", f"/api/v1/history?limit={max(1, min(limit, 100))}").get("items", [])

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None, *, request_id: str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.credential}", "Accept": "application/json"}
        if request_id:
            headers["X-Request-ID"] = request_id
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if self.transport is not None:
            return self.transport(method, path, payload, headers)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                data = json.loads(exc.read().decode("utf-8"))
                detail = data.get("detail", data)
                raise GatewayClientError(detail.get("message", str(detail)), code=detail.get("code", "http_error"), status=exc.code, request_id=detail.get("request_id")) from exc
            except json.JSONDecodeError:
                raise GatewayClientError(str(exc), status=exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise GatewayClientError(f"Gateway connection failed: {exc}", code="connection_error") from exc
