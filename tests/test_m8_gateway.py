from __future__ import annotations

import threading
import time
import unittest

from ai_assessment.services.ai_gateway import GatewayClient, SelectedContext
from server.gateway import (
    AuthenticationError,
    BudgetError,
    ConcurrentRequestError,
    GatewayConfig,
    GatewayService,
    IdempotencyError,
    InputLimitError,
    ProviderResult,
    ProviderTimeout,
)


class CountingProvider:
    def __init__(self, *, prompt_tokens: int = 1000) -> None:
        self.calls = 0
        self.prompt_tokens = prompt_tokens

    def complete(self, **_kwargs) -> ProviderResult:
        self.calls += 1
        return ProviderResult("safe practice answer", f"provider-{self.calls}", self.prompt_tokens, 0)


class TimeoutProvider:
    def complete(self, **_kwargs) -> ProviderResult:
        raise ProviderTimeout("provider did not finish")


class BlockingProvider:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def complete(self, **_kwargs) -> ProviderResult:
        self.started.set()
        self.release.wait(2)
        return ProviderResult("released", "provider-block", 1000, 0)


def config(**overrides) -> GatewayConfig:
    values = {
        "approved_models": {"practice-mock": (10.0, 0.0)},
        "session_allowance_cents": 100,
        "global_allowance_cents": 100,
        "max_input_chars": 1000,
        "max_context_chars": 1000,
        "max_output_tokens": 20,
        "max_requests_per_minute": 20,
        "reservation_ttl_seconds": 1,
    }
    values.update(overrides)
    return GatewayConfig(**values)


class M8GatewayTests(unittest.TestCase):
    def provision(self, gateway: GatewayService, session: str = "S1", token: str = "practice-token-S1") -> str:
        gateway.provision_session(session, token, mode="practice")
        return token

    def test_authentication_and_assessment_trust_boundary(self) -> None:
        gateway = GatewayService(":memory:", config=config())
        with self.assertRaises(AuthenticationError):
            gateway.allowance("")
        with self.assertRaises(Exception):
            gateway.provision_session("A1", "assessment-token", mode="assessment")
        gateway.provision_session("A1", "trusted-assessment", mode="assessment", trusted=True)
        self.assertEqual(gateway.allowance("trusted-assessment")["mode"], "assessment")

    def test_session_and_global_budget_caps(self) -> None:
        provider = CountingProvider()
        gateway = GatewayService(":memory:", provider=provider, config=config(session_allowance_cents=10, global_allowance_cents=20))
        token_a = self.provision(gateway, "A", "practice-token-A")
        token_b = self.provision(gateway, "B", "practice-token-B")
        gateway.send_message(token_a, message="one", idempotency_key="a1")
        with self.assertRaises(BudgetError):
            gateway.send_message(token_a, message="two", idempotency_key="a2")
        gateway.send_message(token_b, message="one", idempotency_key="b1")
        with self.assertRaises(BudgetError):
            gateway.send_message(token_b, message="two", idempotency_key="b2")

    def test_concurrent_reservation_is_rejected_without_overspend(self) -> None:
        provider = BlockingProvider()
        gateway = GatewayService(":memory:", provider=provider, config=config(max_concurrent_per_session=1))
        token = self.provision(gateway)
        result: list[dict[str, object]] = []
        thread = threading.Thread(target=lambda: result.append(gateway.send_message(token, message="first", idempotency_key="first")))
        thread.start()
        self.assertTrue(provider.started.wait(1))
        with self.assertRaises(ConcurrentRequestError):
            gateway.send_message(token, message="second", idempotency_key="second")
        provider.release.set()
        thread.join(2)
        self.assertEqual(result[0]["status"], "completed")

    def test_idempotency_returns_same_completed_request_without_second_provider_call(self) -> None:
        provider = CountingProvider()
        gateway = GatewayService(":memory:", provider=provider, config=config())
        token = self.provision(gateway)
        first = gateway.send_message(token, message="same", idempotency_key="fixed-key")
        second = gateway.send_message(token, message="same", idempotency_key="fixed-key")
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertEqual(provider.calls, 1)

    def test_timeout_is_uncertain_then_stale_reservation_can_be_reaped(self) -> None:
        gateway = GatewayService(":memory:", provider=TimeoutProvider(), config=config(reservation_ttl_seconds=0))
        token = self.provision(gateway)
        with self.assertRaises(ProviderTimeout):
            gateway.send_message(token, message="wait", idempotency_key="timeout-key")
        self.assertEqual(gateway.history(token)[0]["status"], "uncertain")
        self.assertGreaterEqual(gateway.reap_stale(), 1)
        self.assertEqual(gateway.history(token)[0]["status"], "expired")

    def test_input_bounds_and_provider_secret_non_disclosure(self) -> None:
        provider = CountingProvider()
        gateway = GatewayService(":memory:", provider=provider, config=config(max_input_chars=4))
        token = self.provision(gateway)
        with self.assertRaises(InputLimitError):
            gateway.send_message(token, message="too long", idempotency_key="bounded")
        self.assertNotIn(token, repr(gateway.allowance(token)))
        self.assertNotIn("OPENAI_API_KEY", repr(gateway.history(token)))

    def test_transcripts_are_isolated_by_authenticated_session(self) -> None:
        gateway = GatewayService(":memory:", provider=CountingProvider(), config=config())
        token_a = self.provision(gateway, "A", "practice-token-A")
        token_b = self.provision(gateway, "B", "practice-token-B")
        gateway.send_message(token_a, message="only A", idempotency_key="a")
        gateway.send_message(token_b, message="only B", idempotency_key="b")
        self.assertEqual(len(gateway.history(token_a)), 1)
        self.assertEqual(gateway.history(token_a)[0]["message_text"], "only A")
        self.assertEqual(gateway.history(token_a)[0]["response_text"], "safe practice answer")
        self.assertEqual(len(gateway.history(token_b)), 1)


class M8ClientTests(unittest.TestCase):
    def test_selected_context_is_explicit_and_hashable(self) -> None:
        captured = {}

        def transport(method, path, payload, headers):
            captured.update(method=method, path=path, payload=payload, headers=headers)
            return {"request_id": headers["X-Request-ID"], "status": "completed", "answer": "ok"}

        context = SelectedContext.create("error_traceback", "Traceback: selected only")
        client = GatewayClient("http://gateway.test", "practice-token", transport=transport)
        result = client.send_message("Explain this", context, idempotency_key="same-request")
        self.assertEqual(result["answer"], "ok")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/api/v1/messages")
        self.assertEqual(captured["payload"]["context"], {"type": "error_traceback", "text": "Traceback: selected only"})
        self.assertEqual(captured["headers"]["Idempotency-Key"], "same-request")
        self.assertEqual(len(context.sha256), 64)
        self.assertNotIn("OPENAI_API_KEY", repr(captured))


if __name__ == "__main__":
    unittest.main()
