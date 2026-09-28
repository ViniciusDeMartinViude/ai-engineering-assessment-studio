from __future__ import annotations

import argparse
import os

import uvicorn

from .gateway import GatewayConfig, GatewayService, FakeProvider, OpenAIProvider, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Organizer-side AI gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default="gateway.sqlite3")
    parser.add_argument("--practice-session", help="Create an explicitly labelled practice session before serving")
    parser.add_argument("--practice-token", help="Operator-provisioned practice token; never an OpenAI key")
    parser.add_argument("--provider", choices=("fake", "openai"), default=os.environ.get("AI_GATEWAY_PROVIDER", "fake"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.provider == "fake":
        config = GatewayConfig.practice()
        provider = FakeProvider()
    else:
        model = os.environ.get("AI_GATEWAY_MODEL")
        if not model:
            raise SystemExit("AI_GATEWAY_MODEL must be set for the server-only live provider")
        config = GatewayConfig(approved_models={model: (float(os.environ.get("AI_GATEWAY_INPUT_CENTS_PER_1K", "1")), float(os.environ.get("AI_GATEWAY_OUTPUT_CENTS_PER_1K", "1")))})
        provider = OpenAIProvider()
    service = GatewayService(args.db, provider=provider, config=config)
    if args.practice_session:
        token = service.create_practice_session(args.practice_session, credential=args.practice_token)
        print(f"practice_session={args.practice_session} practice_token={token}")
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
