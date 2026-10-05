import { JsonRequester } from "../common/http";
import { GatewayAllowance, GatewayResponse, SelectedAIContext } from "../common/types";

export class AiClient {
  constructor(
    private readonly requestJson: JsonRequester,
    private readonly credential: string
  ) {}

  allowance(): Promise<GatewayAllowance> {
    return this.requestJson<GatewayAllowance>("/api/v1/allowance", {
      headers: this.headers()
    });
  }

  sendMessage(message: string, context: SelectedAIContext, idempotencyKey: string): Promise<GatewayResponse> {
    return this.requestJson<GatewayResponse>("/api/v1/messages", {
      method: "POST",
      headers: this.headers(idempotencyKey),
      body: JSON.stringify({
        message,
        context: {
          type: context.type,
          text: context.text
        }
      })
    });
  }

  private headers(idempotencyKey?: string): Record<string, string> {
    return {
      Authorization: `Bearer ${this.credential}`,
      ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey, "X-Request-ID": idempotencyKey } : {})
    };
  }
}
