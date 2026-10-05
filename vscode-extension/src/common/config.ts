export interface ExtensionConfig {
  agentUrl: string;
  gatewayUrl: string;
  gatewayToken?: string;
}

export interface ConfigReader {
  get<T>(key: string, defaultValue: T): T;
}

export interface EnvironmentReader {
  AI_GATEWAY_URL?: string;
  AI_GATEWAY_SESSION_TOKEN?: string;
}

export function normalizeBaseUrl(value: string | undefined, fallback: string): string {
  const raw = (value ?? "").trim() || fallback;
  return raw.replace(/\/+$/, "");
}

export function resolveConfigValues(settings: ConfigReader, env: EnvironmentReader = process.env): ExtensionConfig {
  const agentUrl = normalizeBaseUrl(settings.get("agentUrl", "http://127.0.0.1:8787"), "http://127.0.0.1:8787");
  const configuredGateway = settings.get("gatewayUrl", "").trim();
  return {
    agentUrl,
    gatewayUrl: normalizeBaseUrl(configuredGateway || env.AI_GATEWAY_URL, ""),
    gatewayToken: env.AI_GATEWAY_SESSION_TOKEN
  };
}
