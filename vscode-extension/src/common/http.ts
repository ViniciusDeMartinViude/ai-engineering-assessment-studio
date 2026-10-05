export type JsonRequester = <T>(path: string, init?: RequestInit) => Promise<T>;

export class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: unknown
  ) {
    super(message);
  }
}

export function createJsonRequester(baseUrl: string, timeoutMs = 10000): JsonRequester {
  const root = baseUrl.replace(/\/+$/, "");
  return async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(`${root}${path}`, {
        ...init,
        headers: {
          Accept: "application/json",
          ...(init.body ? { "Content-Type": "application/json" } : {}),
          ...(init.headers ?? {})
        },
        signal: controller.signal
      });
      const text = await response.text();
      const body = text ? JSON.parse(text) : {};
      if (!response.ok) {
        const message = typeof body?.error === "string" ? body.error : `HTTP ${response.status}`;
        throw new HttpError(message, response.status, body);
      }
      return body as T;
    } finally {
      clearTimeout(timer);
    }
  };
}
