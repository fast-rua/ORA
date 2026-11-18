// The page's requests to its run's web server (server/app.ts), through the `ora web` gateway.

type Json = any;

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function call(path: string, body?: Json): Promise<Json> {
  const response = await fetch(path, body === undefined ? { cache: "no-store" } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), cache: "no-store",
  });
  let data: Json = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) throw new ApiError(response.status, String(data?.error ?? `${path} answered ${response.status}`));
  return data;
}

/** Server-sent events of the run; reconnects by itself (the browser's EventSource). */
export function subscribe(onEvent: (event: Json) => void, onConnected: (connected: boolean) => void): () => void {
  const source = new EventSource("/api/events");
  source.onopen = () => onConnected(true);
  source.onerror = () => onConnected(false);
  source.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data));
    } catch {
      // a torn event; the next snapshot repairs the page
    }
  };
  return () => source.close();
}
