// The run's live cameras from ORA's world view (ora/orchestrator/world_view.py), as its
// own page (ora/orchestrator/world_view_assets/static/world_view.js) reads them: one status request at a
// time, every camera of a frame fetched and decoded before it is shown, then
// the decode acknowledged, so the terminal can say which frame the browser shows.

type Json = any;

export interface LiveCamera {
  name: string;
  url: string;
}

export interface LiveView {
  /** The world view's phase (waiting, executing, holding, finished, unavailable, error, ...). */
  phase: string;
  message: string;
  frame: Json | null;
  cameras: LiveCamera[];
  connected: boolean;
  error: string | null;
}

const TIMEOUT_MS = 4000;

async function fetchTimed(url: string, init: RequestInit = {}): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal, cache: "no-store" });
    if (!response.ok) throw new Error(`world view answered ${response.status}`);
    return response;
  } finally {
    clearTimeout(timer);
  }
}

async function decode(camera: Json): Promise<LiveCamera> {
  const blob = await (await fetchTimed(String(camera.url))).blob();
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.src = url;
  try {
    await image.decode();
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
  return { name: String(camera.name), url };
}

/** Poll the world view until stopped; `onView` receives each change. */
export function followLive(onView: (view: LiveView) => void, interval = 250): () => void {
  let stopped = false;
  let binding: string | null = null;
  let shownKey: string | null = null;
  let shown: LiveCamera[] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  let last: LiveView = { phase: "waiting", message: "Connecting to the world view…", frame: null, cameras: [],
    connected: false, error: null };
  const emit = (change: Partial<LiveView>) => {
    last = { ...last, ...change };
    if (!stopped) onView(last);
  };
  const release = (cameras: LiveCamera[]) => cameras.forEach((camera) => URL.revokeObjectURL(camera.url));

  async function poll(): Promise<void> {
    try {
      const state = await (await fetchTimed("/api/live/status")).json();
      if (binding !== state.binding_id) {
        binding = state.binding_id;
        shownKey = null;
        release(shown);
        shown = [];
        emit({ cameras: [] });
      }
      emit({ phase: String(state.phase ?? "waiting"), message: String(state.message ?? ""), frame: state.frame ?? null,
        connected: true, error: null });
      const frame = state.frame;
      if (frame) {
        const key = `${binding}:${frame.index}:${frame.hash}`;
        if (key !== shownKey) {
          const results = await Promise.allSettled((frame.cameras ?? []).map(decode));
          const cameras = results.flatMap((result) => (result.status === "fulfilled" ? [result.value] : []));
          if (cameras.length !== results.length || stopped || binding !== state.binding_id) {
            release(cameras);
            if (cameras.length !== results.length) throw new Error("a camera of this frame could not be decoded");
            return;
          }
          release(shown);
          shown = cameras;
          shownKey = key;
          emit({ cameras });
          await fetchTimed("/api/live/ack", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ binding_id: binding, ack_token: state.ack_token, frame_index: frame.index,
              frame_hash: frame.hash }),
          });
        }
      }
    } catch (error) {
      emit({ connected: false, error: (error as Error).message });
    } finally {
      if (!stopped) timer = setTimeout(() => void poll(), interval);
    }
  }

  void poll();
  return () => {
    stopped = true;
    if (timer !== null) clearTimeout(timer);
    release(shown);
  };
}
