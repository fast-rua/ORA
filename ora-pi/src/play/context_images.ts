// Free play's image window: what Pi sends of the images its conversation holds.
//
// A free-play run keeps its whole conversation in Pi's own session, and every act adds an observation
// with the robot's cameras (three PNGs on RoboDojo) besides the images tool results return. Sent in
// full, a long session's request outgrows a provider's body limit: DeepSeek refused a 50.8 MB request
// with HTTP 413 after 54 calls of one session. Before every model call (Pi's `context` hook, which
// changes only that call's messages, never the session) the images of the latest
// `observations` observation messages and of the latest `tool_results` tool results with images are
// sent; every older image becomes one line of text naming the observation step and camera, or the
// tool and image, it stood for. Images the operator attached to a message are always sent. The
// provider_request artifact records the request as sent.
//
// When the images leave decides what the provider's prefix cache is worth. A window that slides at
// every act rewrites an old message in the middle of the prompt each time, so everything after it is
// computed again: on the YAM station with four cameras, 62% of a session's calls recomputed more than
// 8k tokens (2026-11-14, 13-18k for an observation leaving, 100-150k for a tool result near the start
// of the prompt). With `prune_at` the images leave in batches instead (`batchedContextImages`): they
// all stay until a new instruction begins while `prune_at` or more messages of a kind still carry
// theirs, and then every one but the latest `observations` (or `tool_results`) goes at once. Inside an
// instruction nothing sent earlier changes, so its calls share one prefix; `HARD_LIMIT` messages of a
// kind prune in the middle of one that never ends. What is pruned is a function of the conversation
// alone, so a restarted agent process, the terminal and the web page send the same request.

type Json = any;

export interface ContextImageWindow {
  /** Observation messages (the robot's cameras after each act) whose images are sent. */
  observations: number;
  /** Tool results with images (an observe, a grasp plan's review) whose images are sent. */
  tool_results: number;
  /**
   * Batch pruning: images stay until an instruction begins with this many messages of a kind still
   * carrying theirs, then all but the latest `observations` / `tool_results` leave at once. Absent
   * (a plan resolved before it existed): the window slides at every call.
   */
  prune_at?: number;
}

/** A plan without the field: the sliding window every plan before `prune_at` ran with. */
export const DEFAULT_CONTEXT_IMAGES: ContextImageWindow = { observations: 3, tool_results: 3 };
/** Messages of one kind that may carry images before they are pruned inside a running instruction. */
export const HARD_LIMIT = 50;
/** The observation message the play driver sends at every act (driver.ts). */
export const OBSERVATION_MESSAGE = "ora.play_observation";

const CAMERA_LABEL = /·\s*([A-Za-z0-9_.-]+)\s*·\s*latest at this message/;

function kindOf(message: Json): "observation" | "tool" | null {
  if (message?.role === "custom" && message.customType === OBSERVATION_MESSAGE) return "observation";
  if (message?.role === "toolResult") return "tool";
  return null;
}

function imageCount(message: Json): number {
  const content = message?.content;
  return Array.isArray(content) ? content.filter((block: Json) => block?.type === "image").length : 0;
}

function placeholders(message: Json, kind: "observation" | "tool"): Json[] {
  const content: Json[] = message.content;
  const step = message.details?.ora_observation_step;
  let index = 0;
  return content.map((block: Json, position: number) => {
    if (block?.type !== "image") return block;
    index += 1;
    if (kind === "observation") {
      const previous = content[position - 1];
      const camera = previous?.type === "text" ? CAMERA_LABEL.exec(String(previous.text ?? ""))?.[1] : undefined;
      const where = [step !== undefined && step !== null ? `observation step ${step}` : "an earlier observation",
        camera ?? `image ${index}`].join(", ");
      return { type: "text", text: `[earlier image not resent: ${where}; the latest observations carry the current view]` };
    }
    return { type: "text", text: `[earlier image not resent: ${String(message.toolName ?? "tool")} result image ${index}]` };
  });
}

/** The operator's instruction, which opens a turn (driver.ts sends it as the user's own message). */
function opensInstruction(message: Json): boolean {
  return message?.role === "user";
}

/**
 * Batch pruning (`window.prune_at`): the positions whose images are not sent. Walking the conversation
 * forward, a kind's image-bearing messages accumulate; where an instruction begins with `prune_at` or
 * more of them, or `HARD_LIMIT` is passed inside one, all but the latest `observations` /
 * `tool_results` are dropped together.
 */
function batchedOmissions(messages: Json[], window: ContextImageWindow): Set<number> {
  const keep = { observation: window.observations, tool: window.tool_results };
  const live: Record<"observation" | "tool", number[]> = { observation: [], tool: [] };
  const dropped = new Set<number>();
  const prune = (kind: "observation" | "tool") => {
    const cut = live[kind].length - keep[kind];
    if (cut <= 0) return;
    for (const position of live[kind].splice(0, cut)) dropped.add(position);
  };
  const pruneAt = Math.max(window.prune_at ?? HARD_LIMIT, keep.observation, keep.tool);
  messages.forEach((message, position) => {
    if (opensInstruction(message)) {
      for (const kind of ["observation", "tool"] as const) if (live[kind].length >= pruneAt) prune(kind);
    }
    const kind = kindOf(message);
    if (kind === null || imageCount(message) === 0) return;
    live[kind].push(position);
    if (live[kind].length > HARD_LIMIT) prune(kind);
  });
  return dropped;
}

/**
 * The messages with images only where `window` sends them: the latest observation and tool-result
 * messages (a sliding window), or everything since the last batch pruning (`window.prune_at`).
 * Returns the input list itself when nothing is omitted.
 */
export function boundContextImages(messages: Json[], window: ContextImageWindow = DEFAULT_CONTEXT_IMAGES):
  { messages: Json[]; omitted: number } {
  if (window.prune_at !== undefined) {
    const dropped = batchedOmissions(messages, window);
    if (dropped.size === 0) return { messages, omitted: 0 };
    let count = 0;
    const sent = messages.map((message, position) => {
      if (!dropped.has(position)) return message;
      count += imageCount(message);
      return { ...message, content: placeholders(message, kindOf(message)!) };
    });
    return { messages: sent, omitted: count };
  }
  let observations = 0;
  let tools = 0;
  let omitted = 0;
  let result: Json[] | null = null;
  for (let position = messages.length - 1; position >= 0; position -= 1) {
    const message = messages[position];
    const kind = kindOf(message);
    const count = kind === null ? 0 : imageCount(message);
    if (kind === null || count === 0) continue;
    const keep = kind === "observation" ? observations++ < window.observations : tools++ < window.tool_results;
    if (keep) continue;
    result ??= messages.slice();
    result[position] = { ...message, content: placeholders(message, kind) };
    omitted += count;
  }
  return { messages: result ?? messages, omitted };
}

/** The plan's window (``plan.play.context_images``), or the default. */
export function contextImageWindow(value: Json): ContextImageWindow {
  const valid = (item: Json) => Number.isInteger(item) && item >= 1 && item <= HARD_LIMIT;
  if (value && valid(value.observations) && valid(value.tool_results)) {
    const window: ContextImageWindow = { observations: value.observations, tool_results: value.tool_results };
    if (valid(value.prune_at) && value.prune_at >= Math.max(window.observations, window.tool_results)) {
      window.prune_at = value.prune_at;
    }
    return window;
  }
  return { ...DEFAULT_CONTEXT_IMAGES };
}
