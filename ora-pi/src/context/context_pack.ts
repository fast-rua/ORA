// A plan's context pack, applied to each projected request (`policy.context_pack`).
//
// The Context pipeline projects ORA's history as it does for any plan (the
// bound composition, lossless retention, round static facts); the pack then
// changes only what the request delivers:
//
// 1. static messages: each message the pack replaces is found by the sha256
//    of its exact text and replaced, or dropped (the project instructions
//    when the pack composes its own system message);
// 2. history: the compact view (`compact_view.ts`), its kept images
//    downscaled by Python (Pillow, byte-identical to the pack's reference);
// 3. tools: the advertised tools with the pack's description overlay.
//
// A pack's ``context_management`` may also turn off lossless retention's
// layers that act on token usage (``packContextManagement``, wired into the
// pipeline by core/context_wiring.ts).
//
// The replacements and tools come once per session from Python
// (`context_pack.delivery`, `ora/agent/pi/context_pack_rpc.py`), checked there
// against the plan; downscaled images are cached by their source URL. The
// manifest gains a `context_pack` record of what changed. The history itself
// is never edited, so every Context-retention invariant still holds for it.

import { createHash } from "node:crypto";

import { clonePy, isMapping, PolicySessionError, sha256Hex, canonicalSha256 } from "./pyutil.ts";
import { compactView, withImageUrls, type CompactViewConfig, type Message } from "./compact_view.ts";

type Json = any;

export const CONTEXT_PACK_VIEW_SCHEMA = "ora.context-pack-view.v1";

/** The part of the RPC peer the pack needs (``RpcPeer.request``). */
export interface PackRpc {
  request(method: string, params: Json): Promise<Json>;
}

export interface Projected {
  messages: Message[];
  tools: Json[];
  manifest: Record<string, Json>;
}

/** One static message the pack replaces (``content: null`` drops it). */
interface Replacement {
  role: string;
  content_sha256: string;
  content: string | null;
}

interface Delivery {
  messages: Replacement[];
  tools: Json[] | null;
}

/**
 * The Context pipeline options of a pack's ``context_management``: lossless
 * retention's L4 pressure externalization and L5/L6 rollout compaction, both
 * on unless the pack turns them off. Nothing for a plan without a pack.
 */
export function packContextManagement(binding: Json): { pressureExternalization?: boolean; rolloutCompaction?: boolean } {
  const switches = isMapping(binding) ? binding.profile?.context_management : null;
  if (!isMapping(switches)) return {};
  return {
    pressureExternalization: switches.pressure_externalization !== false,
    rolloutCompaction: switches.rollout_compaction !== false,
  };
}

function urlKey(url: string): string {
  return createHash("sha256").update(url).digest("hex");
}

export class ContextPackView {
  readonly binding: Json;
  private readonly rpc: PackRpc;
  private delivery: Delivery | null = null;
  private readonly downscaled = new Map<string, string>();

  constructor(binding: Json, rpc: PackRpc) {
    if (!isMapping(binding) || binding.schema !== "ora.context-pack-binding.v1") {
      throw new PolicySessionError("policy.context_pack is not an ora.context-pack-binding.v1 binding");
    }
    this.binding = binding;
    this.rpc = rpc;
  }

  private get view(): CompactViewConfig | null {
    return this.binding.profile?.compact_view ?? null;
  }

  private async load(): Promise<Delivery> {
    if (this.delivery === null) {
      const answer = await this.rpc.request("context_pack.delivery", {});
      if (!isMapping(answer) || !Array.isArray(answer.messages)) {
        throw new PolicySessionError("context_pack.delivery returned no message replacements");
      }
      this.delivery = { messages: answer.messages, tools: Array.isArray(answer.tools) ? answer.tools : null };
    }
    return this.delivery;
  }

  /** The static replacements: each target must be exactly one message of the request. */
  private replaceStatic(messages: Message[], replacements: Replacement[]): [Message[], number, number] {
    const output: Array<Message | null> = messages.map((message) => message);
    let replaced = 0;
    let dropped = 0;
    for (const item of replacements) {
      const matches: number[] = [];
      output.forEach((message, index) => {
        if (message !== null && message.role === item.role && typeof message.content === "string"
          && sha256Hex(message.content) === item.content_sha256) matches.push(index);
      });
      if (matches.length !== 1) {
        throw new PolicySessionError(
          `context pack: the ${item.role} message it replaces (sha256 ${item.content_sha256.slice(0, 12)}) `
          + `occurs ${matches.length} times in the request`,
        );
      }
      if (item.content === null) {
        output[matches[0]] = null;
        dropped += 1;
      } else {
        output[matches[0]] = { ...clonePy(output[matches[0]]!), content: item.content };
        replaced += 1;
      }
    }
    return [output.filter((message): message is Message => message !== null), replaced, dropped];
  }

  private async downscale(urls: string[]): Promise<string[]> {
    const missing = [...new Set(urls.filter((url) => !this.downscaled.has(urlKey(url))))];
    if (missing.length) {
      const answer = await this.rpc.request("context_pack.images", { images: missing });
      const images = isMapping(answer) ? answer.images : null;
      if (!Array.isArray(images) || images.length !== missing.length) {
        throw new PolicySessionError("context_pack.images returned another number of images");
      }
      missing.forEach((url, index) => this.downscaled.set(urlKey(url), String(images[index])));
    }
    return urls.map((url) => this.downscaled.get(urlKey(url))!);
  }

  async apply(projected: Projected): Promise<Projected> {
    const delivery = await this.load();
    const [replacedMessages, replaced, dropped] = this.replaceStatic(projected.messages, delivery.messages);
    let messages = replacedMessages;
    const record: Record<string, Json> = {
      schema: CONTEXT_PACK_VIEW_SCHEMA,
      name: this.binding.name,
      replaced_messages: replaced,
      dropped_messages: dropped,
      compact_view: null,
    };
    const management = packContextManagement(this.binding);
    if (Object.keys(management).length) {
      record.context_management = {
        pressure_externalization: management.pressureExternalization,
        rollout_compaction: management.rolloutCompaction,
      };
    }
    const config = this.view;
    if (config !== null) {
      const view = compactView(messages, config);
      const urls = view.keptImages.map((image) => image.url);
      const delivered = config.image_long_edge_px === null ? urls : await this.downscale(urls);
      messages = withImageUrls(view, delivered);
      record.compact_view = {
        ...view.stats,
        resized_images: delivered.filter((url, index) => url !== urls[index]).length,
      };
    }
    const tools = delivery.tools ?? projected.tools;
    record.tools_sha256 = canonicalSha256(tools);
    record.messages = messages.length;
    // In place: the request assembler adds its evidence to the same manifest object.
    projected.manifest.context_pack = record;
    return { messages, tools: clonePy(tools), manifest: projected.manifest };
  }
}
