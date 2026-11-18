// Port of ora/agent/providers/continuation.py: the content-addressed envelope
// for provider-owned state that must be replayed with one assistant message.

import { canonicalBytesPy, clonePy } from "../pyjson.ts";
import { PolicySessionError, sha256Hex } from "./py.ts";

export const PROVIDER_CONTINUATION_STATE_SCHEMA = "ora.provider-continuation-state.v1";
export const PROVIDER_CONTINUATION_MESSAGE_KEY = "_provider_continuation_state";
export const REPLAY_POLICY = "required_unmodified";

type Json = { [key: string]: any };

function isMapping(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && Object.getPrototypeOf(value) === Object.prototype;
}

function define(target: Json, key: string, value: unknown): void {
  Object.defineProperty(target, key, { value, enumerable: true, writable: true, configurable: true });
}

/** The provider-neutral assistant fields a continuation binds. */
export function canonicalAssistantReplayView(message: Json): Json {
  const view: Json = {};
  for (const key of Object.keys(message)) {
    if (!key.startsWith("_") && key !== "reasoning_content") define(view, key, clonePy(message[key]));
  }
  return view;
}

export function canonicalAssistantSha256(message: Json): string {
  return sha256Hex(canonicalBytesPy(canonicalAssistantReplayView(message)));
}

/** Remove internal continuation state from one canonical message. */
export function stripProviderContinuation(message: Json): Json {
  const result: Json = {};
  for (const key of Object.keys(message)) {
    if (key !== PROVIDER_CONTINUATION_MESSAGE_KEY) define(result, key, clonePy(message[key]));
  }
  return result;
}

const HEX64 = /^[0-9a-f]{64}$/;

/**
 * Native kinds whose payload no Provider signs (OpenAI-compatible
 * `reasoning_content`, openai.ts `OPENAI_CONTINUATION_KIND`): the Provider
 * accepts it beside any assistant message, so the state may be bound to a
 * rewrite of its message. A signed payload (Anthropic thinking blocks) replays
 * only with the message it was signed with.
 */
export const UNSIGNED_CONTINUATION_KINDS: ReadonlySet<string> = new Set(["openai_chat_reasoning_content_v1"]);

function unsignedBody(fields: {
  providerProtocol: string; model: string; nativeKind: string; nativePayload: Json;
  canonicalMessageSha256: string; responseId: string | null;
}): Json {
  return {
    schema: PROVIDER_CONTINUATION_STATE_SCHEMA,
    provider_protocol: fields.providerProtocol,
    model: fields.model,
    native_kind: fields.nativeKind,
    native_payload: clonePy(fields.nativePayload),
    canonical_message_sha256: fields.canonicalMessageSha256,
    response_id: fields.responseId,
    replay_policy: REPLAY_POLICY,
  };
}

function nonEmptyText(label: string, value: unknown): void {
  if (typeof value !== "string" || !value) {
    throw new PolicySessionError(`provider continuation ${label} must be non-empty text`);
  }
}

export class ProviderContinuationState {
  readonly providerProtocol: string;
  readonly model: string;
  readonly nativeKind: string;
  readonly nativePayload: Json;
  readonly canonicalMessageSha256: string;
  readonly responseId: string | null;
  readonly stateSha256: string;

  constructor(fields: {
    providerProtocol: string; model: string; nativeKind: string; nativePayload: Json;
    canonicalMessageSha256: string; responseId: string | null; stateSha256: string;
  }) {
    this.providerProtocol = fields.providerProtocol;
    this.model = fields.model;
    this.nativeKind = fields.nativeKind;
    this.nativePayload = fields.nativePayload;
    this.canonicalMessageSha256 = fields.canonicalMessageSha256;
    this.responseId = fields.responseId;
    this.stateSha256 = fields.stateSha256;
  }

  static create(args: {
    providerProtocol: string; model: string; nativeKind: string; nativePayload: Json;
    canonicalMessage: Json; responseId?: string | null;
  }): ProviderContinuationState {
    nonEmptyText("provider_protocol", args.providerProtocol);
    nonEmptyText("model", args.model);
    nonEmptyText("native_kind", args.nativeKind);
    const responseId = args.responseId ?? null;
    if (responseId !== null && (typeof responseId !== "string" || !responseId)) {
      throw new PolicySessionError("provider continuation response_id must be non-empty text");
    }
    if (!isMapping(args.nativePayload)) {
      throw new PolicySessionError("provider continuation native_payload must be an object");
    }
    const messageSha256 = canonicalAssistantSha256(args.canonicalMessage);
    const fields = {
      providerProtocol: args.providerProtocol, model: args.model, nativeKind: args.nativeKind,
      nativePayload: args.nativePayload, canonicalMessageSha256: messageSha256, responseId,
    };
    return new ProviderContinuationState({
      ...fields,
      nativePayload: clonePy(args.nativePayload),
      stateSha256: sha256Hex(canonicalBytesPy(unsignedBody(fields))),
    });
  }

  static fromMapping(value: Json): ProviderContinuationState {
    const expected = [
      "schema", "provider_protocol", "model", "native_kind", "native_payload",
      "canonical_message_sha256", "response_id", "replay_policy", "state_sha256",
    ];
    const keys = Object.keys(value);
    if (keys.length !== expected.length || !expected.every((key) => keys.includes(key))) {
      throw new PolicySessionError("provider continuation state fields are not canonical");
    }
    if (value.schema !== PROVIDER_CONTINUATION_STATE_SCHEMA) {
      throw new PolicySessionError("unsupported provider continuation schema");
    }
    if (value.replay_policy !== REPLAY_POLICY) {
      throw new PolicySessionError("unsupported provider continuation replay policy");
    }
    nonEmptyText("provider_protocol", value.provider_protocol);
    nonEmptyText("model", value.model);
    nonEmptyText("native_kind", value.native_kind);
    if (!isMapping(value.native_payload)) {
      throw new PolicySessionError("provider continuation native_payload must be an object");
    }
    const messageSha256 = value.canonical_message_sha256;
    if (typeof messageSha256 !== "string" || !HEX64.test(messageSha256)) {
      throw new PolicySessionError("provider continuation lacks canonical message identity");
    }
    const responseId = value.response_id;
    if (responseId !== null && (typeof responseId !== "string" || !responseId)) {
      throw new PolicySessionError("provider continuation has invalid response_id");
    }
    const stateSha256 = value.state_sha256;
    if (typeof stateSha256 !== "string" || !HEX64.test(stateSha256)) {
      throw new PolicySessionError("provider continuation lacks state identity");
    }
    const fields = {
      providerProtocol: value.provider_protocol, model: value.model, nativeKind: value.native_kind,
      nativePayload: value.native_payload, canonicalMessageSha256: messageSha256,
      responseId: responseId as string | null,
    };
    if (stateSha256 !== sha256Hex(canonicalBytesPy(unsignedBody(fields)))) {
      throw new PolicySessionError("provider continuation state hash mismatch");
    }
    return new ProviderContinuationState({
      ...fields, nativePayload: clonePy(value.native_payload), stateSha256,
    });
  }

  validateBinding(args: { message: Json; providerProtocol: string; model: string }): void {
    if (this.providerProtocol !== args.providerProtocol) {
      throw new PolicySessionError("provider continuation protocol does not match the active adapter");
    }
    if (this.model !== args.model) {
      throw new PolicySessionError("provider continuation model does not match the active request");
    }
    if (canonicalAssistantSha256(args.message) !== this.canonicalMessageSha256) {
      throw new PolicySessionError("provider continuation is not bound to this assistant message");
    }
  }

  /** This state bound to ``message`` (a rewrite of its own message) when its payload is unsigned; else null. */
  reboundTo(message: Json): ProviderContinuationState | null {
    if (!UNSIGNED_CONTINUATION_KINDS.has(this.nativeKind)) return null;
    return ProviderContinuationState.create({
      providerProtocol: this.providerProtocol, model: this.model, nativeKind: this.nativeKind,
      nativePayload: this.nativePayload, canonicalMessage: message, responseId: this.responseId,
    });
  }

  toDict(): Json {
    const body = unsignedBody(this);
    body.state_sha256 = this.stateSha256;
    return body;
  }
}
