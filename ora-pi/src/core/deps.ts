// Wire the ported context policy, request assembler and provider into OraCore.

import { clonePy, PyFloat } from "../pyjson.ts";
import {
  buildAnthropicMessagesProvider, buildOpenAICompatibleProvider, FakeProvider, prepareRequest, ReplayProvider,
  stripProviderContinuation, PROVIDER_CONTINUATION_MESSAGE_KEY,
} from "../provider/index.ts";
import type { ContextDeps, InitParams, OraCore } from "./session.ts";

type Json = any;

/**
 * A numeric provider-spec field as a plain number. The spec crosses from Python
 * as JSON, so an integral float (`900.0`, `120.0`) arrives as a PyFloat, which
 * arithmetic and Number() would otherwise turn into NaN or a string.
 */
function num(value: Json): number | null {
  if (value === null || value === undefined) return null;
  if (value instanceof PyFloat) return value.value;
  if (typeof value === "bigint") return Number(value);
  return typeof value === "number" ? value : Number(value);
}

function wireOptions(provider: Record<string, Json>) {
  return {
    outputTokenParameter: provider.output_token_parameter ?? undefined,
    toolStrictness: provider.tool_strictness ?? null,
    reasoningField: provider.reasoning_field ?? undefined,
    requestOptions: provider.request_options ?? null,
  };
}

/**
 * The provider the plan names. The rollout compactor's own instance passes
 * `maxRetries: 0` (the adapters default to 2) and the policy provider's
 * `fetchImpl`, as Python's shallow copy keeps max_retries=0 and shares the client.
 */
export function createProvider(
  config: InitParams, overrides: { maxRetries?: number; fetchImpl?: typeof fetch } = {},
): ContextDeps["provider"] & { prepareRequest(execution: Json): Json } {
  return providerFromSpec(config.provider, {
    apiKey: String(process.env.ORA_PI_API_KEY ?? ""),
    standbyApiKeys: String(process.env.ORA_PI_STANDBY_API_KEYS ?? "").split(",").filter((key) => key),
  }, overrides);
}

/**
 * The provider one spec names, with its credentials (they travel by environment,
 * never in the spec). A compactor bound to its own profile passes its own key,
 * no standby keys and `maxRetries: 0`, and gets its own HTTP client.
 */
export function providerFromSpec(
  spec: Record<string, Json>, credentials: { apiKey: string; standbyApiKeys?: string[] },
  overrides: { maxRetries?: number; fetchImpl?: typeof fetch } = {},
): ContextDeps["provider"] & { prepareRequest(execution: Json): Json } {
  const retries = {
    ...(overrides.maxRetries === undefined ? {} : { maxRetries: overrides.maxRetries }),
    ...(overrides.fetchImpl === undefined ? {} : { fetchImpl: overrides.fetchImpl }),
  };
  const options = wireOptions(spec);
  if (spec.kind === "fake") {
    const fake = new FakeProvider(spec.responses ?? []);
    const delayMs = Number(process.env.ORA_PI_FAKE_DELAY_MS ?? 0);
    // Test hook: the scripted request at this index answers only an abort (a stop mid-request).
    const hangAt = num(spec.hang_at);
    if (delayMs > 0 || hangAt !== null) {
      // Test hook: a slow scripted model, so steering, aborts and restarts can land mid-request.
      const complete = fake.complete.bind(fake);
      (fake as Json).complete = async (request: Json, options: Json) => {
        const hang = fake.index === hangAt;
        await new Promise<void>((resolve, reject) => {
          const aborted = () => reject(Object.assign(new Error("aborted"), { name: "ProviderAborted" }));
          if (options?.signal?.aborted) return aborted();
          const timer = hang ? null : setTimeout(resolve, delayMs);
          options?.signal?.addEventListener("abort", () => {
            if (timer !== null) clearTimeout(timer);
            aborted();
          }, { once: true });
        });
        return complete(request, options);
      };
    }
    return Object.assign(fake as Json, {
      prepareRequest: (execution: Json) => clonePy(execution),
    });
  }
  if (spec.kind === "replay") {
    const replay = new ReplayProvider(spec.recordings ?? [], options as Json);
    return Object.assign(replay as Json, {
      prepareRequest: (execution: Json) => prepareRequest(execution, options as Json),
    });
  }
  if (spec.kind === "anthropic_messages") {
    const anthropic = buildAnthropicMessagesProvider({
      baseUrl: String(spec.endpoint),
      apiKey: credentials.apiKey,
      standbyApiKeys: credentials.standbyApiKeys ?? [],
      credentialStartIndex: num(spec.credential_start_index) ?? 0,
      timeoutSeconds: num(spec.timeout_s) ?? 900,
      recoveryPolicy: spec.recovery_policy ?? undefined,
      recoveryWindowS: num(spec.recovery_window_s),
      authScheme: spec.auth_scheme ?? undefined,
      continuationReplay: spec.continuation_replay ?? undefined,
      sessionAffinityId: spec.session_affinity_id ?? null,
      outputTokenParameter: spec.output_token_parameter ?? null,
      toolStrictness: spec.tool_strictness ?? null,
      reasoningField: spec.reasoning_field ?? null,
      requestOptions: spec.request_options ?? null,
      ...retries,
    });
    // AnthropicMessagesProvider exposes prepareRequest(execution) = translate_anthropic_request.
    return anthropic as Json;
  }
  const provider = buildOpenAICompatibleProvider({
    baseUrl: String(spec.endpoint),
    apiKey: credentials.apiKey,
    timeoutSeconds: num(spec.timeout_s) ?? 900,
    recoveryPolicy: spec.recovery_policy ?? undefined,
    recoveryWindowS: num(spec.recovery_window_s),
    outputTokenParameter: spec.output_token_parameter ?? undefined,
    sessionAffinityId: spec.session_affinity_id ?? null,
    toolStrictness: spec.tool_strictness ?? null,
    reasoningField: spec.reasoning_field ?? undefined,
    requestOptions: spec.request_options ?? null,
    authScheme: spec.auth_scheme ?? null,
    ...retries,
  });
  // OpenAICompatibleProvider already exposes prepareRequest(execution).
  return provider as Json;
}

/** `NativeRequestAssembler.prepare` without the lossless cache-stability evidence. */
export function legacyAssembler(config: InitParams, provider: { prepareRequest(execution: Json): Json }) {
  return {
    prepare(projected: Json) {
      const execution: Record<string, Json> = {
        model: config.policy.model,
        messages: clonePy(projected.messages),
        tools: clonePy(projected.tools),
        tool_choice: "auto",
        max_tokens: config.policy.max_output_tokens,
        stream: config.policy.stream,
      };
      if (config.policy.reasoning_effort !== "provider_default") execution.reasoning_effort = config.policy.reasoning_effort;
      if (config.policy.stream) execution.stream_options = { include_usage: true };
      const canonical = clonePy(execution);
      canonical.messages = execution.messages.map((message: Json) => (
        message && typeof message === "object" ? stripProviderContinuation(message) : clonePy(message)));
      const hasContinuation = execution.messages.some((message: Json) => (
        message && typeof message === "object" && PROVIDER_CONTINUATION_MESSAGE_KEY in message));
      void hasContinuation;
      const wire = provider.prepareRequest(execution);
      return { canonical, wire, execution: clonePy(execution) };
    },
  };
}

export async function createDeps(core: OraCore, config: InitParams): Promise<ContextDeps> {
  const provider = createProvider(config);
  const { contextDeps } = await import("./context_wiring.ts");
  return contextDeps(core, config, provider);
}
