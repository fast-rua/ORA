// ORA's OpenAI-compatible provider (the default DeepSeek path), ported from
// ora/agent/providers/{native,openai_wire_options,continuation,tool_names,
// recovery,recovery_worker,attempts,stream_failures}.py.

export {
  OPENAI_CONTINUATION_KIND, OPENAI_PROTOCOL, OpenAICompatibleProvider, accumulateChatCompletion,
  buildOpenAICompatibleProvider, completionResult, prepareRequest, rejectTruncatedTools,
  type Accumulated, type CompleteOptions, type OpenAICompatibleOptions, type WireOptions,
} from "./openai.ts";
export { FakeProvider, ReplayProvider } from "./fake.ts";
export {
  ANTHROPIC_CONTINUATION_KIND, ANTHROPIC_PROTOCOL, AnthropicMessagesProvider, accumulateAnthropicEvents,
  buildAnthropicMessagesProvider, declaredPropertyTypes, hasCompleteThinkingSignature, normalizeAnthropicBlocks,
  normalizeAnthropicUsage, parseAnthropicEvents, translateAnthropicRequest, type AnthropicMessagesOptions,
} from "./anthropic.ts";
export {
  PROVIDER_CONTINUATION_MESSAGE_KEY, PROVIDER_CONTINUATION_STATE_SCHEMA, ProviderContinuationState,
  canonicalAssistantReplayView, canonicalAssistantSha256, stripProviderContinuation,
} from "./continuation.ts";
export { openaiToolNames, projectOpenaiToolNames, responseToolName, toolAlias } from "./tool_names.ts";
export {
  REQUEST_OPTION_FIELDS, checkpointRequestOptions, normalizeUsage, projectToolStrictness, projectWireBody,
  validateRequestOptions,
} from "./wire_options.ts";
export {
  type Clock, DEADLINE_RECOVERY_POLICY, FIRST_TOKEN_RECOVERY_POLICY, LEGACY_RECOVERY_POLICY, RECOVERY_POLICIES,
  RECOVERY_POLICY, RECOVERY_WINDOW_S, RecoveryWindow, STALL_S, SYSTEM_CLOCK, retryAfterSeconds, transientTransport,
} from "./recovery.ts";
export { ATTEMPT_FAILURE_SCHEMA, CREDENTIAL_ECHO_MESSAGE, attemptFailureRecord, failureEvidence } from "./attempts.ts";
export { STREAM_BREAK_RETRIES, httpStatusFailure, isStreamBreak } from "./stream_failures.ts";
export { ProviderAborted, requestHeaders } from "./transport.ts";
export { agentOptions, connectTimeoutMs, providerFetch } from "./http.ts";
export {
  PolicySessionError, ProviderAttemptTimeout, ProviderClosed, ProviderRecoveryDeadline, PyError, StreamInterrupted, TransportError, sha256Hex,
} from "./py.ts";
