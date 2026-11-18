// `ora/agent/error_classes.py` (ToolError, schema_refusal) and its
// infrastructure code table.

export const ARGUMENT_REFUSAL_CODES = new Set([
  "invalid_arguments",
  "invalid_motion_arguments",
  "invalid_servo_delta",
  "invalid_tool_call",
  "out_of_bounds",
  "target_ref_unavailable",
  "unknown_section",
  "workspace_bounds",
]);

export const INFRASTRUCTURE_CODES = new Set([
  "internal_tool_error",
  "environment_contract",
  "environment_unavailable",
  "artifact_hash_mismatch",
  "artifact_integrity_fault",
  "invalid_camera_calibration",
  "pose_state_unavailable",
  "embodiment_binding",
  "grounding_unavailable",
  "tool_capability_unavailable",
  "tool_surface_capability_mismatch",
]);

export interface ToolErrorOptions {
  code?: string;
  retryable?: boolean;
  field?: string | null;
  received?: unknown;
  constraints?: Record<string, unknown> | null;
  retryRequiresChangedArguments?: boolean | null;
}

/** A structured, policy-visible tool failure. */
export class ToolError extends Error {
  code: string;
  retryable: boolean;
  field: string | null;
  received: unknown;
  constraints: Record<string, unknown>;
  retryRequiresChangedArguments: boolean | null;

  constructor(message: string, options: ToolErrorOptions = {}) {
    super(message);
    this.name = "ToolError";
    this.code = options.code ?? "invalid_arguments";
    this.retryable = options.retryable ?? true;
    this.field = options.field ?? null;
    this.received = options.received === undefined ? null : options.received;
    this.constraints = { ...(options.constraints ?? {}) };
    let changed = options.retryRequiresChangedArguments ?? null;
    if (changed === null && ARGUMENT_REFUSAL_CODES.has(this.code)) changed = true;
    this.retryRequiresChangedArguments = changed;
  }

  asDict(): Record<string, unknown> {
    const reported: Record<string, unknown> = {
      code: this.code,
      message: this.message,
      retryable: this.retryable,
      field: this.field,
      received: this.received,
      constraints: this.constraints,
      infrastructure: INFRASTRUCTURE_CODES.has(this.code),
    };
    if (this.retryRequiresChangedArguments !== null) {
      reported.retry_requires_changed_arguments = Boolean(this.retryRequiresChangedArguments);
    }
    return reported;
  }
}

/** The policy-visible refusal for one input SchemaViolation. */
export function schemaRefusal(error: {
  message: string; path?: string; keyword?: string | null;
  constraints?: Record<string, unknown>; received?: unknown;
}): ToolError {
  const path = String(error.path ?? "");
  const field = path.startsWith("$input.") ? path.slice("$input.".length) : null;
  const keyword = error.keyword ?? null;
  const constraints = { ...(error.constraints ?? {}) };
  return new ToolError(error.message, {
    code: "invalid_arguments", field,
    received: error.received === undefined ? null : error.received,
    constraints: keyword ? { violated: keyword, ...constraints } : constraints,
  });
}
