// Every robot-specific rule the Native Context port hard-codes in Python.
//
// The TypeScript Context core is meant to know no robot: the tool names that
// mutate the world, the result fields that are private, the outcome tools whose
// results must stay exact, the checkpoint prose filters and the compactor
// prompt are policy owned by the Python side. They are collected here,
// verbatim from the Python modules named beside each one, and reach the
// pipeline as one `ContextDomain` value (`NativeContextPipeline({ domain })`),
// so Python can send its own values over the RPC instead of these defaults.

import { pythonRegex } from "./pyutil.ts";

export interface ContextDomainSpec {
  /** The Python pipeline's NativeContextPipeline.mutation_tools default (context/native.py). */
  mutationTools: string[];
  /** projection._action_step_inventory: an accepted mutation's result status. */
  acceptedMutationStatuses: string[];
  /** projection._tool_result_has_exact_causal_outcome: results never externalized. */
  exactOutcomeTools: string[];
  /** compactor_privacy.py: the checker whose verdict the compactor never sees. */
  taskCheckTool: string;
  taskCheckVerdictField: string;
  taskCheckWithheld: Record<string, string>;
  /** compactor_privacy.py: finish refusals that repeat the checker's verdict. */
  finishTool: string;
  finishVerdictRefusalCodes: string[];
  finishVerdictWithheld: Record<string, string>;
  /** Code mode (trace_privacy.py): the script tools of this run, none in tool-call mode, and the fields of a
   *  script's result, and of its handed_back, that the script wrote itself. */
  scriptTools: string[];
  scriptAuthoredFields: string[];
  scriptHandedBackFields: string[];
  /** projection._CAUSAL_FORBIDDEN_KEYS. */
  causalForbiddenKeys: string[];
  /** projection._COMPACTOR_PRIVATE_KEYS. */
  compactorPrivateKeys: string[];
  /** projection._CHECKPOINT_FORBIDDEN (Python re source, IGNORECASE). */
  checkpointForbiddenPattern: string;
  /** projection._CHECKPOINT_OUTCOME_ASSERTION (Python re source, IGNORECASE). */
  checkpointOutcomeAssertionPattern: string;
  /** projection._recoverable_tool_marker: result keys kept beside an externalized preview. */
  markerStatusKeys: string[];
  /** historical_observation: the measured observation body and the keys its marker keeps. */
  observationBodyKey: string;
  observationIdentityKeys: string[];
  /** native.py _lossless_episode_capsule: the per-round official verdict message. */
  officialRoundResultSourcePrefix: string;
  officialRoundResultTag: string;
  /** native.py: compactor instructions in the capsule/emergency packet headers. */
  episodeCapsuleInstruction: string;
  emergencyCheckpointInstruction: string;
  /** rollout_compactor.COMPACTOR_SYSTEM_PROMPT. */
  compactorSystemPrompt: string;
  /** composition/surface/tool_names.CANONICAL_TOOL_NAMES: snake_case surfaces get `artifact_read`. */
  canonicalToolNames: string[];
  /** Legacy rewrite_compact_v1 texts (projection._strip_historical_images etc.). */
  legacyHistoricalImageText: string;
  legacyCausalImageText: string;
  legacyCausalObservationText: string;
  legacySpillRetrieveWith: string;
}

export interface ContextDomain extends ContextDomainSpec {
  checkpointForbidden: RegExp;
  checkpointOutcomeAssertion: RegExp;
}

/** recoverable_artifact_open / pointer validation: the tool that reopens an image. */
export const IMAGE_OPEN_TOOL = "observe";

export const COMPACTOR_SYSTEM_PROMPT = `You condense completed robot ActionSteps from one rollout.
Return exactly one JSON object with these seven keys:
task_progress, confirmed_facts, invalidated_beliefs, failed_attempts,
current_constraints, do_not_repeat, re_ground_next.

All fields are arrays of concise strings except failed_attempts, whose entries are
objects with exactly failure, evidence, and correction string fields.

Reconstruct intention/action -> tool receipt -> later visual evidence. Compare the
agent's belief with later images. Preserve empty grasps, wrong-object grasps,
slippage, false success beliefs, corrections, constraints, and unresolved state.
Use confirmed/inferred/unknown language when confidence matters. Do not narrate
frames one by one. Do not claim the official task verdict. Do not emit run IDs,
hashes, paths, historical pixels, world coordinates, or replayable target poses.
Treat repeated point, mask, and view queries within one search episode as one
consolidated attempt: preserve the accepted result, invalidated belief, and
remaining uncertainty, not a query-by-query inventory.
Express spatial state only qualitatively, for example "above the basket" or
"the descent stopped short". Never copy numeric positions, orientations,
distances, dimensions, apertures, pixel locations, axis assignments, or metric
units from the evidence. In particular, strings such as x=, y=, z=, xyz_m,
position_m, delta_mm, point_uv, quaternions, radians, millimeters, centimeters,
and meters are forbidden in the JSON. The retained exact observation and tools,
not this checkpoint, own all geometry. The next decision must re-ground against
the retained current observation.
Prefer at most two entries per array, each a short sentence. Use empty arrays
when there is nothing useful to preserve; do not fill every field by repeating
the same fact. Finish every array and the enclosing JSON object.
Return JSON only, without markdown.`;

export const DEFAULT_CONTEXT_DOMAIN_SPEC: ContextDomainSpec = {
  mutationTools: ["move_pose", "close_gripper", "open_gripper"],
  acceptedMutationStatuses: ["submitted", "action_submitted"],
  exactOutcomeTools: ["check_task", "finish_episode"],
  taskCheckTool: "check_task",
  taskCheckVerdictField: "task_success",
  taskCheckWithheld: {
    status: "task_check_result_withheld_from_compactor",
    reason: "official_outcome_reserved",
  },
  finishTool: "finish_episode",
  finishVerdictRefusalCodes: ["task_success_latched", "task_success_not_confirmed"],
  finishVerdictWithheld: { verdict_withheld_from_compactor: "official_outcome_reserved" },
  scriptTools: [],
  scriptAuthoredFields: ["output", "stderr", "stdout", "value"],
  scriptHandedBackFields: ["data", "reason"],
  causalForbiddenKeys: [
    "schema", "schema_version", "action_id", "receipt_id",
    "run_id", "event_id", "causal_parent_id", "plan_id", "sha256", "hash",
    "path", "artifact_path", "timestamp", "timestamps", "started_unix_s",
    "finished_unix_s", "worker", "transport", "transport_status",
    "reached", "verified", "state_changed",
    "object_gt", "object_gt_pose", "contact", "success_predicate",
    "task_source", "private_mask", "official_outcome",
  ],
  compactorPrivateKeys: [
    "object_gt", "object_gt_pose", "contact", "success_predicate",
    "task_source", "private_mask", "official_outcome", "official_success",
    "official_verdict", "task_success",
  ],
  checkpointForbiddenPattern:
    "(?:\\brun[_ -]?id\\b|\\bsha-?256\\b|"
    + "\\b(?:official[_-](?:success|verdict|outcome)|task[_-]success)\\b|"
    + "\\bpixel coordinates?\\b|\\bworld coordinates?\\b|"
    + "\\b(?:target|historical) pose\\b|"
    + "\\b(?:pose_xyz|position_m|delta_mm|point_uv|xyz_m)\\b|"
    + "\\b[xyz]\\s*(?:=|~)|"
    + "(?:/home/|outputs/|native/agent-artifacts/)|\\b[0-9a-f]{64}\\b)",
  checkpointOutcomeAssertionPattern:
    "\\b(?:official[ ]+(?:success|verdict|outcome)|task[ ]+success)\\b"
    + "\\s*(?:(?:is|was|=|:|returned|reported|indicated|showed)\\s*)"
    + "(?:true|false|success(?:ful)?|fail(?:ed|ure)?)\\b",
  markerStatusKeys: ["status", "success", "ok", "observation_id", "views", "view_dimensions"],
  observationBodyKey: "current_actual",
  observationIdentityKeys: ["step", "actual_sequence", "observation_id"],
  officialRoundResultSourcePrefix: "official_round_result_",
  officialRoundResultTag: "official_round_result",
  episodeCapsuleInstruction:
    "Summarize only this closed Episode's durable decision evidence. "
    + "Do not repeat identifiers, coordinates, hashes, or the official verdict; "
    + "the exact transcript remains recoverable separately.",
  emergencyCheckpointInstruction:
    "This is the L6 emergency fallback after L1-L5. Produce one cumulative "
    + "checkpoint from existing Episode Capsules and only the newly eligible "
    + "post-lossless ActionSteps. Do not copy identifiers, coordinates, hashes, "
    + "or official verdicts.",
  compactorSystemPrompt: COMPACTOR_SYSTEM_PROMPT,
  canonicalToolNames: [
    "align_gripper_propose", "artifact_read", "grasp_plan", "grasp_query",
    "grasp_verify_retention", "motion_probe", "payload_bind", "payload_unbind",
    "place_plan", "place_verify", "vision_ground", "vision_offset", "vision_segment",
  ],
  legacyHistoricalImageText:
    "[historical observation image omitted; recover step {step} "
    + "with observation.dump_state or vision.inspect]",
  legacyCausalImageText: "[historical image omitted from the active causal context]",
  legacyCausalObservationText:
    "[earlier actual observation omitted; later actions and tool results follow]",
  legacySpillRetrieveWith: "repeat the read-only tool call to refresh the complete result",
};

export function resolveContextDomain(overrides: Partial<ContextDomainSpec> = {}): ContextDomain {
  const spec: ContextDomainSpec = { ...DEFAULT_CONTEXT_DOMAIN_SPEC, ...overrides };
  return {
    ...spec,
    checkpointForbidden: pythonRegex(spec.checkpointForbiddenPattern, "gi"),
    checkpointOutcomeAssertion: pythonRegex(spec.checkpointOutcomeAssertionPattern, "gi"),
  };
}

export const DEFAULT_CONTEXT_DOMAIN: ContextDomain = resolveContextDomain();
