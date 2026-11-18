// L6's summary writer: the bounded emergency checkpoint of every ActionStep
// (or, `unit: exchange`, every exchange) before the exact tail (ported from
// the Python Native pipeline's `_lossless_emergency_checkpoint`).
//
// The packet carries the previous checkpoint, the capsules it does not cover
// yet and the newly absorbed ActionSteps (exchanges) as the model was shown
// them; the stored transcript is cumulative, so replacing a checkpoint never
// makes an earlier exact ActionStep undiscoverable.

import { recoverableArtifactOpen } from "../../memory/artifacts.ts";
import {
  CHECKPOINT_FIELDS, EMERGENCY_CHECKPOINT_SCHEMA, ROLLOUT_PACKET_SCHEMA, compactorHistoryMessage,
  compactorPrivateViolations, estimatedTokens, observationGroups, packetMessageBlocks, packetSourceEventIds,
  semanticCapsuleSlot, validateCheckpoint,
} from "../../projection.ts";
import {
  ContextCompactionRejected, PolicySessionError, canonicalBytesPy, canonicalSha256, clonePy, dumpsCompact, get,
  hasKey, isMapping, num,
} from "../../pyutil.ts";
import type { SummaryWriter } from "../episode_capsule/capsule.ts";
import { checkSummaryIdentity, checkpointBudget, protectedSemantics } from "../summaries.ts";

export interface EmergencySummarizeArgs {
  messages: Record<string, any>[];
  tools: Record<string, any>[];
  inventory: Record<string, any>[];
  candidateAbsorbed: number;
}

/** The checkpoint and capsules a new emergency checkpoint builds on. */
export interface EmergencyPrior {
  summarizedActionSteps: number;
  episodeCheckpointedActionSteps: number;
  checkpoint: Record<string, any> | null;
  checkpointSha256: string | null;
  checkpointRef: Record<string, any> | null;
  transcriptRef: Record<string, any> | null;
  episodeCapsules: Record<string, any>[];
  compactionCount: number;
}

/** Create the bounded L6 emergency checkpoint of every ActionStep before the exact tail. */
export async function writeEmergencyCheckpoint(
  args: EmergencySummarizeArgs, writer: SummaryWriter, prior: EmergencyPrior,
): Promise<Record<string, any>> {
  const { messages, inventory, candidateAbsorbed } = args;
  const groups = observationGroups(messages);
  const semanticFrontier = Math.max(prior.summarizedActionSteps, prior.episodeCheckpointedActionSteps);
  if (candidateAbsorbed <= semanticFrontier) {
    throw new PolicySessionError("semantic fallback has no new complete ActionStep");
  }
  const absorbedStartActionStep = semanticFrontier;
  const absorbedStart = groups[absorbedStartActionStep][0];
  const absorbedEnd = groups[candidateAbsorbed][0];
  const newlyAbsorbed = messages.slice(absorbedStart, absorbedEnd)
    .filter((message) => get(message, "_protected_context") !== true);
  return writeCheckpoint(args, writer, prior, {
    groups,
    activeEmergency: prior.summarizedActionSteps > prior.episodeCheckpointedActionSteps,
    newlyAbsorbed,
    newlyAbsorbedLabel: "newly_absorbed_post_lossless_action_steps",
    transcriptEnd: absorbedEnd,
    transcriptActionStep: absorbedStartActionStep,
    packetFields: {
      semantic_frontier_before: absorbedStartActionStep,
      absorbed_action_steps: clonePy(inventory.slice(absorbedStartActionStep, candidateAbsorbed)),
    },
    coverage: {
      absorbed_action_step_count: candidateAbsorbed,
      through_action_step_index: candidateAbsorbed - 1,
    },
  });
}

/** What an exchange-unit checkpoint absorbs (stage.ts): exchanges by index, and the coverage it records. */
export interface ExchangeSelection {
  spans: [number, number][];
  /** The exchanges the packet carries, in order: those before the exact tail not yet summarized. */
  absorbed: number[];
  /** Where the cumulative exact transcript ends (the first exact-tail message). */
  transcriptEnd: number;
  /** The exchange frontier before and after, the exchange kept exact, the ActionSteps fully covered. */
  summarizedExchangesBefore: number;
  summarizedExchanges: number;
  keptExchange: number | null;
  coveredActionSteps: number;
}

/**
 * Create the bounded L6 emergency checkpoint of every exchange before the
 * exact tail (`unit: exchange`), the anchor exchange left out: it stays exact.
 */
export async function writeExchangeCheckpoint(
  args: Omit<EmergencySummarizeArgs, "candidateAbsorbed" | "inventory">, writer: SummaryWriter,
  prior: EmergencyPrior, selection: ExchangeSelection,
): Promise<Record<string, any>> {
  if (!selection.absorbed.length) throw new PolicySessionError("semantic fallback has no new exchange");
  const groups = observationGroups(args.messages);
  const newlyAbsorbed = selection.absorbed.flatMap((index) => args.messages
    .slice(...selection.spans[index]).filter((message) => get(message, "_protected_context") !== true));
  let transcriptActionStep = 0;
  groups.forEach(([start], index) => {
    if (start <= selection.spans[selection.absorbed[0]][0]) transcriptActionStep = index;
  });
  return writeCheckpoint(args, writer, prior, {
    groups,
    // An exchange checkpoint stays active until an Episode capsule supersedes it (which clears it).
    activeEmergency: selection.summarizedExchangesBefore > 0 && prior.checkpoint !== null,
    newlyAbsorbed,
    newlyAbsorbedLabel: "newly_absorbed_post_lossless_exchanges",
    transcriptEnd: selection.transcriptEnd,
    transcriptActionStep,
    packetFields: {
      unit: "exchange",
      semantic_frontier_before_exchange: selection.summarizedExchangesBefore,
      absorbed_exchanges: [...selection.absorbed],
      kept_exchange: selection.keptExchange,
    },
    coverage: {
      unit: "exchange",
      absorbed_exchange_count: selection.summarizedExchanges,
      through_exchange_index: selection.summarizedExchanges - 1,
      kept_exchange: selection.keptExchange,
      absorbed_action_step_count: selection.coveredActionSteps,
    },
  });
}

interface CheckpointPlan {
  groups: [number, number][];
  /** Whether an earlier checkpoint is in the request (the new one builds on it). */
  activeEmergency: boolean;
  newlyAbsorbed: Record<string, any>[];
  newlyAbsorbedLabel: string;
  transcriptEnd: number;
  transcriptActionStep: number;
  packetFields: Record<string, any>;
  coverage: Record<string, any>;
}

async function writeCheckpoint(
  args: { messages: Record<string, any>[]; tools: Record<string, any>[] }, writer: SummaryWriter,
  prior: EmergencyPrior, plan: CheckpointPlan,
): Promise<Record<string, any>> {
  const { messages, tools } = args;
  const { summaryProvider, catalog, domain } = writer;
  const { groups, activeEmergency } = plan;
  if (activeEmergency && (prior.checkpoint === null || prior.checkpointSha256 === null
    || prior.checkpointRef === null || prior.transcriptRef === null)) {
    throw new PolicySessionError("active emergency coverage lacks its checkpoint evidence");
  }
  const [taskMessage, projectMessage] = protectedSemantics(messages, groups);
  if (taskMessage === null || projectMessage === null) {
    throw new ContextCompactionRejected("semantic fallback lacks protected task/project semantics");
  }
  let previousCheckpoint: Record<string, any> | null = null;
  if (activeEmergency) {
    previousCheckpoint = {};
    for (const field of CHECKPOINT_FIELDS) previousCheckpoint[field] = clonePy(prior.checkpoint![field]);
  }
  const header = {
    exact_task: get(taskMessage, "content") ?? null,
    embodiment_semantics: get(projectMessage, "content") ?? null,
    tool_semantics: clonePy([...tools]),
    previous_checkpoint: previousCheckpoint,
    instruction: domain.emergencyCheckpointInstruction,
  };
  const packetBlocks: Record<string, any>[] = [{
    type: "text",
    text: "<rollout_compaction_input>\n" + dumpsCompact(header) + "\n</rollout_compaction_input>",
  }];
  const frontier = activeEmergency ? prior.summarizedActionSteps : 0;
  const capsuleMessages = prior.episodeCapsules
    .filter((capsule) => {
      const coverage = get(capsule, "coverage");
      const through = isMapping(coverage) && hasKey(coverage, "through_action_step_index")
        ? coverage.through_action_step_index : -1;
      return num(through) >= frontier;
    })
    .map((capsule) => semanticCapsuleSlot(capsule));
  if (capsuleMessages.length) {
    packetBlocks.push(...packetMessageBlocks("existing_episode_capsules", capsuleMessages, domain));
  }
  packetBlocks.push(...packetMessageBlocks(plan.newlyAbsorbedLabel, plan.newlyAbsorbed, domain));
  const compactorMessages = [{ role: "user", content: packetBlocks }];
  const violations = compactorPrivateViolations(compactorMessages, domain);
  if (violations.length) {
    throw new ContextCompactionRejected("rollout_compaction_private_evidence: " + violations.slice(0, 12).join(", "));
  }
  const sourceRefs = packetSourceEventIds(plan.newlyAbsorbed).map((eventId) => ({ event_id: eventId }));
  // The current visible ref is cumulative: replacing an older L6 slot must
  // never make an earlier exact ActionStep transcript undiscoverable.
  const absorbedRef = catalog.putBytes(canonicalBytesPy(
    messages.slice(groups[0][0], plan.transcriptEnd).map((message) => compactorHistoryMessage(message, domain)),
  ), {
    kind: "emergency_policy_visible_transcript", suffix: ".json", mediaType: "application/json",
    roundIndex: null, actionStep: plan.transcriptActionStep,
  });
  const packetBody = {
    schema: ROLLOUT_PACKET_SCHEMA,
    layer: "emergency_session_l6",
    previous_checkpoint_sha256: activeEmergency ? prior.checkpointSha256 : null,
    ...plan.packetFields,
    source_refs: sourceRefs,
    exact_transcript: { ...clonePy(absorbedRef), open: recoverableArtifactOpen(absorbedRef) },
    compactor_messages: compactorMessages,
    trigger: "post_l5_emergency_high_watermark",
  };
  const packetSha = canonicalSha256(packetBody);
  const packetRef = writer.artifacts.putJson({ ...packetBody, packet_sha256: packetSha }, {
    kind: "lossless_context_semantic_packet",
  });
  const summary = await summaryProvider.summarize({ messages: compactorMessages, sourceRefs });
  checkSummaryIdentity(summary, sourceRefs, "context compactor");
  const checkpoint = validateCheckpoint(summary.content, domain);
  const checkpointEstimatedTokens = estimatedTokens(checkpoint);
  const checkpointBudgetRecord = checkpointBudget(checkpointEstimatedTokens, summary.metadata, writer.limits);
  const checkpointBody = {
    schema: EMERGENCY_CHECKPOINT_SCHEMA,
    revision: prior.compactionCount + 1,
    previous_checkpoint_sha256: activeEmergency ? prior.checkpointSha256 : null,
    packet_sha256: packetSha,
    coverage: plan.coverage,
    source_refs: sourceRefs,
    checkpoint,
  };
  const checkpointSha = canonicalSha256(checkpointBody);
  const checkpointRef = writer.artifacts.putJson({ ...checkpointBody, checkpoint_sha256: checkpointSha }, {
    kind: "emergency_context_checkpoint",
  });
  return {
    checkpoint,
    checkpoint_sha256: checkpointSha,
    checkpoint_ref: checkpointRef,
    packet_ref: packetRef,
    metadata: clonePy({ ...summary.metadata }),
    checkpoint_estimated_tokens: checkpointEstimatedTokens,
    checkpoint_budget: checkpointBudgetRecord,
    exact_transcript: clonePy(absorbedRef),
  };
}
