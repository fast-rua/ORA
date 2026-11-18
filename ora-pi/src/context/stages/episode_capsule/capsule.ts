// L5's summary writer: one closed Episode, summarized by the compactor into an
// `ora.episode-context-capsule.v1` (ported from the Python Native pipeline's
// `_lossless_episode_capsule`).
//
// It stores the Episode's exact policy-visible transcript, builds the
// compaction packet from the history as the model was shown it, checks the
// packet for private evidence, calls the compactor, validates its checkpoint
// and stores the capsule.

import type { ArtifactStore } from "../../../evidence/artifacts.ts";
import type { ContextDomain } from "../../domain_defaults.ts";
import type { SummaryProvider } from "../../native.ts";
import { type ContextArtifactCatalog, recoverableArtifactOpen } from "../../memory/artifacts.ts";
import {
  EPISODE_CAPSULE_SCHEMA, ROLLOUT_PACKET_SCHEMA, compactorHistoryMessage, compactorPrivateViolations,
  estimatedTokens, observationGroups, packetMessageBlocks, packetSourceEventIds, validateCheckpoint,
} from "../../projection.ts";
import {
  ContextCompactionRejected, canonicalBytesPy, canonicalSha256, clonePy, dumpsCompact, eq, get, isInt, isMapping,
  num, tryLoads,
} from "../../pyutil.ts";
import {
  type CheckpointLimits, checkSummaryIdentity, checkpointBudget, contentText, protectedSemantics,
} from "../summaries.ts";

export interface EpisodeSummarizeArgs {
  messages: Record<string, any>[];
  tools: Record<string, any>[];
  inventory: Record<string, any>[];
  startActionStep: number;
  endActionStep: number;
}

/** What a summary writer needs: the compactor, the stores and the robot rules. */
export interface SummaryWriter {
  summaryProvider: SummaryProvider;
  catalog: ContextArtifactCatalog;
  artifacts: ArtifactStore;
  domain: ContextDomain;
  limits: CheckpointLimits;
}

/** Summarize exactly one closed Episode; ``revision`` is the capsule count after it. */
export async function writeEpisodeCapsule(
  args: EpisodeSummarizeArgs, writer: SummaryWriter, revision: number,
): Promise<Record<string, any>> {
  const { messages, tools, inventory, startActionStep, endActionStep } = args;
  const { summaryProvider, catalog, domain } = writer;
  const groups = observationGroups(messages);
  if (!(0 <= startActionStep && startActionStep < endActionStep && endActionStep < groups.length)) {
    throw new ContextCompactionRejected("episode capsule coverage is invalid");
  }
  const selected = inventory.slice(startActionStep, endActionStep);
  const rounds = new Set(selected.map((item) => JSON.stringify(get(item, "session_round") ?? null)));
  if (rounds.size !== 1 || selected.some((item) => get(item, "complete") !== true)) {
    throw new ContextCompactionRejected("episode capsule must cover one closed Episode");
  }
  const sessionRound = get(selected[0], "session_round");
  if (!isInt(sessionRound) && typeof sessionRound !== "boolean") {
    throw new ContextCompactionRejected("episode capsule lacks its Session round");
  }
  const start = groups[startActionStep][0];
  const end = groups[endActionStep][0];
  const exactMessages = messages.slice(start, end).map((message) => compactorHistoryMessage(message, domain));
  const transcriptRef = catalog.putBytes(canonicalBytesPy(exactMessages), {
    kind: "episode_policy_visible_transcript", suffix: ".json", mediaType: "application/json",
    roundIndex: num(sessionRound), actionStep: startActionStep,
  });
  const verdictMessage = messages.slice(start, end).find((message) => eq(
    get(message, "_context_source"), `${domain.officialRoundResultSourcePrefix}${num(sessionRound)}`,
  ));
  if (verdictMessage === undefined) {
    throw new ContextCompactionRejected("episode capsule lacks its exact official round result");
  }
  const rawVerdict = contentText(get(verdictMessage, "content"));
  const opening = `<${domain.officialRoundResultTag}>\n`;
  const closing = `\n</${domain.officialRoundResultTag}>`;
  const openAt = rawVerdict.indexOf(opening);
  let verdict: unknown;
  let malformed = openAt < 0;
  if (!malformed) {
    const rest = rawVerdict.slice(openAt + opening.length);
    const closeAt = rest.indexOf(closing);
    const decoded = tryLoads(closeAt >= 0 ? rest.slice(0, closeAt) : rest);
    if (decoded.ok) verdict = decoded.value;
    else malformed = true;
  }
  if (malformed) throw new ContextCompactionRejected("episode capsule official round result is malformed");
  if (!isMapping(verdict) || !eq(get(verdict, "round"), sessionRound) || typeof get(verdict, "passed") !== "boolean") {
    throw new ContextCompactionRejected("episode capsule official round result identity drift");
  }
  const [taskMessage, projectMessage] = protectedSemantics(messages, groups);
  if (taskMessage === null || projectMessage === null) {
    throw new ContextCompactionRejected("episode capsule lacks protected task/project semantics");
  }
  const header = {
    exact_task: get(taskMessage, "content") ?? null,
    embodiment_semantics: get(projectMessage, "content") ?? null,
    tool_semantics: clonePy([...tools]),
    session_round: sessionRound,
    instruction: domain.episodeCapsuleInstruction,
  };
  const packetBlocks: Record<string, any>[] = [{
    type: "text",
    text: "<episode_compaction_input>\n" + dumpsCompact(header) + "\n</episode_compaction_input>",
  }];
  const compactorSource = messages.slice(start, end).filter((message) => get(message, "_protected_context") !== true);
  packetBlocks.push(...packetMessageBlocks("closed_episode_post_lossless_projection", compactorSource, domain));
  const compactorMessages = [{ role: "user", content: packetBlocks }];
  const violations = compactorPrivateViolations(compactorMessages, domain);
  if (violations.length) {
    throw new ContextCompactionRejected("episode_compaction_private_evidence: " + violations.slice(0, 12).join(", "));
  }
  const sourceEventIds = packetSourceEventIds(compactorSource);
  const sourceRefs = sourceEventIds.map((eventId) => ({ event_id: eventId }));
  const packetBody = {
    schema: ROLLOUT_PACKET_SCHEMA,
    layer: "episode_capsule_l5",
    session_round: sessionRound,
    coverage: {
      from_action_step_index: startActionStep,
      through_action_step_index: endActionStep - 1,
      action_step_count: endActionStep - startActionStep,
    },
    source_refs: sourceRefs,
    exact_transcript: transcriptRef,
    compactor_messages: compactorMessages,
    trigger: "closed_episode_provider_high_watermark",
  };
  const packetSha = canonicalSha256(packetBody);
  const packetRef = writer.artifacts.putJson({ ...packetBody, packet_sha256: packetSha }, {
    kind: "episode_context_capsule_packet",
  });
  const summary = await summaryProvider.summarize({ messages: compactorMessages, sourceRefs });
  checkSummaryIdentity(summary, sourceRefs, "episode compactor");
  const checkpoint = validateCheckpoint(summary.content, domain);
  const checkpointEstimatedTokens = estimatedTokens(checkpoint);
  const checkpointBudgetRecord = checkpointBudget(checkpointEstimatedTokens, summary.metadata, writer.limits);
  const capsuleBody = {
    schema: EPISODE_CAPSULE_SCHEMA,
    revision,
    session_round: sessionRound,
    coverage: packetBody.coverage,
    official_verdict: clonePy({ ...verdict }),
    exact_transcript: { ...clonePy(transcriptRef), open: recoverableArtifactOpen(transcriptRef) },
    source_refs: sourceRefs,
    packet_sha256: packetSha,
    summary: checkpoint,
  };
  const capsuleSha = canonicalSha256(capsuleBody);
  const capsuleRef = writer.artifacts.putJson({ ...capsuleBody, capsule_sha256: capsuleSha }, {
    kind: "episode_context_capsule",
  });
  return {
    ...capsuleBody,
    capsule_sha256: capsuleSha,
    capsule_ref: capsuleRef,
    packet_ref: packetRef,
    metadata: clonePy({ ...summary.metadata }),
    checkpoint_estimated_tokens: checkpointEstimatedTokens,
    checkpoint_budget: checkpointBudgetRecord,
  };
}
