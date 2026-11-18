// The operator's messages in a free-play session, as the terminal and the web page show them.
//
// Display only: what the model reads is what the Supervisor delivers. A message is
//   queued      typed while an instruction runs; it joins that instruction and the agent
//               reads it before its next model call (episode_play.py); or, sent as a
//               `queued_instruction` (`next`), held by the Supervisor until the running
//               instruction ends, then it opens the next one
//   delivered   the Supervisor delivered it at an observation step: as the instruction it
//               opened (joined false) or joined to the running one (joined true)
// From the input hook (`submitted`) and the Supervisor's announcements
// (`apply`: instruction, message_queued, message_queued_next, message_delivered). Every change is
// handed to `onChange` (the TUI and Pi RPC clients: `ora_play_message`).
// The instructions that ended (`endings`: the Supervisor's instruction records,
// with when this process heard of them) let a client mark where each ended.

type Json = any;

export interface PlayMessage {
  id: string;
  text: string;
  state: "queued" | "delivered";
  /** True when it joined a running instruction, false when it opened one, null while unknown. */
  joined: boolean | null;
  /** Held by the Supervisor until the running instruction ends (a `queued_instruction`), then the next one. */
  next: boolean;
  /** The instruction it opened or joins (1-based), when known. */
  instruction: number | null;
  /** The observation step the agent read it at. */
  step: number | null;
  /** When it was typed or admitted, and delivered (ms since the epoch, this process's clock). */
  at: number;
  deliveredAt: number | null;
}

const LIMIT = 500;

export class PlayMessages {
  private readonly byId = new Map<string, PlayMessage>();
  /** The instructions that ended (ora.play-instruction.v1 records) and when (`at`, ms). */
  readonly endings: Json[] = [];
  onChange: ((message: PlayMessage) => void) | null = null;

  list(): PlayMessage[] {
    return [...this.byId.values()];
  }

  get(id: string): PlayMessage | null {
    return this.byId.get(id) ?? null;
  }

  private upsert(id: string, change: Partial<PlayMessage>): PlayMessage {
    const previous = this.byId.get(id);
    const next: PlayMessage = {
      id, text: "", state: "queued", joined: null, next: false, instruction: null, step: null, at: Date.now(), deliveredAt: null,
      ...previous, ...Object.fromEntries(Object.entries(change).filter(([, value]) => value !== undefined)),
    };
    // A delivered message stays delivered (an announcement may arrive after its delivery).
    if (previous?.state === "delivered") next.state = "delivered";
    this.byId.set(id, next);
    while (this.byId.size > LIMIT) this.byId.delete(this.byId.keys().next().value!);
    this.onChange?.(next);
    return next;
  }

  /**
   * Typed in a terminal or a page and admitted to the inbox; `running` is the running instruction's number.
   * `queued`: sent as a `queued_instruction`, held until that instruction ends (with none running it opens one).
   */
  submitted(id: string, text: string, running: number | null, queued = false): PlayMessage {
    // The Supervisor's announcement came first: it knows where the message went.
    if (this.byId.has(id)) return this.upsert(id, { text });
    if (queued && running !== null) return this.upsert(id, { text, instruction: running, joined: false, next: true });
    return this.upsert(id, { text, instruction: running ?? undefined, joined: running === null ? null : true });
  }

  /** The Supervisor's announcement (ora.play-state.v1), when it is about a message. */
  apply(record: Json): PlayMessage | null {
    if (record?.state === "instruction_finished" && record.instruction && typeof record.instruction === "object") {
      this.endings.push({ ...record.instruction, at: Date.now() });
      if (this.endings.length > LIMIT) this.endings.splice(0, this.endings.length - LIMIT);
      return null;
    }
    const message = record?.message;
    if (!message || typeof message.id !== "string") return null;
    const text = typeof message.text === "string" ? message.text : undefined;
    const instruction = Number.isInteger(message.instruction) ? Number(message.instruction) : undefined;
    switch (record.state) {
      case "instruction":
        return this.upsert(message.id, { text, instruction, joined: false, next: false });
      case "message_queued":
        return this.upsert(message.id, { text, instruction, joined: true });
      case "message_queued_next":
        // Held until the running instruction ends; its number then is the next one's.
        return this.upsert(message.id, { text, instruction, joined: false, next: true });
      case "message_delivered":
        return this.upsert(message.id, {
          text, instruction, joined: Boolean(message.joined), next: false, state: "delivered",
          step: Number.isInteger(message.observation_step) ? Number(message.observation_step) : undefined,
          deliveredAt: Date.now(),
        });
      default:
        return null;
    }
  }

  /** Messages still waiting for the agent to read them. */
  queued(): PlayMessage[] {
    return this.list().filter((message) => message.state === "queued");
  }

  /** Messages the Supervisor holds for the next instruction, oldest first. */
  queuedNext(): PlayMessage[] {
    return this.queued().filter((message) => message.next);
  }
}
