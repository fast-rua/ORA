// The extension UI of Pi's RPC protocol, shared by every client attached to a run.
//
// Pi binds one uiContext per session; Pi's own RPC mode has one client. Here
// several clients attach: one controller and any number of observers.
// Notifications (notify, setStatus, setWidget, setTitle, set_editor_text) go
// to every client (ORA's `notifyLocal` to the terminal alone); dialogs (select, confirm, input, editor) go to the
// controller only and are answered by it alone. A client that disconnects
// detaches: its open dialogs resolve as cancelled. Requests and responses are
// Pi's (`extension_ui_request`, `extension_ui_response`), so a standard Pi RPC
// client needs nothing ORA-specific.
//
// `base` is another uiContext the composite also drives: the attached Pi
// TUI's (its InteractiveMode builds its uiContext through this composite,
// tui/host.ts). It receives the notifications too. A dialog goes to the RPC
// client whose prompt is running (it asked, so it answers), else to the TUI,
// else to the controlling RPC client.

import { randomUUID } from "node:crypto";

type Json = any;

/** One attached RPC client, as the UI needs it. */
export interface UiClient {
  readonly id: number;
  send(message: Json): void;
}

interface Pending {
  clientId: number;
  resolve(response: Json): void;
}

/** Text in, the same text out: what a theme does where nothing is drawn. */
const PLAIN_THEME: Json = new Proxy({}, {
  get: (_target, key) => (key === "name" ? "ora-rpc" : (...args: Json[]) => String(args[args.length - 1] ?? "")),
});

export class RpcUi {
  private readonly pending = new Map<string, Pending>();
  private readonly clients: () => Iterable<UiClient>;
  private readonly controller: () => UiClient | null;
  /** The RPC client whose prompt is running: its dialogs are its own. */
  driver: number | null = null;

  constructor(clients: () => Iterable<UiClient>, controller: () => UiClient | null) {
    this.clients = clients;
    this.controller = controller;
  }

  private client(id: number | null): UiClient | null {
    if (id === null) return null;
    for (const client of this.clients()) if (client.id === id) return client;
    return null;
  }

  private broadcast(request: Json): void {
    const message = { type: "extension_ui_request", id: randomUUID(), ...request };
    for (const client of this.clients()) client.send(message);
  }

  /** A dialog for the controller; its default when no one may answer or the call is aborted. */
  private dialog<T>(opts: Json, fallback: T, request: Json, parse: (response: Json) => T,
                    viaBase: (() => Promise<T>) | null): Promise<T> {
    if (opts?.signal?.aborted) return Promise.resolve(fallback);
    const driver = this.client(this.driver);
    if (driver === null && viaBase) return viaBase();
    const controller = driver ?? this.controller();
    if (controller === null) return Promise.resolve(fallback);
    const id = randomUUID();
    return new Promise<T>((resolve) => {
      let timer: NodeJS.Timeout | null = null;
      const finish = (value: T) => {
        if (timer) clearTimeout(timer);
        opts?.signal?.removeEventListener?.("abort", onAbort);
        this.pending.delete(id);
        resolve(value);
      };
      const onAbort = () => finish(fallback);
      opts?.signal?.addEventListener?.("abort", onAbort, { once: true });
      if (opts?.timeout) timer = setTimeout(() => finish(fallback), opts.timeout);
      this.pending.set(id, { clientId: controller.id, resolve: (response) => finish(parse(response)) });
      controller.send({ type: "extension_ui_request", id, ...request });
    });
  }

  /** An `extension_ui_response` from a client; only the client asked may answer. */
  respond(clientId: number, response: Json): boolean {
    const pending = this.pending.get(String(response?.id));
    if (!pending || pending.clientId !== clientId) return false;
    pending.resolve(response);
    return true;
  }

  /** The client went away: its dialogs end as cancelled. */
  detach(clientId: number): void {
    for (const pending of [...this.pending.values()]) {
      if (pending.clientId === clientId) pending.resolve({ cancelled: true });
    }
  }

  /** Open dialogs end as cancelled (the run is over). */
  cancelAll(): void {
    for (const pending of [...this.pending.values()]) pending.resolve({ cancelled: true });
  }

  /** The uiContext to bind to the session (Pi's RPC-mode surface), over `base` when a TUI has one. */
  context(base: Json = null): Json {
    const valueOf = (r: Json) => (r?.cancelled ? undefined : "value" in (r ?? {}) ? r.value : undefined);
    const ui: Json = {
      select: (title: string, options: string[], opts?: Json) => this.dialog(opts, undefined,
        { method: "select", title, options, timeout: opts?.timeout }, valueOf,
        base ? () => base.select(title, options, opts) : null),
      confirm: (title: string, message: string, opts?: Json) => this.dialog(opts, false,
        { method: "confirm", title, message, timeout: opts?.timeout },
        (r) => (r?.cancelled ? false : r?.confirmed === true), base ? () => base.confirm(title, message, opts) : null),
      input: (title: string, placeholder?: string, opts?: Json) => this.dialog(opts, undefined,
        { method: "input", title, placeholder, timeout: opts?.timeout }, valueOf,
        base ? () => base.input(title, placeholder, opts) : null),
      editor: (title: string, prefill?: string) => this.dialog(undefined, undefined,
        { method: "editor", title, prefill }, valueOf, base ? () => base.editor(title, prefill) : null),
      notify: (message: string, type?: string) => {
        base?.notify?.(message, type);
        this.broadcast({ method: "notify", message, notifyType: type });
      },
      // ORA's own: a notice for the terminal alone (free play's announcements, which every RPC
      // client already receives as `ora_play_state` and says in its own words and language).
      notifyLocal: (message: string, type?: string) => {
        base?.notify?.(message, type);
      },
      setStatus: (key: string, text: string | undefined) => {
        base?.setStatus?.(key, text);
        this.broadcast({ method: "setStatus", statusKey: key, statusText: text });
      },
      setWidget: (key: string, content: Json, options?: Json) => {
        base?.setWidget?.(key, content, options);
        // Pi's RPC mode carries string lines only; component factories stay with the terminal.
        if (content === undefined || Array.isArray(content)) {
          this.broadcast({ method: "setWidget", widgetKey: key, widgetLines: content, widgetPlacement: options?.placement });
        }
      },
      setTitle: (title: string) => {
        base?.setTitle?.(title);
        this.broadcast({ method: "setTitle", title });
      },
      setEditorText: (text: string) => {
        base?.setEditorText?.(text);
        this.broadcast({ method: "set_editor_text", text });
      },
      pasteToEditor: (text: string) => ui.setEditorText(text),
      getEditorText: () => base?.getEditorText?.() ?? "",
      onTerminalInput: (handler: Json) => base?.onTerminalInput?.(handler) ?? (() => undefined),
      async custom(...args: Json[]) {
        return base?.custom ? base.custom(...args) : undefined;
      },
      get theme() {
        return base?.theme ?? PLAIN_THEME;
      },
      getAllThemes: () => base?.getAllThemes?.() ?? [],
      getTheme: (name: string) => base?.getTheme?.(name),
      setTheme: (theme: Json) => base?.setTheme?.(theme) ?? { success: false, error: "no terminal to theme" },
      getToolsExpanded: () => base?.getToolsExpanded?.() ?? false,
      setToolsExpanded: (expanded: boolean) => base?.setToolsExpanded?.(expanded),
    };
    // Terminal-only surfaces: the attached TUI draws them; RPC clients have none.
    for (const name of ["setWorkingMessage", "setWorkingVisible", "setWorkingIndicator", "setHiddenThinkingLabel",
      "setFooter", "setHeader", "addAutocompleteProvider", "setEditorComponent"]) {
      ui[name] = (...args: Json[]) => base?.[name]?.(...args);
    }
    ui.getEditorComponent = () => base?.getEditorComponent?.();
    return ui;
  }
}
