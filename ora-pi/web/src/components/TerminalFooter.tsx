// The terminal's footer under the prompt box (the composer, or a replay's controls): the run's mode
// in the accent, where it is (free play's scene and instruction, an episode's step) and the context
// the model holds; on the right the reasoning effort. On a real robot cell the mode is the cell's
// permission mode (ask, safe, auto: lib/cell.ts), as the terminal's footer shows it; a click opens a
// small menu that switches it (App asks first for auto, as the terminal's y/N does).

import { useEffect, useRef, useState, type ReactNode } from "react";

import { modeStateText, PERMISSION_MODES, safeLimitsText, type ModeState, type PermissionMode } from "../lib/cell.ts";
import { t, useLang, type Key } from "../lib/i18n.ts";
import { tokens } from "../lib/terminal.ts";

/** What the footer says, from what the page already has (App.tsx footerInfo). */
export interface FooterInfo {
  /** Free play's scene and instruction, or an episode's step, as short phrases. */
  where: string[];
  /** The context the model held at its last call, and its window when the model says it. */
  context: number | null;
  window: number | null;
  effort: string | null;
}

/** A real cell's permission mode in the footer, and how it is switched. */
export interface FooterMode {
  state: ModeState;
  /** The cell's rig mode: `auto` needs the worker that drives the arms (`on`). */
  rigMode: string;
  /** A switch is running. */
  busy: boolean;
  onSwitch: (mode: PermissionMode) => void;
}

function ModeMenu({ mode }: { mode: FooterMode }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (event: MouseEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  const current = mode.state.mode;
  const words = (item: PermissionMode) => t(`cell.mode.label.${item}` as Key);
  return (
    <span className="term-mode-box" ref={box}>
      <button type="button" className={`term-mode-button mode-${current}`} data-testid="term-mode" data-mode={current}
        aria-haspopup="menu" aria-expanded={open} title={`${modeStateText(mode.state)} · ${t("cell.mode.title")}`}
        disabled={mode.busy} onClick={() => setOpen((value) => !value)}>
        <span aria-hidden="true">▸▸ </span>{words(current)}
      </button>
      {open ? (
        <span className="term-mode-menu" role="menu" aria-label={t("cell.mode.menu")}>
          {PERMISSION_MODES.map((item) => {
            const blocked = item === "auto" && mode.rigMode !== "on" && current !== "auto";
            return (
              <button type="button" role="menuitemradio" key={item} aria-checked={item === current} data-mode={item}
                className={`mode-${item}`} disabled={blocked} title={blocked ? t("cell.mode.autoNeedsOn") : undefined}
                onClick={() => {
                  setOpen(false);
                  if (item !== current) mode.onSwitch(item);
                }}>
                <span className="term-mode-name">{words(item)}</span>
                {item === "safe" ? <span className="term-mode-limits">{safeLimitsText(mode.state)}</span> : null}
              </button>
            );
          })}
        </span>
      ) : null}
    </span>
  );
}

export function TerminalFooter({ info, mode = null, children = null }: {
  info: FooterInfo;
  mode?: FooterMode | null;
  children?: ReactNode;
}) {
  useLang();
  const context = info.context !== null
    ? t("footer.context", { used: info.window ? `${tokens(info.context)} / ${tokens(info.window)}` : tokens(info.context) }) : null;
  const rest = [...info.where, context].filter(Boolean).map((part) => ` · ${part}`).join("");
  return (
    <div className="term-footer" data-testid="term-footer">
      <span className="term-mode">
        {mode ? <ModeMenu mode={mode} />
          : <><span className="accent" aria-hidden="true">▸▸ </span>{t("footer.robotMode")}</>}
        <span className="dim">{rest}</span>
      </span>
      <span className="term-right">
        {children}
        {info.effort ? <span className="dim effort">● {t("footer.effort", { level: info.effort })}</span> : null}
      </span>
    </div>
  );
}
