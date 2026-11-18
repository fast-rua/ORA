// An extension's dialog (Pi's extension_ui_request: select, confirm, input,
// editor), answered with Pi's extension_ui_response: {id, value} for a choice
// or text, {id, confirmed} for a confirmation, {id, cancelled: true} to
// decline. The page's own confirmations use the same card; one with `danger`
// (a real cell's command that lets the arms move or fall) is answered yes
// only by its button, as the terminal's y/N takes only `y`: Enter and Esc say no.

import { useEffect, useRef, useState } from "react";

import { t } from "../lib/i18n.ts";

type Json = any;

export type Answer = { value: string } | { confirmed: boolean } | { cancelled: true };

function strip(text: Json): string {
  return String(text ?? "").replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
}

export function Dialog({ request, onAnswer }: { request: Json; onAnswer: (answer: Answer) => void }) {
  const method = String(request.method);
  const options: string[] = Array.isArray(request.options) ? request.options.map(strip) : [];
  const [index, setIndex] = useState(0);
  const [value, setValue] = useState(String(request.prefill ?? ""));
  const card = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLInputElement & HTMLTextAreaElement>(null);

  const danger = request.danger === true;
  const no = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    (field.current ?? (danger ? no.current : null) ?? card.current)?.focus();
  }, [request.id, danger]);

  const cancel = () => onAnswer(method === "confirm" ? { confirmed: false } : { cancelled: true });
  const accept = () => {
    if (method === "confirm") onAnswer({ confirmed: true });
    else if (method === "select") onAnswer(options[index] === undefined ? { cancelled: true } : { value: options[index] });
    else onAnswer({ value });
  };

  return (
    <div className="dialog-backdrop">
      <div ref={card} className="dialog" role="dialog" aria-modal="true" aria-label={strip(request.title) || t("dialog.question")} tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            cancel();
          } else if (method === "select" && event.key === "ArrowDown") {
            event.preventDefault();
            setIndex((current) => Math.min(options.length - 1, current + 1));
          } else if (method === "select" && event.key === "ArrowUp") {
            event.preventDefault();
            setIndex((current) => Math.max(0, current - 1));
          } else if (event.key === "Enter" && !(method === "editor" && !event.ctrlKey && !event.metaKey)) {
            event.preventDefault();
            if (danger) cancel();
            else accept();
          }
        }}>
        <div className="dialog-title">{strip(request.title) || t("dialog.question")}</div>
        {request.message ? <div className="dialog-message">{strip(request.message)}</div> : null}
        {method === "select" ? (
          <div className="dialog-options" role="listbox">
            {options.map((option, at) => (
              <button type="button" key={at} role="option" aria-selected={at === index} className={at === index ? "on" : ""}
                onMouseEnter={() => setIndex(at)} onClick={() => onAnswer({ value: option })}>{option}</button>
            ))}
          </div>
        ) : null}
        {method === "input" ? (
          <input ref={field} value={value} placeholder={strip(request.placeholder)} onChange={(event) => setValue(event.target.value)} />
        ) : null}
        {method === "editor" ? (
          <textarea ref={field} rows={8} value={value} onChange={(event) => setValue(event.target.value)} />
        ) : null}
        <div className="dialog-actions">
          <button ref={no} type="button" className="ghost" onClick={cancel}>{method === "confirm" ? t("dialog.no") : t("dialog.cancel")}</button>
          <button type="button" className={danger ? "primary danger" : "primary"} onClick={accept}>
            {method === "confirm" ? t("dialog.yes") : t("dialog.ok")}
          </button>
        </div>
      </div>
    </div>
  );
}
