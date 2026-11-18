// Notices: the run's notifications (Pi's notify), the page's own messages, and a
// widget's lines when an extension shows one.

import { useEffect } from "react";

import { t } from "../lib/i18n.ts";
import type { Notice } from "../state.ts";
import { Icon } from "./Icons.tsx";

export function Notices({ notices, widgets, onDismiss }: {
  notices: Notice[];
  widgets: Record<string, string[]>;
  onDismiss: (id: string) => void;
}) {
  useEffect(() => {
    if (!notices.length) return;
    const timers = notices.map((item) => setTimeout(() => onDismiss(item.id),
      Math.max(0, (item.tone === "info" ? 6000 : 12000) - (Date.now() - item.at))));
    return () => timers.forEach(clearTimeout);
  }, [notices, onDismiss]);
  const lines = Object.entries(widgets);
  return (
    <div className="notices" aria-live="polite">
      {lines.map(([key, rows]) => (
        <div key={key} className="widget" title={key}>{rows.map((row, index) => <div key={index}>{row}</div>)}</div>
      ))}
      {notices.map((item) => (
        <div key={item.id} className={`notice tone-${item.tone}`} role={item.tone === "error" ? "alert" : "status"}>
          <span>{item.text}</span>
          <button type="button" className="icon-button" onClick={() => onDismiss(item.id)} title={t("notice.dismiss")}><Icon name="close" size={12} /></button>
        </div>
      ))}
    </div>
  );
}
