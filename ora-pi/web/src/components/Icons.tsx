// A few line icons, drawn here (no icon package).

import type { ReactElement } from "react";

const paths: Record<string, ReactElement> = {
  send: <path d="M4 12h13M12 6l6 6-6 6" />,
  stop: <rect x="7" y="7" width="10" height="10" rx="1.5" />,
  play: <path d="M8 5.5v13l10.5-6.5z" />,
  pause: <path d="M8 5.5v13M16 5.5v13" />,
  prev: <path d="M15 6l-6 6 6 6" />,
  next: <path d="M9 6l6 6-6 6" />,
  chevron: <path d="M9 6l6 6-6 6" />,
  live: <circle cx="12" cy="12" r="4.5" />,
  camera: <path d="M4 8h3l1.5-2h7L17 8h3v11H4zM12 16.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z" />,
  settings: <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1" />,
  close: <path d="M6 6l12 12M18 6L6 18" />,
  skip: <path d="M6 6l8 6-8 6zM17 6v12" />,
  exit: <path d="M10 5H5v14h5M14 8l4 4-4 4M18 12H9" />,
  eye: <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 14.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z" />,
  replay: <path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3M4.5 4.5v4h4" />,
  film: <path d="M4 5h16v14H4zM8 5v14M16 5v14M4 9h4M4 15h4M16 9h4M16 15h4" />,
};

export function Icon({ name, size = 16, title }: { name: keyof typeof paths | string; size?: number; title?: string }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden={title ? undefined : true} role={title ? "img" : undefined}>
      {title ? <title>{title}</title> : null}
      {paths[name] ?? null}
    </svg>
  );
}

/** ORA's mark, the terminal's pixel robot arm (base, column, shoulder joint, arm, two-finger gripper), in the accent. */
const LOGO = ["..####.......", ".##..##......", ".##..#######.", "..####....##.", "...##.....##.", "...##....####", "...##....#..#",
  "..####...#..#", ".######......"];

export function Mark({ px = 4 }: { px?: number }) {
  return (
    <svg className="mark" width={LOGO[0].length * px} height={LOGO.length * px} viewBox={`0 0 ${LOGO[0].length} ${LOGO.length}`}
      shapeRendering="crispEdges" role="img" aria-label="ORA">
      {LOGO.flatMap((row, y) => [...row].map((cell, x) => (cell === "#" ? <rect key={`${x}.${y}`} x={x} y={y} width="1" height="1" /> : null)))}
    </svg>
  );
}

/** The working indicator: the terminal's spinner glyph in the accent (drawn by the stylesheet), then what it says. */
export function Working({ label }: { label: string }) {
  return (
    <span className="working" role="status">
      <span className="spinner" aria-hidden="true" />
      {label}
    </span>
  );
}
