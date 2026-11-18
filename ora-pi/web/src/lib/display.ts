// How the conversation is drawn, on the live page and in a replay alike:
//   compact   (the default) one line per thought ("Thought for 12 s") and per tool call (its name
//             and a short line of its arguments), each opened by a click; an observation is one
//             line, its images folded away
//   detailed  each tool call as a card with its step's image, each observation with its images
// and whether every thought starts open ("Expand all thinking"). The page's choice is remembered
// by this browser (localStorage, when it allows); a link's `view=detailed` and
// `thinking=expanded` choose for that page only (an exported video's look).

export type ViewMode = "compact" | "detailed";

export interface Display {
  view: ViewMode;
  /** Every thought starts open. */
  thinking: boolean;
}

export const DEFAULT_DISPLAY: Display = { view: "compact", thinking: false };

const STORE_KEY = "ora.web.display";

function stored(): Partial<Display> {
  try {
    const value = JSON.parse(globalThis.localStorage?.getItem(STORE_KEY) ?? "null");
    return value && typeof value === "object" ? value : {};
  } catch {
    return {};
  }
}

/** The display a page starts with: the link's choice, else this browser's, else compact. */
export function initialDisplay(search: string = globalThis.location?.search ?? ""): Display {
  const params = new URLSearchParams(search);
  const remembered = stored();
  const view = params.get("view") ?? remembered.view;
  const thinking = params.get("thinking");
  return {
    view: view === "detailed" || view === "compact" ? view : DEFAULT_DISPLAY.view,
    thinking: thinking === "expanded" ? true : thinking === "collapsed" ? false
      : typeof remembered.thinking === "boolean" ? remembered.thinking : DEFAULT_DISPLAY.thinking,
  };
}

export function rememberDisplay(display: Display): void {
  try {
    globalThis.localStorage?.setItem(STORE_KEY, JSON.stringify(display));
  } catch {
    // a browser that keeps nothing: the choice lasts this page
  }
}
