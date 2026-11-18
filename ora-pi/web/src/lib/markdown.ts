// The model's text as Markdown, never as HTML: raw HTML is shown as text, links
// open only http(s) and mailto targets, and Markdown images stay links (the
// page loads no image from elsewhere).

import { Marked } from "marked";

type Json = any;

export function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeHref(href: string): string | null {
  try {
    const url = new URL(href, "https://invalid.example/");
    return url.origin !== "https://invalid.example" && ["http:", "https:", "mailto:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

const marked = new Marked({ gfm: true, breaks: false, async: false });
marked.use({
  renderer: {
    html(token: Json) {
      return escapeHtml(String(token.text ?? token.raw ?? ""));
    },
    link(this: Json, token: Json) {
      const inner = this.parser.parseInline(token.tokens ?? []);
      const href = safeHref(String(token.href ?? ""));
      return href === null ? inner
        : `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer nofollow">${inner}</a>`;
    },
    image(token: Json) {
      const text = escapeHtml(String(token.text || token.href || "image"));
      const href = safeHref(String(token.href ?? ""));
      return href === null ? text : `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer nofollow">${text}</a>`;
    },
  },
});

export function renderMarkdown(text: string): string {
  return marked.parse(text) as string;
}
