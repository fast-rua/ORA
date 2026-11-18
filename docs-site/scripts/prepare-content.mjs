/**
 * Convert the canonical pages under ../docs (plus the few repository files published as
 * pages, `externalPages` in ../site.json) into Blume content under .generated/content.
 *
 * Nothing here is an editable copy: .generated/ and public/ are rebuilt on every run.
 * The conversion:
 *   - README.md becomes index.md, so a folder's README is its section page;
 *   - relative links to published pages are re-pointed at the converted files, links to
 *     other repository files go to the public GitHub source, and links that escape the
 *     repository fail the build;
 *   - `::: module.Object` lines are replaced by reference text that
 *     scripts/prepare-python-docs.py extracts statically with Griffe (it never imports ORA);
 *   - Material grid cards and buttons become Blume cards, and pages that need MDX features
 *     (cards, Mermaid) are written as .mdx with prose made MDX-safe.
 */
import { execFileSync } from "node:child_process";
import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { generateRobotPages } from "./generate-robot-pages.mjs";

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repo = path.resolve(site, "..");
const docs = path.join(repo, "docs");
const out = path.join(site, ".generated", "content");
const publicDir = path.join(site, "public");

/** Site inputs that are not pages: see site.json. */
async function siteSettings() {
  const config = JSON.parse(await readFile(path.join(site, "site.json"), "utf8"));
  for (const key of ["sourceRepository", "griffe", "externalPages"])
    if (!config[key]) throw new Error(`docs-site/site.json: ${key} missing`);
  return { external: config.externalPages, sourceRepository: config.sourceRepository, griffe: config.griffe };
}

const EXCLUDED = [/^design\//, /^_/, /(^|\/)__pycache__\//, /^requirements\./];

async function walk(dir, base = dir) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...(await walk(full, base)));
    else files.push(path.relative(base, full).split(path.sep).join("/"));
  }
  return files;
}

/** docs-relative page path -> content-relative path (README.md is a folder's index). */
function contentPath(uri) {
  return uri.replace(/(^|\/)README\.md$/, "$1index.md");
}

const FENCE = /^\s*(`{3,}|~{3,})/;

/** Apply fn to every line outside fenced code blocks. */
function outsideFences(text, fn) {
  let fence = null;
  return text
    .split("\n")
    .map((line) => {
      const marker = line.match(FENCE);
      if (marker) {
        if (fence === null) fence = marker[1];
        else if (marker[1][0] === fence[0] && marker[1].length >= fence.length) fence = null;
        return line;
      }
      return fence === null ? fn(line) : line;
    })
    .join("\n");
}

/** Apply fn to the prose of a line, leaving inline code spans untouched. */
function outsideInlineCode(line, fn) {
  return line
    .split(/(`+[^`]*`+)/)
    .map((part, index) => (index % 2 ? part : fn(part)))
    .join("");
}

const HTML_TAGS = new Set(
  "a abbr b blockquote br code dd del details div dl dt em figcaption figure h1 h2 h3 h4 h5 h6 hr i img ins kbd li mark ol p pre s small span strong sub summary sup table tbody td th thead tr u ul video source".split(
    " ",
  ),
);

/** Make Python-Markdown prose acceptable to MDX: braces, autolinks, comments, placeholders. */
function mdxSafe(text) {
  text = text.replace(/<!--[\s\S]*?-->/g, "");
  return outsideFences(text, (line) =>
    outsideInlineCode(line, (prose) =>
      prose
        .replace(/<(https?:\/\/[^>\s]+)>/g, "[$1]($1)")
        .replace(/[{}]/g, (brace) => "\\" + brace)
        .replace(/<(\/?)([A-Za-z][\w-]*)([^>]*)>/g, (tag, slash, name, rest) =>
          HTML_TAGS.has(name.toLowerCase())
            ? `<${slash}${name}${rest.replace(/\bclass=/g, "className=").replace(/<br>/g, "<br />")}>`
            : `&lt;${slash}${name}${rest}&gt;`,
        )
        .replace(/<(br|img|source|hr)(\s[^>]*?)?(?<!\/)>/g, "<$1$2 />"),
    ),
  );
}

/** Material "grid cards" lists and button rows -> Blume CardGroup/Card. */
function cards(text) {
  // Button rows: a <div ... markdown> holding only [label](href){ .md-button ... } lines.
  text = text.replace(/<div class="([^"]*)" markdown>\n([\s\S]*?)\n<\/div>/g, (block, cls, body) => {
    if (cls.split(/\s+/).includes("grid")) return block;
    const buttons = [...body.matchAll(/^\[([^\]]+)\]\(([^)\s]+)\)\{[^}]*\.md-button[^}]*\}\s*$/gm)];
    if (!buttons.length || buttons.length !== body.trim().split("\n").length) return block;
    const items = buttons.map(([, label, href]) => `  <Card title="${label.replace(/"/g, "&quot;")}" href="${href}" />`);
    return `<CardGroup cols={${Math.min(buttons.length, 3)}}>\n${items.join("\n")}\n</CardGroup>`;
  });
  // Grid cards: "-   **Title**" items with an indented body.
  return text.replace(/<div class="grid cards" markdown>\n([\s\S]*?)\n<\/div>/g, (block, body) => {
    const items = body.split(/^-   /m).slice(1);
    if (!items.length) return block;
    const rendered = items.map((item) => {
      const [head, ...rest] = item.split("\n");
      const title = head.replace(/^\*\*(.*)\*\*\s*$/, "$1").trim();
      const content = rest
        .map((line) => line.replace(/^ {4}/, ""))
        .join("\n")
        .trim();
      return `<Card title="${title.replace(/"/g, "&quot;")}">\n\n${content}\n\n</Card>`;
    });
    return `<CardGroup cols={2}>\n\n${rendered.join("\n\n")}\n\n</CardGroup>`;
  });
}

/** Material admonitions (`!!! type "Title"` + 4-space body) -> Blume directives (MDX only). */
const ADMONITION_TYPES = {
  note: "note", info: "info", tip: "tip", hint: "tip", warning: "warning", caution: "warning",
  attention: "warning", danger: "danger", failure: "danger", bug: "danger", example: "note",
  important: "note", abstract: "info", question: "info", success: "tip",
};
function admonitions(text) {
  const lines = text.split("\n");
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const head = lines[i].match(/^!!!\s+(\w+)(?:\s+"([^"]*)")?\s*$/);
    if (!head) {
      out.push(lines[i]);
      continue;
    }
    const type = ADMONITION_TYPES[head[1].toLowerCase()] ?? "note";
    const body = [];
    while (i + 1 < lines.length && (/^ {4}/.test(lines[i + 1]) || lines[i + 1].trim() === "")) {
      i++;
      body.push(lines[i].replace(/^ {4}/, ""));
    }
    while (body.length && body.at(-1).trim() === "") body.pop();
    out.push(head[2] ? `:::${type}[${head[2]}]` : `:::${type}`, ...body, ":::", "");
  }
  return out.join("\n");
}

function stripAttributeLists(text, page, warnings) {
  return outsideFences(text, (line) =>
    outsideInlineCode(line, (prose) =>
      prose.replace(/(\)|\]|^#.*?)\s*\{\s*[.#:][^}]*\}/g, (match, before) => {
        warnings.push(`${page}: dropped attribute list ${match.slice(before.length).trim()}`);
        return before;
      }),
    ),
  );
}

async function main() {
  const { external, sourceRepository, griffe } = await siteSettings();
  await rm(path.join(site, ".generated"), { recursive: true, force: true });
  await rm(publicDir, { recursive: true, force: true });
  await mkdir(out, { recursive: true });
  await mkdir(publicDir, { recursive: true });

  // Published pages: every Markdown file under docs/ except the excluded ones, plus EXTERNAL_PAGES.
  const pages = new Map(); // docs-relative uri -> absolute source path
  for (const uri of await walk(docs)) {
    if (!uri.endsWith(".md") || EXCLUDED.some((pattern) => pattern.test(uri))) continue;
    pages.set(uri, path.join(docs, uri));
  }
  for (const [uri, source] of Object.entries(external)) pages.set(uri, path.join(repo, source));
  const bySource = new Map([...pages].map(([uri, source]) => [path.resolve(source), uri]));

  // Python reference text for every `:::` identifier on any page.
  const identifiers = new Set();
  for (const source of pages.values())
    for (const [, id] of (await readFile(source, "utf8")).matchAll(/^:::\s+([\w.]+)\s*$/gm)) identifiers.add(id);
  let reference = {};
  if (identifiers.size) {
    // Griffe at the version site.json pins, in an isolated uv environment that
    // never touches the project's own virtual environment. ORA_DOCS_PYTHON overrides it with
    // an interpreter that already has griffe.
    const script = [path.join(site, "scripts", "prepare-python-docs.py"), ...identifiers];
    let command;
    let args;
    if (process.env.ORA_DOCS_PYTHON) {
      command = process.env.ORA_DOCS_PYTHON;
      args = script;
    } else {
      command = "uv";
      args = ["run", "--isolated", "--no-project", "--python", "3.11", "--with", griffe, "python", ...script];
    }
    reference = JSON.parse(
      execFileSync(command, args, { cwd: repo, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }),
    );
  }

  const warnings = [];
  for (const [uri, source] of pages) {
    let text = await readFile(source, "utf8");
    const target = contentPath(uri);

    text = text.replace(/^:::\s+([\w.]+)\s*$/gm, (line, id) => {
      if (!reference[id]) throw new Error(`${uri}: no Python reference for ${id}`);
      return reference[id];
    });

    // Links: published pages -> converted files; other repository files -> public source.
    const rewrite = (url) => {
      if (/^([a-z][a-z0-9+.-]*:|\/\/|#|\/)/i.test(url)) return url;
      const [pathPart, fragment] = url.split("#", 2);
      if (!pathPart) return url;
      const resolved = path.resolve(path.dirname(source), decodeURIComponent(pathPart));
      if (!resolved.startsWith(repo + path.sep) && resolved !== repo)
        throw new Error(`${uri}: link escapes the repository: ${url}`);
      const suffix = fragment === undefined ? "" : `#${fragment}`;
      // A link to an external page (site.json externalPages) names
      // its docs/ path, not its source file.
      const docsUri = resolved.startsWith(docs + path.sep) ? path.relative(docs, resolved).split(path.sep).join("/") : null;
      const linked = bySource.get(resolved) ?? (docsUri && pages.has(docsUri) ? docsUri : undefined);
      if (linked) {
        const relative = path.posix.relative(path.posix.dirname(target), contentPath(linked));
        return (relative.startsWith(".") ? relative : `./${relative}`) + suffix;
      }
      if (resolved.startsWith(path.join(docs, "assets") + path.sep)) {
        const relative = path.posix.relative(path.posix.dirname(target), path.relative(docs, resolved).split(path.sep).join("/"));
        return relative + suffix;
      }
      if (existsSync(resolved)) {
        const relative = path.relative(repo, resolved).split(path.sep).join("/");
        return sourceRepository + relative.split("/").map(encodeURIComponent).join("/") + suffix;
      }
      warnings.push(`${uri}: unresolved link ${url}`);
      return url;
    };
    // Whole lines, not prose between code spans: link text is often a code span itself.
    text = outsideFences(text, (line) =>
      line
        .replace(/(!?\[[^\]\n]*?\]\()([^\s)]+)/g, (match, start, url) => start + rewrite(url))
        .replace(/(\s(?:src|href)=")([^"]+)(")/g, (match, start, url, end) => start + rewrite(url) + end),
    );

    const usesCards = /<div class="[^"]*" markdown>/.test(text);
    const usesMermaid = /^\s*```mermaid\s*$/m.test(text);
    const usesAdmonitions = /^!!!\s+\w+/m.test(text);
    const mdx = usesCards || usesMermaid || usesAdmonitions;
    if (mdx) {
      if (usesAdmonitions) text = admonitions(text);
      // Button and card markup is rebuilt as JSX after the prose is made MDX-safe.
      const placeholders = [];
      text = cards(text).replace(/<\/?Card(Group)?[^>]*>/g, (tag) => {
        placeholders.push(tag);
        return `\u0000${placeholders.length - 1}\u0000`;
      });
      text = mdxSafe(text).replace(/\u0000(\d+)\u0000/g, (match, index) => placeholders[Number(index)]);
    }
    text = stripAttributeLists(text, uri, warnings);

    const destination = path.join(out, mdx ? target.replace(/\.md$/, ".mdx") : target);
    await mkdir(path.dirname(destination), { recursive: true });
    await writeFile(destination, text);
  }

  // Images and media that pages link relatively, and the brand mark.
  for (const folder of ["images", "videos"]) {
    const from = path.join(docs, "assets", folder);
    if (existsSync(from)) await cp(from, path.join(out, "assets", folder), { recursive: true });
  }
  await cp(path.join(docs, "assets", "images", "ora-mark.svg"), path.join(publicDir, "logo.svg"));

  const generated = await generateRobotPages(repo, out);

  for (const warning of warnings) console.warn(`warning: ${warning}`);
  const count = (await walk(out)).filter((file) => /\.mdx?$/.test(file)).length;
  console.log(
    `prepared ${pages.size} pages from docs/ and generated ${generated.length} robot and task pages into ${path.relative(site, out)}`,
  );
  if ((await stat(out)).isDirectory() && count !== pages.size + generated.length) throw new Error("page count mismatch");
}

await main();
