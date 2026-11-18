/** Checks on the prepared content. Run `npm run prepare:content` first (the build does). */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const content = path.join(site, ".generated", "content");

function pages(dir = content) {
  const found = [];
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) found.push(...pages(full));
    else if (/\.mdx?$/.test(name)) found.push(path.relative(content, full).split(path.sep).join("/"));
  }
  return found;
}

function route(file) {
  const bare = file.replace(/\.mdx?$/, "").replace(/(^|\/)index$/, "");
  return "/" + bare;
}

function hrefs(items) {
  return items.flatMap((item) => (item.items ? hrefs(item.items) : item.href ? [item.href] : []));
}

const navigation = JSON.parse(readFileSync(path.join(site, "navigation.json"), "utf8"));
const routes = new Set(pages().map(route));

test("every sidebar entry is a prepared page", () => {
  const missing = hrefs(navigation).filter((href) => !routes.has(href === "/" ? "/" : href.replace(/\/$/, "")));
  assert.deepEqual(missing, []);
});

test("every prepared page is in the sidebar", () => {
  const listed = new Set(hrefs(navigation));
  const orphans = [...routes].filter((page) => !listed.has(page));
  assert.deepEqual(orphans, []);
});

test("no unconverted page syntax survives the conversion", () => {
  const leftovers = [];
  for (const file of pages()) {
    let fence = null;
    readFileSync(path.join(content, file), "utf8")
      .split("\n")
      .forEach((line, index) => {
        const marker = line.match(/^\s*(`{3,}|~{3,})/);
        if (marker) fence = fence === null ? marker[1] : marker[1][0] === fence[0] ? null : fence;
        else if (fence === null && /^:::\s+[\w.]+\s*$|^\s*(!!!|\?\?\?\+?|===) |\{ *\.md-button|<div class="grid cards"/.test(line))
          leftovers.push(`${file}:${index + 1}: ${line.trim()}`);
      });
  }
  assert.deepEqual(leftovers, []);
});

test("Python API pages carry generated reference text", () => {
  for (const file of ["api/policy.md", "api/environment.md", "api/actions.md", "api/plans.md"]) {
    const text = readFileSync(path.join(content, file), "utf8");
    assert.match(text, /^## \w+ \[#ora\./m, `${file} has no generated reference heading`);
  }
});
