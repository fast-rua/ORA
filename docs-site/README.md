# ORA documentation site

[Blume](https://github.com/haydenbleasel/blume) builds the ORA documentation from the
pages in [`docs/`](../docs/README.md) for a local preview and for CI's checks. The published
documentation is https://fastrua.ai/ora/docs/, which the FastRUA site builds from the same pages
of each release; this site is not deployed.
[Maintain the documentation](../docs/developer/documentation.md) is the contributor guide;
this file describes the site's own layout.

## Develop

Use Node.js 22.19 or later (CI uses Node.js 24) and [uv](https://docs.astral.sh/uv/).
From the repository root, `./scripts/docs serve` and `./scripts/docs build` wrap these:

```bash
npm ci --prefix docs-site
npm run dev --prefix docs-site          # http://localhost:4321/ORA
npm run build --prefix docs-site        # writes docs-site/dist/
npm run preview --prefix docs-site      # serves the static build
npm run validate --prefix docs-site     # strict link, anchor and asset check
npm run test:content --prefix docs-site # sidebar coverage and conversion checks
```

`ORA_DOCS_BASE` overrides the `/ORA` path prefix (for example `ORA_DOCS_BASE=/`). Set
`ORA_DOCS_PYTHON` to an interpreter that has `griffe` to skip the isolated uv environment
for the Python API reference.

## Where things live

| Path | Contents |
| --- | --- |
| `../docs/` | The pages. Edit here; never under `.generated/`. |
| `navigation.json` | The sidebar: sections, labels and order. Every page must appear in it. |
| `site.json` | Repository files published as pages, the source-link prefix and the Griffe pin. |
| `blume.config.ts` | Title, logo, colors, header links, search shortcuts and the deployment path. |
| `theme.css` | The few styles ORA adds to Blume's theme. |
| `scripts/prepare-content.mjs` | Converts `../docs` into `.generated/content` before every run. |
| `scripts/generate-robot-pages.mjs` | Generates the robot and task pages from the runtime's own files. |
| `scripts/prepare-python-docs.py` | Renders `::: module.Object` lines with Griffe, statically. |
| `tests/content.test.mjs` | Sidebar coverage, leftover-syntax and Python API checks. |

`.generated/`, `public/` and `dist/` are build outputs and are not committed.

## What the conversion does

- A folder's `README.md` becomes its `index` page, so `docs/reference/README.md` is
  served at `/reference`.
- The repository files listed in `site.json` (`externalPages`: contributing, examples, the
  viewer guide and the RoboDojo installer) are published at the docs
  paths given there.
- Relative links to published pages are re-pointed at the converted files. Links to other
  repository files go to the public GitHub source (`sourceRepository`). A link that leaves
  the repository stops the build.
- `:::` lines become a heading per object, its signature, its docstring, a table of public
  attributes and an entry per public method. Griffe reads the source with inspection off,
  so neither ORA nor its simulators are imported.
- Admonitions (`!!! type "Title"`), grid cards and button rows become Blume callouts and
  cards. Pages that use them, or Mermaid diagrams, are written as MDX, with braces,
  autolinks and stray angle brackets in prose escaped.

## Publishing

`../.github/workflows/ci.yml` builds and checks the site on every pull request
(`./scripts/docs build`); it never deploys. The documentation readers see is
https://fastrua.ai/ora/docs/: the FastRUA site converts the same `docs/` pages, the same
navigation and the same generated robot and task pages of a release into its own layout.
