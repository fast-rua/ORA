# Maintain the documentation

The pages live in `docs/`. `docs-site/` builds them into the documentation site
with [Blume](https://github.com/haydenbleasel/blume), the framework behind other
Hao AI Lab documentation sites. Current contracts stay under `docs/reference/`.
What goes on a page is in the [documentation style guide](docs-style.md).

## Preview and build

You need Node.js 22.19 or later and uv. From the source checkout:

```bash
./scripts/docs serve
```

This installs the site's locked dependencies on first use and serves a live
preview at `http://localhost:4321/ORA`. Editing a page in `docs/` reloads it.

```bash
./scripts/docs build
```

This writes the static site to `docs-site/dist/` and fails on a broken link,
anchor or asset, on a page missing from the sidebar, and on page syntax the
conversion does not handle. CI runs the same command. Neither command launches a
model or a simulator.

The underlying commands, if you want them one at a time:

```bash
npm ci --prefix docs-site
npm run dev --prefix docs-site          # live preview
npm run build --prefix docs-site        # static site
npm run validate --prefix docs-site     # strict link, anchor and asset check
npm run test:content --prefix docs-site # sidebar coverage and conversion checks
```

## Add or change a page

1. Edit the page in `docs/`. Use relative links to the Markdown file.
2. List a new page in `docs-site/navigation.json` under the reader's task (see
   the [style guide](docs-style.md#sections-and-where-a-page-belongs)).
3. Run `./scripts/docs build` and fix what it reports.
4. Preview wide and narrow layouts, both color modes, code blocks, tables and search.

Contributing, examples, the viewer guide and the RoboDojo
installer are published from their repository READMEs (`externalPages` in
`docs-site/site.json`). Edit those originals rather than copying them into `docs/`.

## Package READMEs

Every package directory the [repository structure](repository-structure.md#9-docs-and-the-package-readmes)
lists carries a `README.md` in the same five sections: responsibility; what
belongs here and what does not; dependency direction; entry points;
documentation. They are the source-side half of the documentation: the
[code map](../code-map.md) links each one, and each links back to the pages
here that document its contracts. When a change moves a module, adds an entry
point, changes what a package may import or adds a pinned file, update the
package's README in the same change. A new package that the structure lists
fails `tests/repository/test_structure.py` until its README exists. READMEs
under `ora/` ship in the wheel; `ora/policy/README.md` is a locked policy
asset, so editing it needs `./scripts/ora lock`.

## How a build works

Every run first converts `docs/` into `docs-site/.generated/content/`
(`docs-site/scripts/prepare-content.mjs`):

- a folder's `README.md` becomes its section page;
- links to other repository files point at the public GitHub source
  (`sourceRepository` in `docs-site/site.json`); a link that leaves the
  repository stops the build;
- `::: module.Object` lines on the Python API pages become reference text;
- the Robots and tasks pages are generated from the files the runtime reads
  (`docs-site/scripts/generate-robot-pages.mjs`): the robot manual each default
  prompt bundle gives the model, and the task pages of every environment
  (LIBERO / LIBERO-Pro, RoboTwin, RoboDojo, RoboCasa365, RoboLab) with each
  task's public instruction, action limit or completion contract, as that
  environment defines them. These
  pages exist only in the built site; link them by their site URL
  (`https://fastrua.ai/ora/docs/robots/.../`) so the link also works on GitHub.

`.generated/`, `public/` and `dist/` are build outputs and are not committed.

## Python reference

Pages under `docs/api/` name the supported symbols with `::: module.Object` lines.
Griffe reads the source statically, with inspection off, at the version pinned in
`docs-site/site.json`, so simulator dependencies and credentials are not needed.
Update the matching current-reference prose when behavior changes; generated
signatures do not replace an authority or evidence contract.

## Dependencies

`docs-site/package.json` and `docs-site/package-lock.json` pin the site's Node
packages; the Griffe pin is in `docs-site/site.json`. The runtime's `uv.lock` is
unaffected. Change a pin deliberately and rebuild.

## Hosting

The published documentation is https://fastrua.ai/ora/docs/, part of the FastRUA
site. That site builds it from the public tree of a release: the same `docs/`
pages, `docs-site/navigation.json` and the generated robot and task pages, in the
site's own layout. A page's absolute link to `https://fastrua.ai/ora/docs/<route>/`
becomes a link inside that site. The Blume build here is a local preview and CI's
check; it is not deployed. `ORA_DOCS_BASE` changes its path prefix.
