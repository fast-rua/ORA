# Shared frontend style

The default for ORA frontends is a minimal light interface. Use the
same template for run registries, task catalogs and review tools. A new
page inherits this style unless the user explicitly requests a different
presentation.

The public documentation site uses Blume, following the explicitly selected
FastVideo documentation layout. Its navigation and theme live in
`docs-site/navigation.json`, `docs-site/blume.config.ts` and `docs-site/theme.css`;
the Viewer retains the shared template below.

## Shared sources

| Source | Owns |
|---|---|
| `viewer/templates/base.html` | Page shell, title, viewport and shared stylesheet links |
| `viewer/static/base.css` | Portable tokens, typography, controls, tables, focus and responsive primitives |
| `viewer/static/viewer.css` | Viewer component layout and states using the shared tokens |
| `viewer/ui.py` | Shared shell rendering, including rendering outside a Flask request |
| `viewer/templates/style_guide.html` | Small example of the standard controls, table and details |

ORA's world view (`ora/orchestrator/world_view.py`) ships in the `ora` package, so it carries a byte
copy of `base.css` under `ora/orchestrator/world_view_assets/static/`;
`tests/ora/orchestrator/test_world_view.py` keeps the two equal.

Open `/style-guide` in the running Viewer to inspect the template. Example rows
on that page are labeled as examples and carry no experiment claims. The HTML
templates and CSS ship with the installed Viewer package.

All Viewer pages call the shared `page()` renderer. Add component layout to
`viewer.css`; do not paste another page theme or reset into an HTML template.
New standalone tools should reuse `base.css` and the same page structure:
navigation, a plain title, compact controls, content, then supporting details.
The personal `swiss-knife-design` skill also contains a portable HTML/CSS copy
for future frontends outside this repository. Keep the copies aligned when
changing the shared default; the project remains runnable without that skill.

## Defaults

| Element | Default |
|---|---|
| Surface | White page and panels; light gray headers and supporting areas |
| Type | System sans-serif; monospace for code and identifiers |
| Scale | 14px body, 24px page title, 18px section title, 12px supporting text |
| Spacing | 4, 8, 12, 16, 24 and 32px; 24px desktop page padding |
| Borders | 1px gray lines, 3px corner radius, no decorative shadows |
| Controls | Native inputs and quiet bordered buttons with visible keyboard focus |
| Color | Muted blue links and restrained, labeled success/failure/warning states |
| Media | Original colors and opacity; a black video backing only for letterboxing |

Prefer headings, spacing and tables over nested cards. Do not add decorative
serif headlines, slogans, rotated stamps, accent stripes, dark dashboard
panels, gradients or entrance animations. Status text remains readable without
color. Long tables scroll inside their container; the page itself fits a phone.

The one exception is the page of a run (`ora web` and `ora replay`, `ora-pi/web`): it
reproduces the project's terminal teaser (ora-site `teaser/terminal/tui.py`), a dark
coding-agent transcript in a monospace font on the left and the instruction with every
camera on the right. Its palette, at the top of `ora-pi/web/src/styles.css`, is the
teaser's: background `#0b0c0f`, panes `#111216`, text `#d6d8dd`, dim `#848992`, teal
`#4ecdc4` for marks and markers only, green `#7fd3a0` for verdicts only.

## Review

Inspect a content-heavy page and `/style-guide` on desktop and a narrow screen.
Check readable labels, keyboard focus, selected and empty states, unfiltered
images and overflow. For a review page, verify that video, round selection and
Notebook navigation still work. Frontend styling does not change recorded
evidence, official verdicts or already compiled MP4 content.
