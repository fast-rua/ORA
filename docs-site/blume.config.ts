import { defineConfig } from "blume";
import { readFileSync } from "node:fs";

// The pages are canonical under ../docs; scripts/prepare-content.mjs converts them into
// .generated/content before every dev, build and validate run. Edit ../docs, not .generated.
export default defineConfig({
  title: "ORA",
  description: "Open Robot-Use Agents: run a language model as a robot policy.",
  content: { root: ".generated/content" },
  logo: { image: "/logo.svg", text: "ORA" },
  theme: {
    accent: { light: "#2c4366", dark: "#4ecdc4" },
    background: { light: "#ffffff", dark: "#1c1d20" },
    mode: "system",
  },
  markdown: { externalLinks: true },
  search: {
    popular: [
      { label: "First episode", href: "/quickstart", icon: "rocket" },
      { label: "Installation", href: "/getting-started/installation", icon: "download" },
      { label: "Run on one GPU", href: "/getting-started/single-gpu", icon: "cpu" },
      { label: "Free play", href: "/usage/play", icon: "gamepad-2" },
      { label: "Command-line reference", href: "/usage/cli-reference", icon: "terminal" },
    ],
  },
  navigation: {
    actions: [
      { label: "Blog", href: "https://fastrua.ai/blog/" },
      { label: "GitHub", href: "https://github.com/fast-rua/ORA" },
    ],
    sidebar: {
      items: JSON.parse(readFileSync(new URL("./navigation.json", import.meta.url), "utf8")),
    },
  },
  // A local preview only (the published docs are fastrua.ai/ora/docs/); builds keep the /ORA
  // prefix so asset-path mistakes show.
  deployment: { site: "https://fastrua.ai", base: process.env.ORA_DOCS_BASE ?? "/ORA" },
  seo: { robots: true, sitemap: true, og: { enabled: false } },
});
