import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App, ReplayApp } from "./App.tsx";
import { lang, setLang } from "./lib/i18n.ts";
import { replayOptions } from "./lib/replay.ts";
import "./styles.css";

// A link's `lang=en|zh` chooses the page's language for this page only (an exported video's).
const params = new URLSearchParams(location.search);
const asked = params.get("lang");
if ((asked === "en" || asked === "zh") && asked !== lang()) setLang(asked, false);
// `?replay=1`: a recorded run (`ora replay`), else the live run.
const replay = replayOptions(location.search);

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      {replay ? <ReplayApp options={replay} /> : <App />}
    </StrictMode>,
  );
}
