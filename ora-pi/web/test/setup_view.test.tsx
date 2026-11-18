// @vitest-environment jsdom
// A real cell's setup before its worker starts (components/SetupView.tsx): the checklist with each step's
// state, the latest progress lines, how it ended, and where its questions are answered.

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, test } from "vitest";

import { SetupView, setupTitle } from "../src/components/SetupView.tsx";
import { setLang } from "../src/lib/i18n.ts";

afterEach(cleanup);

const STEPS = [
  { id: "devices", title: "设备", state: "done", detail: "四台相机，两条总线" },
  { id: "table", title: "桌面高度", state: "running", detail: "标定之前测的" },
  { id: "verify", title: "验证", state: "missing", detail: "" },
];

test("a running setup lists its steps, its latest lines and where to answer", () => {
  setLang("en");
  render(<SetupView setup={{ phase: "running", steps: STEPS, lines: ["measuring the table", "front camera sees it"],
    event: { progress: "front camera sees it" } }} />);
  expect(screen.getByText("Cell setup: running")).toBeTruthy();
  const items = [...document.querySelectorAll(".setup-step")].map((item) => item.textContent);
  expect(items).toEqual(["✓设备四台相机，两条总线", "✻桌面高度标定之前测的", "○验证"]);
  expect(document.querySelector(".setup-lines")?.textContent).toBe("measuring the table\nfront camera sees it");
  expect(screen.getByText(/answer them there, in the terminal or on the cell board/)).toBeTruthy();
});

test("a failed setup says why and that no arm was enabled", () => {
  setLang("zh");
  render(<SetupView setup={{ phase: "failed", steps: [{ ...STEPS[1], state: "failed", why: "没有看到桌面" }],
    result: { failed: "table", why: "没有看到桌面" } }} />);
  expect(screen.getByText("机器人设置停止：手臂没有通电")).toBeTruthy();
  expect(document.querySelector(".setup-step.failed .note")?.textContent).toBe("没有看到桌面");
  expect(document.querySelector(".setup-end")?.textContent).toBe("没有看到桌面");
  setLang("en");
});

test("the title follows the phase", () => {
  expect(setupTitle({ phase: "status" })).toBe("status");
  expect(setupTitle({ phase: "done", skipped: true })).toBe("skipped");
  expect(setupTitle({ phase: "done" })).toBe("done");
  expect(setupTitle({})).toBe("status");
});
