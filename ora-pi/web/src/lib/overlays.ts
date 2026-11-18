// What a perception call found, drawn over the camera image it looked at.
//
//   vision_ground / vision_segment  boxes: candidates[].bbox_xyxy, in the pixel frame of the
//                                   grounded image (result.overlay {width, height}, else
//                                   candidate_previews[].source_dimensions)
//   mark_point                      points: the clicked pixel(s) of `view`
//                                   ({point_id, view, x, y}, or batched {view, point_ids, points_uv}),
//                                   in the pixels of that view's image (OpenCV top-left x, y)
//
// Coordinates map exactly from the original frame to the drawn image: the image
// is drawn with object-fit: contain, so its content box is the largest box of
// the image's aspect inside the stage, centred; a frame coordinate scales by
// (content size / frame size). A box is drawn on pixel edges (x0..x1), a point
// at its pixel's centre (x + 0.5).

import { t } from "./i18n.ts";

type Json = any;

export interface Size {
  width: number;
  height: number;
}

export interface Box {
  label: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  score: number | null;
  color: string;
}

export interface Point {
  label: string;
  x: number;
  y: number;
  color: string;
  status: string | null;
}

export interface Overlay {
  kind: "grounding" | "mark";
  tool: string;
  /** The camera whose image the coordinates belong to. */
  camera: string | null;
  /** The pixel frame of the coordinates; null: the camera image's own pixels. */
  frame: Size | null;
  boxes: Box[];
  points: Point[];
  caption: string;
}

/** mark_point's batch colours (ora/agent/toolkit/tools/mark_point/tool.py COLOURS), in order. */
export const MARK_COLOURS = ["rgb(255,64,64)", "rgb(64,255,255)", "rgb(255,220,64)", "rgb(96,255,96)",
  "rgb(255,128,255)", "rgb(96,160,255)", "rgb(255,160,64)", "rgb(255,255,255)"];
const BOX_COLOURS = ["#4ecdc4", "#ffb347", "#ff6b9d", "#a3e635", "#60a5fa", "#f472b6", "#facc15", "#c084fc"];

/** The tool a (wire) name calls: `perception.mark_point`, `perception_mark_point` and `mark_point` are one. */
export function toolKind(name: string): "grounding" | "mark" | null {
  const base = name.split(/[.]/).pop() ?? name;
  if (/(^|_)vision_(ground|segment)$/.test(base)) return "grounding";
  if (/(^|_)mark_point$/.test(base)) return "mark";
  return null;
}

function finite(value: Json): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function size(value: Json): Size | null {
  if (Array.isArray(value) && value.length >= 2) {
    const [width, height] = value.map(Number);
    return width > 0 && height > 0 ? { width, height } : null;
  }
  if (value && typeof value === "object") {
    const width = Number(value.width), height = Number(value.height);
    return width > 0 && height > 0 ? { width, height } : null;
  }
  return null;
}

function grounding(name: string, args: Json, result: Json): Overlay | null {
  const candidates: Json[] = Array.isArray(result?.candidates) ? result.candidates : [];
  const boxes: Box[] = [];
  candidates.forEach((candidate, index) => {
    const box = candidate?.bbox_xyxy;
    if (!Array.isArray(box) || box.length !== 4 || box.some((value: Json) => finite(value) === null)) return;
    const [x0, y0, x1, y1] = box as number[];
    const rank = finite(candidate.rank) ?? index;
    boxes.push({
      label: String(candidate.candidate_id || `T${rank}`), x0, y0, x1, y1,
      score: finite(candidate.backend_score) ?? finite(candidate.score), color: BOX_COLOURS[index % BOX_COLOURS.length],
    });
  });
  if (!boxes.length) return null;
  const previews: Json[] = Array.isArray(result?.candidate_previews) ? result.candidate_previews : [];
  const frame = size(result?.overlay) ?? size(previews.find((preview) => preview?.source_dimensions)?.source_dimensions);
  const query = typeof (result?.text ?? args?.text) === "string" ? ` “${result?.text ?? args?.text}”` : "";
  return {
    kind: "grounding", tool: name, camera: typeof (result?.camera ?? args?.camera) === "string" ? (result?.camera ?? args.camera) : null,
    frame, boxes, points: [],
    caption: `${boxes.length === 1 ? t("overlay.candidate") : t("overlay.candidates", { n: boxes.length })}${query}`,
  };
}

function marks(name: string, args: Json, result: Json): Overlay | null {
  const view = typeof args?.view === "string" ? args.view : null;
  if (view === null) return null;
  const camera = view === "candidate_agentview" ? "agentview" : view;
  const points: Point[] = [];
  if (Array.isArray(args.points_uv)) {
    const ids: Json[] = Array.isArray(args.point_ids) ? args.point_ids : [];
    const outcomes: Json[] = Array.isArray(result?.marks) ? result.marks : [];
    args.points_uv.forEach((uv: Json, index: number) => {
      if (!Array.isArray(uv) || finite(uv[0]) === null || finite(uv[1]) === null) return;
      points.push({ label: String(ids[index] ?? `#${index + 1}`), x: uv[0], y: uv[1],
        color: MARK_COLOURS[index % MARK_COLOURS.length], status: outcomes[index]?.status ?? null });
    });
  } else if (finite(args.x) !== null && finite(args.y) !== null) {
    points.push({ label: String(args.point_id ?? "point"), x: args.x, y: args.y, color: MARK_COLOURS[0],
      status: typeof result?.status === "string" ? result.status : null });
  }
  if (!points.length) return null;
  return { kind: "mark", tool: name, camera, frame: null, boxes: [], points,
    caption: points.map((point) => `${point.label} (${point.x}, ${point.y})${point.status ? ` ${point.status}` : ""}`).join(" · ") };
}

/** The overlay of one tool call, or null when it draws nothing (or failed). */
export function overlayFor(name: string, args: Json, result: Json, status = "done"): Overlay | null {
  const kind = toolKind(name);
  if (kind === null || status === "error") return null;
  if (kind === "grounding") return result ? grounding(name, args, result) : null;
  // A mark is known from its arguments, before (and whatever) the measurement says.
  return marks(name, args ?? {}, result);
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The content box of an image of `natural` size drawn with object-fit: contain in `box`. */
export function containRect(natural: Size, box: Size): Rect {
  const scale = Math.min(box.width / natural.width, box.height / natural.height);
  const width = natural.width * scale, height = natural.height * scale;
  return { x: (box.width - width) / 2, y: (box.height - height) / 2, width, height };
}

/** From the overlay's frame (or the image's own pixels) to the stage's pixels. */
export function frameToDisplay(frame: Size, rect: Rect): (x: number, y: number) => [number, number] {
  const sx = rect.width / frame.width, sy = rect.height / frame.height;
  return (x, y) => [rect.x + x * sx, rect.y + y * sy];
}

export interface Drawn {
  boxes: Array<Box & { left: number; top: number; right: number; bottom: number }>;
  points: Array<Point & { cx: number; cy: number }>;
}

/** Where every box and point of `overlay` lands on a stage of `box` size showing an image of `natural` size. */
export function layout(overlay: Overlay, natural: Size, box: Size): Drawn {
  const rect = containRect(natural, box);
  const map = frameToDisplay(overlay.frame ?? natural, rect);
  return {
    boxes: overlay.boxes.map((item) => {
      const [left, top] = map(item.x0, item.y0);
      const [right, bottom] = map(item.x1, item.y1);
      return { ...item, left, top, right, bottom };
    }),
    points: overlay.points.map((item) => {
      const [cx, cy] = map(item.x + 0.5, item.y + 0.5);
      return { ...item, cx, cy };
    }),
  };
}

/** Draw `overlay` on a canvas covering the stage (CSS pixels `box`, device pixel ratio `ratio`). */
export function drawOverlay(context: CanvasRenderingContext2D, overlay: Overlay, natural: Size, box: Size, ratio = 1): void {
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, box.width, box.height);
  const drawn = layout(overlay, natural, box);
  context.font = "600 12px ui-sans-serif, system-ui, sans-serif";
  context.textBaseline = "top";
  const tag = (text: string, x: number, y: number, color: string) => {
    const width = context.measureText(text).width + 8;
    const top = Math.max(0, y - 18);
    context.fillStyle = "rgba(10, 12, 14, 0.82)";
    context.fillRect(x, top, width, 17);
    context.fillStyle = color;
    context.fillText(text, x + 4, top + 2);
  };
  for (const item of drawn.boxes) {
    context.lineWidth = 2;
    context.strokeStyle = "rgba(0, 0, 0, 0.6)";
    context.strokeRect(item.left - 1, item.top - 1, item.right - item.left + 2, item.bottom - item.top + 2);
    context.strokeStyle = item.color;
    context.strokeRect(item.left, item.top, item.right - item.left, item.bottom - item.top);
    tag(item.score === null ? item.label : `${item.label} ${item.score.toFixed(2)}`, item.left, item.top, item.color);
  }
  for (const item of drawn.points) {
    context.lineWidth = 2;
    context.strokeStyle = "rgba(0, 0, 0, 0.7)";
    context.beginPath();
    context.arc(item.cx, item.cy, 7, 0, Math.PI * 2);
    context.stroke();
    context.strokeStyle = item.color;
    context.beginPath();
    context.arc(item.cx, item.cy, 6, 0, Math.PI * 2);
    context.moveTo(item.cx - 10, item.cy);
    context.lineTo(item.cx + 10, item.cy);
    context.moveTo(item.cx, item.cy - 10);
    context.lineTo(item.cx, item.cy + 10);
    context.stroke();
    tag(item.label, item.cx + 9, item.cy - 4, item.color);
  }
}
