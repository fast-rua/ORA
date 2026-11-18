// The robot's recorded motion of one step, played as a short animation: the original
// execution frames (native/execution-capture) at the capture's frame rate, with play,
// pause, play again and a scrubber (a bar over the picture's foot, so the picture keeps its size). `follow` keeps up with frames still arriving (the
// action executing now): it plays on as they come and never falls more than a second behind.

import { useEffect, useRef, useState } from "react";

import { t } from "../lib/i18n.ts";
import { Icon } from "./Icons.tsx";
import { ImageStage } from "./ImageStage.tsx";

export interface MotionImage {
  src: string;
  simTimeS: number;
}

/** Frames ahead of the shown one are fetched before they are needed (the gateway lets the browser cache them). */
const PRELOAD = 12;

export function MotionPlayer({ frames, fps, follow, autoPlay, playKey, caption, className, onFrame }: {
  frames: MotionImage[];
  fps: number;
  /** More frames may still arrive: keep playing as they do. */
  follow: boolean;
  autoPlay: boolean;
  /** A new key starts the motion from its first frame. */
  playKey: string;
  caption: (index: number, count: number) => string;
  className?: string;
  /** The frame shown now (the panel's caption follows it). */
  onFrame?: (index: number, count: number) => void;
}) {
  const [index, setShown] = useState(0);
  const [playing, setPlaying] = useState(autoPlay);
  const shown = useRef(0);
  const setIndex = (value: number) => {
    shown.current = value;
    setShown(value);
  };
  const count = frames.length;
  const interval = 1000 / Math.min(Math.max(fps || 10, 1), 30);
  const loaded = useRef(new Set<string>());
  // Reached the last frame by itself (not paused): frames that arrive later play on.
  const ranOut = useRef(false);

  useEffect(() => {
    if (ranOut.current && shown.current < count - 1) {
      ranOut.current = false;
      setPlaying(true);
    }
  }, [count]);

  useEffect(() => {
    setIndex(0);
    ranOut.current = false;
    setPlaying(autoPlay);
  }, [playKey, autoPlay]);

  useEffect(() => {
    for (const frame of frames.slice(index, index + PRELOAD)) {
      if (loaded.current.has(frame.src) || typeof Image === "undefined") continue;
      loaded.current.add(frame.src);
      const image = new Image();
      image.src = frame.src;
    }
  }, [frames, index]);

  useEffect(() => {
    if (!playing || count === 0) return;
    const timer = setInterval(() => {
      const at = shown.current;
      // Following a motion still being recorded: never more than about a second behind.
      if (follow && count - 1 - at > fps) setIndex(count - 1 - Math.round(fps / 2));
      else if (at + 1 < count) setIndex(at + 1);
      else if (!follow) {
        ranOut.current = true;
        setPlaying(false);
      }
    }, interval);
    return () => clearInterval(timer);
  }, [playing, count, follow, fps, interval]);

  const at = Math.min(index, Math.max(0, count - 1));
  useEffect(() => {
    if (count > 0) onFrame?.(at, count);
  }, [at, count, onFrame]);

  if (count === 0) return null;
  const frame = frames[at];
  const ended = !follow && at === count - 1 && !playing;
  return (
    <div className={`motion ${className ?? ""}`} data-testid="motion" data-frame={at} data-frames={count}>
      <ImageStage className="main" src={frame.src} alt={caption(at, count)} overlay={null} />
      <div className="motion-bar">
        <button type="button" className="icon-button" data-testid="motion-toggle"
          title={ended ? t("camera.replayMotion") : playing ? t("camera.pauseMotion") : t("camera.playMotion")}
          onClick={() => {
            ranOut.current = false;
            if (ended) setIndex(0);
            setPlaying(ended ? true : !playing);
          }}>
          <Icon name={playing ? "pause" : ended ? "replay" : "play"} />
        </button>
        <input type="range" className="scrub" aria-label={t("camera.scrub")} min={0} max={count - 1} value={at}
          onChange={(event) => {
            ranOut.current = false;
            setPlaying(false);
            setIndex(Number(event.target.value));
          }} />
        <span className="motion-count">{at + 1}/{count}</span>
      </div>
    </div>
  );
}
