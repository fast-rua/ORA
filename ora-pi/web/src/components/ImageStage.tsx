// An image drawn whole (object-fit: contain) with a canvas over it for a perception overlay.

import { useEffect, useRef, useState } from "react";

import { drawOverlay, type Overlay, type Size } from "../lib/overlays.ts";

export function ImageStage({ src, alt, overlay, className, onClick, onSize }: {
  src: string;
  alt: string;
  overlay: Overlay | null;
  className?: string;
  onClick?: () => void;
  /** The natural size of each image once loaded. */
  onSize?: (size: Size) => void;
}) {
  const stage = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  // The natural size of the image now shown (a size of the previous one never maps this overlay).
  const [loaded, setLoaded] = useState<{ src: string; size: Size } | null>(null);
  const natural = loaded && loaded.src === src ? loaded.size : null;
  const [box, setBox] = useState<Size | null>(null);

  useEffect(() => {
    const element = stage.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setBox((previous) => (previous && previous.width === width && previous.height === height ? previous : { width, height }));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const context = element.getContext?.("2d");
    if (!context || !box) return;
    const ratio = window.devicePixelRatio || 1;
    element.width = Math.round(box.width * ratio);
    element.height = Math.round(box.height * ratio);
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, element.width, element.height);
    if (overlay && natural) drawOverlay(context, overlay, natural, box, ratio);
  }, [overlay, natural, box]);

  return (
    <div ref={stage} className={`stage ${className ?? ""}`} onClick={onClick}
      data-overlay={overlay ? overlay.kind : undefined}>
      <img src={src} alt={alt} draggable={false}
        onLoad={(event) => {
          const size = { width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight };
          setLoaded({ src, size });
          onSize?.(size);
        }} />
      <canvas ref={canvas} className="overlay" aria-hidden="true" />
    </div>
  );
}
