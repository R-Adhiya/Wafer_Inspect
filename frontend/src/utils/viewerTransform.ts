/**
 * Pure zoom and pan transform mathematics for ComparisonViewer.
 *
 * Coordinate convention:
 *   screenX = tx + imageX * scale
 *   screenY = ty + imageY * scale
 *
 * Keeping all geometry pure makes cursor-anchored zoom, pan, fit-to-viewport,
 * and zoom-to-defect deterministic and directly unit-testable.
 */

export interface ViewerTransform {
  scale: number;
  tx: number;
  ty: number;
}

export interface Size2D {
  width: number;
  height: number;
}

export const DEFAULT_MIN_SCALE = 0.25;
export const DEFAULT_MAX_SCALE = 12.0;

/** Clamp zoom scale into [minScale, maxScale]. */
export function clampScale(
  scale: number,
  minScale: number = DEFAULT_MIN_SCALE,
  maxScale: number = DEFAULT_MAX_SCALE
): number {
  if (!Number.isFinite(scale) || scale <= 0) {
    return minScale;
  }
  return Math.max(minScale, Math.min(maxScale, scale));
}

/**
 * Compute a transform that fits `imageSize` centered inside `viewportSize`
 * with optional padding (px on each side).
 */
export function computeFitTransform(
  viewport: Size2D,
  image: Size2D,
  paddingPx: number = 16
): ViewerTransform {
  if (viewport.width <= 0 || viewport.height <= 0 || image.width <= 0 || image.height <= 0) {
    return { scale: 1, tx: 0, ty: 0 };
  }
  const availW = Math.max(1, viewport.width - 2 * paddingPx);
  const availH = Math.max(1, viewport.height - 2 * paddingPx);
  const scale = clampScale(Math.min(availW / image.width, availH / image.height), 0.1, DEFAULT_MAX_SCALE);
  const tx = (viewport.width - image.width * scale) / 2;
  const ty = (viewport.height - image.height * scale) / 2;
  return { scale, tx, ty };
}

/**
 * Zoom around a specific anchor point `(anchorX, anchorY)` in viewport coordinates
 * so the image pixel under the cursor stays fixed on screen.
 */
export function zoomAtPoint(
  current: ViewerTransform,
  factor: number,
  anchorX: number,
  anchorY: number,
  minScale: number = DEFAULT_MIN_SCALE,
  maxScale: number = DEFAULT_MAX_SCALE
): ViewerTransform {
  if (!Number.isFinite(factor) || factor <= 0) {
    return current;
  }
  const nextScale = clampScale(current.scale * factor, minScale, maxScale);
  const ratio = nextScale / current.scale;
  const nextTx = anchorX - (anchorX - current.tx) * ratio;
  const nextTy = anchorY - (anchorY - current.ty) * ratio;
  return {
    scale: nextScale,
    tx: nextTx,
    ty: nextTy,
  };
}

/** Translate viewport transform by `(dx, dy)` screen pixels. */
export function panBy(current: ViewerTransform, dx: number, dy: number): ViewerTransform {
  return {
    scale: current.scale,
    tx: current.tx + dx,
    ty: current.ty + dy,
  };
}

/** Convert a viewport coordinate `(vx, vy)` to image pixel coordinates `(ix, iy)`. */
export function viewportToImagePoint(
  transform: ViewerTransform,
  vx: number,
  vy: number
): { x: number; y: number } {
  return {
    x: (vx - transform.tx) / transform.scale,
    y: (vy - transform.ty) / transform.scale,
  };
}

/**
 * Compute a transform that centers and zooms onto a defect bounding box `(x, y, w, h)`
 * with comfortable surrounding context margin.
 */
export function computeZoomToBBox(
  viewport: Size2D,
  bbox: [number, number, number, number],
  contextFactor: number = 3.5,
  minScale: number = 1.0,
  maxScale: number = 8.0
): ViewerTransform {
  const [bx, by, bw, bh] = bbox;
  if (viewport.width <= 0 || viewport.height <= 0) {
    return { scale: 1, tx: 0, ty: 0 };
  }
  const targetW = Math.max(24, bw * contextFactor);
  const targetH = Math.max(24, bh * contextFactor);
  const rawScale = Math.min(viewport.width / targetW, viewport.height / targetH);
  const scale = clampScale(rawScale, minScale, maxScale);
  const cx = bx + bw / 2;
  const cy = by + bh / 2;
  const tx = viewport.width / 2 - cx * scale;
  const ty = viewport.height / 2 - cy * scale;
  return { scale, tx, ty };
}
