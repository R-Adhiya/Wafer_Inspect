import { describe, expect, it } from 'vitest';
import {
  clampScale,
  computeFitTransform,
  computeZoomToBBox,
  panBy,
  viewportToImagePoint,
  zoomAtPoint,
} from './viewerTransform';

describe('viewerTransform pure zoom & pan mathematics', () => {
  it('clamps scale within min/max bounds and guards invalid inputs', () => {
    expect(clampScale(0.05, 0.25, 12)).toBe(0.25);
    expect(clampScale(25, 0.25, 12)).toBe(12);
    expect(clampScale(2.5, 0.25, 12)).toBe(2.5);
    expect(clampScale(Number.NaN, 0.25, 12)).toBe(0.25);
    expect(clampScale(-3, 0.25, 12)).toBe(0.25);
  });

  it('computes centered fit transform inside viewport with padding', () => {
    const t = computeFitTransform({ width: 800, height: 600 }, { width: 512, height: 512 }, 16);
    // Available height = 600 - 32 = 568 -> scale = 568 / 512 = 1.109375
    expect(t.scale).toBeCloseTo(568 / 512, 5);
    expect(t.tx).toBeCloseTo((800 - 512 * t.scale) / 2, 5);
    expect(t.ty).toBeCloseTo(16, 5);
  });

  it('zoomAtPoint keeps the image pixel under the cursor stationary', () => {
    const start = { scale: 1.0, tx: 100, ty: 50 };
    const anchorX = 320;
    const anchorY = 240;

    const imgBefore = viewportToImagePoint(start, anchorX, anchorY);
    const zoomed = zoomAtPoint(start, 2.0, anchorX, anchorY);
    const imgAfter = viewportToImagePoint(zoomed, anchorX, anchorY);

    expect(zoomed.scale).toBe(2.0);
    expect(imgAfter.x).toBeCloseTo(imgBefore.x, 6);
    expect(imgAfter.y).toBeCloseTo(imgBefore.y, 6);
  });

  it('panBy translates tx and ty without mutating scale or input object', () => {
    const start = { scale: 1.5, tx: 20, ty: -10 };
    const next = panBy(start, 45, -30);

    expect(next).toEqual({ scale: 1.5, tx: 65, ty: -40 });
    expect(start).toEqual({ scale: 1.5, tx: 20, ty: -10 });
  });

  it('computeZoomToBBox centers the defect bounding box in the viewport', () => {
    const viewport = { width: 800, height: 400 };
    const bbox: [number, number, number, number] = [100, 200, 20, 20];
    const t = computeZoomToBBox(viewport, bbox);

    const centerImg = viewportToImagePoint(t, viewport.width / 2, viewport.height / 2);
    expect(centerImg.x).toBeCloseTo(110, 4);
    expect(centerImg.y).toBeCloseTo(210, 4);
    expect(t.scale).toBeGreaterThan(1.0);
  });
});
