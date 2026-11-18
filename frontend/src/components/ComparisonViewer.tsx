import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AcceptedRegion, getRunImageUrl, RunImageKind } from '../services/api';
import { useInspectionSelection } from '../context/InspectionSelectionContext';
import {
  computeFitTransform,
  computeZoomToBBox,
  panBy,
  ViewerTransform,
  zoomAtPoint,
} from '../utils/viewerTransform';
import { DefectOverlay } from './DefectOverlay';

export const VIEWER_TABS: { key: RunImageKind; label: string }[] = [
  { key: 'original', label: 'Original' },
  { key: 'preprocessed', label: 'Preprocessed' },
  { key: 'difference', label: 'Difference' },
  { key: 'mask', label: 'Mask' },
  { key: 'filtered_mask', label: 'Filtered Mask' },
  { key: 'bboxes', label: 'Bounding Boxes' },
  { key: 'contours', label: 'Contours' },
  { key: 'final_overlay', label: 'Final Overlay' },
  { key: 'pattern_overlay', label: 'Pattern Overlay' },
];

export interface ComparisonViewerProps {
  uploadId: string;
  imageWidth: number;
  imageHeight: number;
  regions: AcceptedRegion[];
}

export const ComparisonViewer: React.FC<ComparisonViewerProps> = ({
  uploadId,
  imageWidth,
  imageHeight,
  regions,
}) => {
  const { activeTab, setActiveTab, zoomRequest, clearZoomRequest } = useInspectionSelection();

  const viewportRef = useRef<HTMLDivElement | null>(null);
  const [viewportSize, setViewportSize] = useState({ width: 720, height: 460 });
  const [transform, setTransform] = useState<ViewerTransform>(() =>
    computeFitTransform({ width: 720, height: 460 }, { width: imageWidth, height: imageHeight })
  );

  // Drag-to-pan state
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  // Before/After split slider state
  const [compareMode, setCompareMode] = useState(false);
  const [sliderPct, setSliderPct] = useState(50);

  // Interactive SVG DefectOverlay layer toggles
  const [showBoxes, setShowBoxes] = useState(true);
  const [showContours, setShowContours] = useState(true);
  const [showLabels, setShowLabels] = useState(true);

  const fitToViewport = useCallback(() => {
    const el = viewportRef.current;
    const vw = el ? el.clientWidth || 720 : viewportSize.width;
    const vh = el ? el.clientHeight || 460 : viewportSize.height;
    setTransform(computeFitTransform({ width: vw, height: vh }, { width: imageWidth, height: imageHeight }));
  }, [imageWidth, imageHeight, viewportSize.width, viewportSize.height]);

  // Measure viewport & fit on mount or resize
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;

    const updateSize = () => {
      const vw = el.clientWidth || 720;
      const vh = el.clientHeight || 460;
      setViewportSize({ width: vw, height: vh });
      setTransform(computeFitTransform({ width: vw, height: vh }, { width: imageWidth, height: imageHeight }));
    };

    updateSize();
    const ro = new ResizeObserver(updateSize);
    ro.observe(el);
    return () => ro.disconnect();
  }, [imageWidth, imageHeight]);

  // Respond to "Zoom to defect" requests from DefectTable
  useEffect(() => {
    if (!zoomRequest) return;
    const el = viewportRef.current;
    const vw = el ? el.clientWidth || 720 : viewportSize.width;
    const vh = el ? el.clientHeight || 460 : viewportSize.height;
    setTransform(computeZoomToBBox({ width: vw, height: vh }, zoomRequest.bbox));
    clearZoomRequest();
  }, [zoomRequest, clearZoomRequest, viewportSize.width, viewportSize.height]);

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const el = viewportRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const anchorX = e.clientX - rect.left;
    const anchorY = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    setTransform((prev) => zoomAtPoint(prev, factor, anchorX, anchorY));
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Ignore clicks originating on interactive SVG defect targets
    if ((e.target as Element)?.closest('[data-defect-id]')) {
      return;
    }
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging || !dragStartRef.current) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    setTransform((prev) => panBy(prev, dx, dy));
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    setIsDragging(false);
    dragStartRef.current = null;
    try {
      e.currentTarget.releasePointerCapture?.(e.pointerId);
    } catch {
      // Ignore if pointer capture wasn't set
    }
  };

  const handleZoomStep = (factor: number) => {
    const el = viewportRef.current;
    const cx = (el ? el.clientWidth || viewportSize.width : viewportSize.width) / 2;
    const cy = (el ? el.clientHeight || viewportSize.height : viewportSize.height) / 2;
    setTransform((prev) => zoomAtPoint(prev, factor, cx, cy));
  };

  const handleReset = () => {
    const el = viewportRef.current;
    const vw = el ? el.clientWidth || viewportSize.width : viewportSize.width;
    const vh = el ? el.clientHeight || viewportSize.height : viewportSize.height;
    setTransform({
      scale: 1,
      tx: (vw - imageWidth) / 2,
      ty: (vh - imageHeight) / 2,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const panStep = 32;
    if (e.key === '+' || e.key === '=') {
      e.preventDefault();
      handleZoomStep(1.25);
    } else if (e.key === '-' || e.key === '_') {
      e.preventDefault();
      handleZoomStep(1 / 1.25);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setTransform((prev) => panBy(prev, panStep, 0));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setTransform((prev) => panBy(prev, -panStep, 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setTransform((prev) => panBy(prev, 0, panStep));
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setTransform((prev) => panBy(prev, 0, -panStep));
    } else if (e.key === '0') {
      e.preventDefault();
      fitToViewport();
    }
  };

  const activeLabel = VIEWER_TABS.find((t) => t.key === activeTab)?.label ?? activeTab;
  const activeImgSrc = getRunImageUrl(uploadId, activeTab);
  const originalImgSrc = getRunImageUrl(uploadId, 'original');
  const transformCss = `translate(${transform.tx}px, ${transform.ty}px) scale(${transform.scale})`;

  return (
    <section
      aria-labelledby="comparison-viewer-heading"
      className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden"
    >
      {/* Header & View Tabs */}
      <div className="p-4 bg-slate-50 border-b border-slate-200 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3
              id="comparison-viewer-heading"
              className="text-xs font-bold text-slate-900 uppercase tracking-wide"
            >
              Interactive Inspection &amp; Comparison Viewer
            </h3>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Scroll to zoom around cursor, drag to pan, or focus the canvas and use + / - and arrow keys.
            </p>
          </div>
          <span className="text-[11px] font-mono text-slate-600 bg-white px-2.5 py-1 rounded border border-slate-200">
            Zoom: {Math.round(transform.scale * 100)}%
          </span>
        </div>

        {/* 9 Image Tabs */}
        <div
          role="tablist"
          aria-label="Inspection imagery views"
          className="flex flex-wrap gap-1 bg-slate-200/70 p-1 rounded-lg"
        >
          {VIEWER_TABS.map((tab) => {
            const isSelected = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                type="button"
                role="tab"
                aria-selected={isSelected}
                data-tab-key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                  isSelected
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Controls Bar: Zoom/Pan, Before/After Compare, Overlay Toggles */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Zoom and pan controls">
            <button
              type="button"
              onClick={() => handleZoomStep(1.25)}
              aria-label="Zoom in"
              className="px-2.5 py-1 text-xs font-semibold bg-white hover:bg-slate-100 text-slate-800 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              Zoom In (+)
            </button>
            <button
              type="button"
              onClick={() => handleZoomStep(1 / 1.25)}
              aria-label="Zoom out"
              className="px-2.5 py-1 text-xs font-semibold bg-white hover:bg-slate-100 text-slate-800 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              Zoom Out (-)
            </button>
            <button
              type="button"
              onClick={fitToViewport}
              className="px-2.5 py-1 text-xs font-semibold bg-white hover:bg-slate-100 text-slate-800 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              Fit
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="px-2.5 py-1 text-xs font-semibold bg-white hover:bg-slate-100 text-slate-800 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              Reset (100%)
            </button>

            <button
              type="button"
              onClick={() => setCompareMode((prev) => !prev)}
              aria-pressed={compareMode}
              className={`ml-1 px-2.5 py-1 text-xs font-semibold rounded-md border transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                compareMode
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-white hover:bg-slate-100 text-slate-800 border-slate-300'
              }`}
            >
              {compareMode ? 'Before/After: ON' : 'Before/After Slider'}
            </button>
          </div>

          {/* Interactive SVG Overlay Toggles */}
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-700" role="group" aria-label="Defect overlay layers">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              SVG Overlay:
            </span>
            <label className="inline-flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showBoxes}
                onChange={(e) => setShowBoxes(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Boxes</span>
            </label>
            <label className="inline-flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showContours}
                onChange={(e) => setShowContours(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Contours</span>
            </label>
            <label className="inline-flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Labels</span>
            </label>
          </div>
        </div>

        {/* Before/After Slider Range Input */}
        {compareMode && (
          <div className="flex items-center gap-3 pt-1 text-xs text-slate-700">
            <span className="font-semibold text-slate-800">Original</span>
            <input
              type="range"
              min={0}
              max={100}
              value={sliderPct}
              onChange={(e) => setSliderPct(Number(e.target.value))}
              aria-label="Before and after comparison split position"
              className="flex-1 accent-indigo-600"
            />
            <span className="font-semibold text-indigo-700">{activeLabel}</span>
            <span className="font-mono text-[11px] text-slate-500 w-9 text-right">{sliderPct}%</span>
          </div>
        )}
      </div>

      {/* Interactive Zoom / Pan Viewport */}
      <div
        ref={viewportRef}
        tabIndex={0}
        role="region"
        aria-label={`Inspection image canvas showing ${activeLabel}. Press plus or minus to zoom, arrow keys to pan.`}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onKeyDown={handleKeyDown}
        className={`relative h-[460px] w-full bg-slate-950 overflow-hidden select-none focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500 ${
          isDragging ? 'cursor-grabbing' : 'cursor-grab'
        }`}
      >
        {/* Primary Active Layer (or Right Side of Compare) */}
        <div
          style={{
            width: imageWidth,
            height: imageHeight,
            transform: transformCss,
            transformOrigin: '0 0',
          }}
          className="absolute top-0 left-0"
        >
          <img
            data-testid="viewer-active-image"
            src={activeImgSrc}
            alt={`Wafer inspection ${activeLabel} view`}
            draggable={false}
            width={imageWidth}
            height={imageHeight}
            className="block w-full h-full pointer-events-none"
          />

          {/* Interactive SVG Defect Overlay in exact original pixel space */}
          <DefectOverlay
            regions={regions}
            imageWidth={imageWidth}
            imageHeight={imageHeight}
            showBoxes={showBoxes}
            showContours={showContours}
            showLabels={showLabels}
          />
        </div>

        {/* Before/After Original Layer Clipped to Left sliderPct% */}
        {compareMode && (
          <>
            <div
              className="absolute inset-0 pointer-events-none overflow-hidden"
              style={{ clipPath: `inset(0 ${100 - sliderPct}% 0 0)` }}
            >
              <div
                style={{
                  width: imageWidth,
                  height: imageHeight,
                  transform: transformCss,
                  transformOrigin: '0 0',
                }}
                className="absolute top-0 left-0"
              >
                <img
                  data-testid="viewer-compare-original-image"
                  src={originalImgSrc}
                  alt="Wafer inspection Original comparison view"
                  draggable={false}
                  width={imageWidth}
                  height={imageHeight}
                  className="block w-full h-full pointer-events-none"
                />
              </div>
            </div>
            {/* Vertical Split Divider Line */}
            <div
              aria-hidden="true"
              style={{ left: `${sliderPct}%` }}
              className="absolute top-0 bottom-0 w-0.5 bg-indigo-400 shadow-sm pointer-events-none"
            />
            <span className="absolute top-2 left-2 px-2 py-0.5 rounded bg-slate-900/80 text-[10px] font-semibold text-white pointer-events-none">
              Original
            </span>
            <span className="absolute top-2 right-2 px-2 py-0.5 rounded bg-indigo-900/85 text-[10px] font-semibold text-white pointer-events-none">
              {activeLabel}
            </span>
          </>
        )}
      </div>

      {/* Visible Severity Overlay Legend */}
      <div className="px-4 py-2.5 bg-slate-50 border-t border-slate-200 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-slate-700">
        <span className="font-bold uppercase tracking-wider text-[10px] text-slate-500">
          Severity Overlay Legend:
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block w-5 h-0 border-t-2 border-dashed border-blue-600" aria-hidden="true" />
          <span className="font-mono font-semibold text-blue-800">[L] LOW</span>
          <span className="text-slate-500">(thin dashed blue)</span>
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block w-5 h-0 border-t-2 border-solid border-orange-600" aria-hidden="true" />
          <span className="font-mono font-semibold text-orange-800">[M] MEDIUM</span>
          <span className="text-slate-500">(medium solid orange)</span>
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block w-5 h-1 border-y-2 border-red-600" aria-hidden="true" />
          <span className="font-mono font-semibold text-red-800">[H] HIGH</span>
          <span className="text-slate-500">(thick double red)</span>
        </span>
      </div>
    </section>
  );
};
