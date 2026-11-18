import React, { useState } from 'react';
import { AcceptedRegion, DefectSeverity } from '../services/api';
import { useInspectionSelection } from '../context/InspectionSelectionContext';

export interface DefectOverlayProps {
  regions: AcceptedRegion[];
  imageWidth: number;
  imageHeight: number;
  showBoxes: boolean;
  showContours: boolean;
  showLabels: boolean;
}

const SEVERITY_SVG_STYLE: Record<
  DefectSeverity,
  { stroke: string; strokeWidth: number; dash?: string; double?: boolean; tag: string }
> = {
  LOW: { stroke: '#2563eb', strokeWidth: 1.5, dash: '4 3', double: false, tag: 'L' },
  MEDIUM: { stroke: '#ea580c', strokeWidth: 2, double: false, tag: 'M' },
  HIGH: { stroke: '#dc2626', strokeWidth: 2.5, double: true, tag: 'H' },
};

export const DefectOverlay: React.FC<DefectOverlayProps> = ({
  regions,
  imageWidth,
  imageHeight,
  showBoxes,
  showContours,
  showLabels,
}) => {
  const { selectedDefectId, selectDefectFromOverlay } = useInspectionSelection();
  const [hoveredRegion, setHoveredRegion] = useState<AcceptedRegion | null>(null);

  if (imageWidth <= 0 || imageHeight <= 0 || (!showBoxes && !showContours && !showLabels)) {
    return null;
  }

  return (
    <svg
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      width={imageWidth}
      height={imageHeight}
      role="img"
      aria-label={`Interactive defect overlay with ${regions.length} defects`}
      data-testid="defect-overlay-svg"
      className="absolute inset-0 w-full h-full pointer-events-none select-none overflow-visible"
    >
      {regions.map((region) => {
        const sev: DefectSeverity = region.severity ?? 'LOW';
        const style = SEVERITY_SVG_STYLE[sev] ?? SEVERITY_SVG_STYLE.LOW;
        const [bx, by, bw, bh] = region.bbox;
        const isSelected = selectedDefectId === region.id;
        const isHovered = hoveredRegion?.id === region.id;
        const contourPoints =
          region.contour && region.contour.length >= 3
            ? region.contour.map(([px, py]) => `${px},${py}`).join(' ')
            : '';

        return (
          <g
            key={region.id}
            data-testid={`overlay-defect-${region.id}`}
            data-defect-id={region.id}
            data-selected={isSelected ? 'true' : 'false'}
            role="button"
            tabIndex={0}
            aria-label={`Select defect #${region.id}, area ${region.area_px} pixels, severity ${sev}`}
            aria-pressed={isSelected}
            className="pointer-events-auto cursor-pointer focus:outline-none"
            onClick={(e) => {
              e.stopPropagation();
              selectDefectFromOverlay(region.id);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                e.stopPropagation();
                selectDefectFromOverlay(region.id);
              }
            }}
            onMouseEnter={() => setHoveredRegion(region)}
            onMouseLeave={() =>
              setHoveredRegion((prev) => (prev?.id === region.id ? null : prev))
            }
          >
            {/* Selection highlight halo */}
            {isSelected && (
              <rect
                x={Math.max(0, bx - 4)}
                y={Math.max(0, by - 4)}
                width={bw + 8}
                height={bh + 8}
                fill="rgba(79, 70, 229, 0.14)"
                stroke="#4f46e5"
                strokeWidth={3}
                rx={2}
              />
            )}

            {/* Simplified contour polygon */}
            {showContours && contourPoints && (
              <polygon
                points={contourPoints}
                fill={isSelected ? 'rgba(79, 70, 229, 0.22)' : isHovered ? 'rgba(255,255,255,0.18)' : 'none'}
                stroke={isSelected ? '#4f46e5' : style.stroke}
                strokeWidth={isSelected ? style.strokeWidth + 1 : style.strokeWidth}
                strokeDasharray={style.dash}
              />
            )}

            {/* Severity-styled bounding box */}
            {showBoxes && (
              <>
                {style.double && (
                  <rect
                    x={Math.max(0, bx - 3)}
                    y={Math.max(0, by - 3)}
                    width={bw + 6}
                    height={bh + 6}
                    fill="none"
                    stroke={isSelected ? '#4f46e5' : style.stroke}
                    strokeWidth={1.2}
                  />
                )}
                <rect
                  x={bx}
                  y={by}
                  width={bw}
                  height={bh}
                  fill="rgba(255,255,255,0.01)"
                  stroke={isSelected ? '#4f46e5' : style.stroke}
                  strokeWidth={isSelected ? style.strokeWidth + 1 : style.strokeWidth}
                  strokeDasharray={style.dash}
                />
              </>
            )}

            {/* Invisible hit area if only labels are enabled */}
            {!showBoxes && !showContours && (
              <rect x={bx} y={by} width={Math.max(bw, 12)} height={Math.max(bh, 12)} fill="transparent" />
            )}

            {/* Optional ID badge */}
            {showLabels && (
              <g transform={`translate(${bx}, ${Math.max(14, by - 4)})`}>
                <rect
                  x={-1}
                  y={-11}
                  width={38}
                  height={13}
                  rx={2}
                  fill={isSelected ? '#4f46e5' : 'rgba(15, 23, 42, 0.82)'}
                />
                <text
                  x={2}
                  y={-1}
                  fill="#ffffff"
                  fontSize="9"
                  fontFamily="monospace"
                  fontWeight="700"
                >
                  #{region.id}[{style.tag}]
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* Hover Tooltip inside SVG coordinates */}
      {hoveredRegion && (
        <g
          data-testid="overlay-tooltip"
          className="pointer-events-none"
          transform={`translate(${Math.min(
            Math.max(8, hoveredRegion.bbox[0] + hoveredRegion.bbox[2] + 8),
            Math.max(8, imageWidth - 150)
          )}, ${Math.min(Math.max(12, hoveredRegion.bbox[1]), Math.max(12, imageHeight - 44))})`}
        >
          <rect
            x={0}
            y={0}
            width={142}
            height={38}
            rx={4}
            fill="rgba(15, 23, 42, 0.92)"
            stroke="#94a3b8"
            strokeWidth={1}
          />
          <text x={7} y={14} fill="#f8fafc" fontSize="10" fontFamily="monospace" fontWeight="700">
            Defect #{hoveredRegion.id} ({hoveredRegion.severity})
          </text>
          <text x={7} y={29} fill="#cbd5e1" fontSize="9.5" fontFamily="monospace">
            Area: {hoveredRegion.area_px} px
          </text>
        </g>
      )}
    </svg>
  );
};
