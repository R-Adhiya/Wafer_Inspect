import React, { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { AcceptedRegion, DefectSeverity, InspectionSeverity } from '../services/api';
import { useInspectionSelection } from '../context/InspectionSelectionContext';
import { EmptyState } from './EmptyState';
import { SeverityBadge } from './SeverityBadge';

export interface DefectTableProps {
  regions: AcceptedRegion[];
  inspectionSeverity?: InspectionSeverity;
  severityReason?: string;
}

type SortKey =
  | 'id'
  | 'severity'
  | 'area_px'
  | 'width_px'
  | 'height_px'
  | 'length_px'
  | 'perimeter_px'
  | 'contrast_to_noise';

type SortDir = 'asc' | 'desc';

const SEVERITY_RANK: Record<DefectSeverity, number> = {
  LOW: 1,
  MEDIUM: 2,
  HIGH: 3,
};

export const DefectTable: React.FC<DefectTableProps> = ({
  regions,
  inspectionSeverity = 'NONE',
  severityReason,
}) => {
  const {
    selectedDefectId,
    selectDefectFromTable,
    lastSelectionSource,
    requestZoomToDefect,
  } = useInspectionSelection();

  const [severityFilter, setSeverityFilter] = useState<'ALL' | DefectSeverity>('ALL');
  const [sortKey, setSortKey] = useState<SortKey>('id');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [expandedWhyIds, setExpandedWhyIds] = useState<Record<number, boolean>>({});
  const rowRefs = useRef<Record<number, HTMLTableRowElement | null>>({});

  // Scroll selected row into view when selected from DefectOverlay
  useEffect(() => {
    if (selectedDefectId != null && lastSelectionSource === 'overlay') {
      const rowEl = rowRefs.current[selectedDefectId];
      rowEl?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [selectedDefectId, lastSelectionSource]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir(key === 'id' ? 'asc' : 'desc');
    }
  };

  const toggleWhy = (id: number) => {
    setExpandedWhyIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const filteredAndSorted = useMemo(() => {
    const filtered =
      severityFilter === 'ALL'
        ? regions
        : regions.filter((r) => (r.severity ?? 'LOW') === severityFilter);

    return [...filtered].sort((a, b) => {
      let cmp = 0;
      if (sortKey === 'severity') {
        cmp = (SEVERITY_RANK[a.severity ?? 'LOW'] ?? 0) - (SEVERITY_RANK[b.severity ?? 'LOW'] ?? 0);
      } else {
        const av = Number(a[sortKey] ?? 0);
        const bv = Number(b[sortKey] ?? 0);
        cmp = av - bv;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [regions, severityFilter, sortKey, sortDir]);

  const renderSortIndicator = (key: SortKey) => {
    if (sortKey !== key) return <span className="opacity-30 ml-1">↕</span>;
    return <span className="ml-1 text-indigo-700">{sortDir === 'asc' ? '↑' : '↓'}</span>;
  };

  return (
    <section
      aria-labelledby="defect-table-heading"
      className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs"
    >
      <div className="p-3.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3
            id="defect-table-heading"
            className="text-xs font-bold text-slate-900 uppercase tracking-wide"
          >
            Defect Measurements ({filteredAndSorted.length}
            {severityFilter !== 'ALL' ? ` of ${regions.length}` : ''})
          </h3>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Click a row (or press Enter) to highlight the defect in the viewer above. All values are in original image pixels.
          </p>
        </div>

        {/* Severity Filter Pills */}
        {regions.length > 0 && (
          <div className="flex items-center gap-1" role="group" aria-label="Filter defects by severity">
            {(['ALL', 'HIGH', 'MEDIUM', 'LOW'] as const).map((lvl) => (
              <button
                key={lvl}
                type="button"
                onClick={() => setSeverityFilter(lvl)}
                aria-pressed={severityFilter === lvl}
                className={`px-2.5 py-1 rounded-md text-[11px] font-semibold border transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                  severityFilter === lvl
                    ? 'bg-slate-900 text-white border-slate-900'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>
        )}
      </div>

      {regions.length === 0 ? (
        <EmptyState
          severity={inspectionSeverity}
          severityReason={severityReason}
        />
      ) : filteredAndSorted.length === 0 ? (
        <div className="p-6 text-center text-xs text-slate-500">
          No defects match severity filter <strong>{severityFilter}</strong>.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase text-[10px] font-bold tracking-wider">
                <th className="py-2.5 px-3">
                  <button type="button" onClick={() => handleSort('id')} className="inline-flex items-center uppercase hover:text-slate-900">
                    ID {renderSortIndicator('id')}
                  </button>
                </th>
                <th className="py-2.5 px-3">
                  <button type="button" onClick={() => handleSort('severity')} className="inline-flex items-center uppercase hover:text-slate-900">
                    Severity {renderSortIndicator('severity')}
                  </button>
                </th>
                <th className="py-2.5 px-3 text-right">
                  <button type="button" onClick={() => handleSort('area_px')} className="inline-flex items-center uppercase hover:text-slate-900">
                    Area (px) {renderSortIndicator('area_px')}
                  </button>
                </th>
                <th className="py-2.5 px-2.5 text-right">
                  <button type="button" onClick={() => handleSort('width_px')} className="inline-flex items-center uppercase hover:text-slate-900">
                    W {renderSortIndicator('width_px')}
                  </button>
                </th>
                <th className="py-2.5 px-2.5 text-right">
                  <button type="button" onClick={() => handleSort('height_px')} className="inline-flex items-center uppercase hover:text-slate-900">
                    H {renderSortIndicator('height_px')}
                  </button>
                </th>
                <th className="py-2.5 px-3 text-right">
                  <button type="button" onClick={() => handleSort('length_px')} className="inline-flex items-center uppercase hover:text-slate-900">
                    Length {renderSortIndicator('length_px')}
                  </button>
                </th>
                <th className="py-2.5 px-3">Centroid</th>
                <th className="py-2.5 px-3 text-right">
                  <button type="button" onClick={() => handleSort('perimeter_px')} className="inline-flex items-center uppercase hover:text-slate-900">
                    Perim {renderSortIndicator('perimeter_px')}
                  </button>
                </th>
                <th className="py-2.5 px-3 text-right">
                  <button type="button" onClick={() => handleSort('contrast_to_noise')} className="inline-flex items-center uppercase hover:text-slate-900">
                    CNR {renderSortIndicator('contrast_to_noise')}
                  </button>
                </th>
                <th className="py-2.5 px-3">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredAndSorted.map((region) => {
                const isSelected = selectedDefectId === region.id;
                const isExpanded = Boolean(expandedWhyIds[region.id]);
                return (
                  <Fragment key={region.id}>
                    <tr
                      ref={(el) => {
                        rowRefs.current[region.id] = el;
                      }}
                      tabIndex={0}
                      role="row"
                      aria-selected={isSelected}
                      data-testid={`defect-row-${region.id}`}
                      data-selected={isSelected ? 'true' : 'false'}
                      onClick={() => selectDefectFromTable(isSelected ? null : region.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          selectDefectFromTable(isSelected ? null : region.id);
                        }
                      }}
                      className={`cursor-pointer transition-colors focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500 ${
                        isSelected
                          ? 'bg-indigo-50/90 ring-1 ring-inset ring-indigo-300'
                          : 'hover:bg-slate-50/80'
                      }`}
                    >
                      <td className="py-2.5 px-3 font-mono font-semibold text-slate-900">
                        #{region.id}
                      </td>
                      <td className="py-2.5 px-3">
                        <SeverityBadge severity={region.severity ?? 'LOW'} />
                      </td>
                      <td className="py-2.5 px-3 font-mono text-right text-slate-800">
                        {region.area_px}
                      </td>
                      <td className="py-2.5 px-2.5 font-mono text-right text-slate-700">
                        {region.width_px ?? region.bbox[2]}
                      </td>
                      <td className="py-2.5 px-2.5 font-mono text-right text-slate-700">
                        {region.height_px ?? region.bbox[3]}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-right text-slate-700">
                        {region.length_px}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-slate-700">
                        ({region.centroid[0]}, {region.centroid[1]})
                      </td>
                      <td className="py-2.5 px-3 font-mono text-right text-slate-700">
                        {region.perimeter_px}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-right text-slate-800">
                        {region.contrast_to_noise}x
                      </td>
                      <td className="py-2.5 px-3">
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              requestZoomToDefect(region.id, region.bbox);
                            }}
                            className="px-2 py-0.5 text-[11px] font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
                          >
                            Zoom to defect
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleWhy(region.id);
                            }}
                            aria-expanded={isExpanded}
                            className="text-[11px] font-medium text-slate-600 hover:text-slate-900 underline underline-offset-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded"
                          >
                            {isExpanded ? 'Hide' : 'Why'}
                          </button>
                        </div>
                      </td>
                    </tr>
                    {(isExpanded || isSelected) && (
                      <tr className={isSelected ? 'bg-indigo-50/40' : 'bg-slate-50/90'}>
                        <td colSpan={10} className="px-4 py-3 text-xs text-slate-700 space-y-1.5">
                          <p className="font-medium text-slate-900">{region.why_flagged}</p>
                          <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] font-mono text-slate-600">
                            {region.detected_area_px != null && (
                              <span>Pre-refinement Area: {region.detected_area_px} px</span>
                            )}
                            <span>Thickness: {region.thickness_px} px</span>
                            <span>Angle: {region.angle_deg}&deg;</span>
                            <span>Elongation: {region.elongation}</span>
                            <span>Eq. Diam: {region.equivalent_diameter_px} px</span>
                            <span>Polarity: {region.polarity}</span>
                            <span>Peak Contrast: {region.peak_contrast}</span>
                            <span>Edge: {region.touches_edge ? 'Rim' : 'Interior'}</span>
                            {region.area_um2 != null && (
                              <span>Calibrated Area: {region.area_um2} &mu;m&sup2;</span>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};
