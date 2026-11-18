import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { PatternResult, PatternType } from '../services/api';

export interface PatternCardProps {
  pattern: PatternResult;
}

const PATTERN_BADGE_STYLES: Record<PatternType, string> = {
  EDGE_RING: 'bg-red-50 text-red-800 border-red-300',
  CENTER_CLUSTER: 'bg-orange-50 text-orange-800 border-orange-300',
  DONUT: 'bg-purple-50 text-purple-800 border-purple-300',
  LOCAL_CLUSTER: 'bg-amber-50 text-amber-900 border-amber-300',
  SCRATCH_LINE: 'bg-rose-50 text-rose-800 border-rose-300',
  RANDOM_SCATTER: 'bg-sky-50 text-sky-800 border-sky-300',
  INSUFFICIENT_DATA: 'bg-slate-100 text-slate-700 border-slate-300',
};

const PATTERN_LABELS: Record<PatternType, string> = {
  EDGE_RING: 'Edge Ring',
  CENTER_CLUSTER: 'Center Cluster',
  DONUT: 'Donut Ring',
  LOCAL_CLUSTER: 'Local Cluster',
  SCRATCH_LINE: 'Scratch Line',
  RANDOM_SCATTER: 'Random Scatter',
  INSUFFICIENT_DATA: 'Insufficient Data',
};

export const PatternCard: React.FC<PatternCardProps> = ({ pattern }) => {
  const badgeStyle = PATTERN_BADGE_STYLES[pattern.type] ?? PATTERN_BADGE_STYLES.INSUFFICIENT_DATA;
  const displayLabel = PATTERN_LABELS[pattern.type] ?? pattern.type;
  const mcFloor = typeof pattern.metrics?.mc_p_floor === 'number' ? pattern.metrics.mc_p_floor : 1 / 2001;
  const isAtMcFloor =
    pattern.p_value != null && (pattern.p_value <= mcFloor + 1e-6 || pattern.p_value < 0.001);
  const confidenceDisplay = isAtMcFloor
    ? '> 99.9%'
    : `${(pattern.confidence * 100).toFixed(1)}%`;
  const pValueDisplay =
    pattern.p_value == null
      ? 'N/A (shape/count rule)'
      : isAtMcFloor
        ? '< 0.001'
        : pattern.p_value.toFixed(4);
  const radialBins = pattern.metrics?.radial_bins ?? [];

  return (
    <div className="border border-slate-200 rounded-xl bg-white shadow-xs overflow-hidden">
      <div className="p-4 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wide">
            Wafer Spatial Pattern Diagnosis
          </h5>
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-bold uppercase tracking-wide border ${badgeStyle}`}
          >
            {displayLabel}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
          <span className="text-slate-700">
            Evidence Strength:{' '}
            <strong className="text-slate-900">{confidenceDisplay}</strong>
          </span>
          <span className="text-slate-700">
            p-value:{' '}
            <strong className="text-slate-900">{pValueDisplay}</strong>
          </span>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <p className="text-sm text-slate-800 leading-relaxed">{pattern.explanation}</p>

        <p className="text-[11px] text-slate-500">
          Evidence strength is rule-based (1 &minus; Monte Carlo p-value against uniform random wafer placement, threshold p &le; 0.01), not a calibrated ML probability. Pattern diagnosis does not alter inspection severity.
        </p>

        {/* Assumed wafer geometry or other pattern warnings */}
        {pattern.warnings && pattern.warnings.length > 0 && (
          <div className="space-y-1.5">
            {pattern.warnings.map((w, idx) => (
              <div
                key={idx}
                role="alert"
                className="px-3 py-2 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900"
              >
                <strong>Geometry Warning:</strong> {w}
              </div>
            ))}
          </div>
        )}

        {/* Radial distribution chart: Observed vs Expected Random per radial bin */}
        {radialBins.length > 0 && (
          <div className="pt-2 border-t border-slate-100">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <span className="text-xs font-semibold text-slate-700">
                Radial Distribution (Normalized Radius r = d / R)
              </span>
              <span className="text-[11px] font-mono text-slate-500">
                Expected random count proportional to ring area (r<sub>1</sub>&sup2; &minus; r<sub>0</sub>&sup2;)
              </span>
            </div>
            <div className="h-52 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={radialBins} margin={{ top: 8, right: 16, left: -12, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="bin" tick={{ fontSize: 11, fill: '#475569' }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#475569' }} />
                  <Tooltip
                    contentStyle={{
                      fontSize: '12px',
                      borderRadius: '8px',
                      border: '1px solid #cbd5e1',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="observed" name="Observed Defects" fill="#4f46e5" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="expected_random" name="Expected Random" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
