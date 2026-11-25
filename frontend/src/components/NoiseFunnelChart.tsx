import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';
import { NoiseExperimentCase } from '../services/api';

interface NoiseFunnelChartProps {
  cases: NoiseExperimentCase[];
  groundTruthCount?: number | null;
  height?: number;
  title?: string;
  showTable?: boolean;
}

const STAGE_COLORS = [
  '#94a3b8', // 1. Raw Detection (slate)
  '#60a5fa', // 2. Min-Area Filter Only (blue)
  '#818cf8', // 3. Morphology + Area (indigo)
  '#10b981', // 4. Final Filtered Defects (emerald)
];

export const NoiseFunnelChart: React.FC<NoiseFunnelChartProps> = ({
  cases,
  groundTruthCount,
  height = 240,
  title = 'Candidate Count by Pipeline Stage',
  showTable = false,
}) => {
  const chartData = cases.map((c, idx) => ({
    stageId: c.stage_id,
    shortName: `Stage ${c.stage_id}`,
    label: c.label,
    count: c.candidate_count,
    tp: c.tp,
    fp: c.fp,
    fn: c.fn,
    precision: c.precision != null ? `${(c.precision * 100).toFixed(1)}%` : null,
    recall: c.recall != null ? `${(c.recall * 100).toFixed(1)}%` : null,
    color: STAGE_COLORS[idx % STAGE_COLORS.length],
  }));

  const maxCount = Math.max(
    ...chartData.map((d) => d.count),
    groundTruthCount ?? 0,
    1
  );

  return (
    <div
      className="w-full flex flex-col space-y-3"
      role="region"
      aria-label={title}
    >
      <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
        <span>{title}</span>
        {groundTruthCount != null && (
          <span className="text-emerald-700 font-semibold flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-0.5 bg-emerald-600 border border-emerald-600 border-dashed" />
            Ground Truth Target: {groundTruthCount} defects
          </span>
        )}
      </div>

      <div style={{ width: '100%', height }} aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 12, right: 16, left: -16, bottom: 20 }}
          >
            <XAxis
              dataKey="shortName"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <YAxis
              stroke="#64748b"
              fontSize={11}
              domain={[0, Math.ceil(maxCount * 1.15)]}
              allowDecimals={false}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-slate-900 text-white text-xs rounded-lg px-3 py-2 shadow-lg border border-slate-700">
                    <div className="font-semibold text-slate-200 mb-1">{d.label}</div>
                    <div className="text-emerald-400 font-bold">
                      Candidates: {d.count}
                    </div>
                    {d.precision && (
                      <div className="text-slate-300 mt-1 flex flex-col gap-0.5 text-[11px]">
                        <div>Precision: {d.precision}</div>
                        <div>Recall: {d.recall}</div>
                        <div>TP: {d.tp} | FP: {d.fp}</div>
                      </div>
                    )}
                  </div>
                );
              }}
            />
            {groundTruthCount != null && (
              <ReferenceLine
                y={groundTruthCount}
                stroke="#059669"
                strokeDasharray="4 4"
                strokeWidth={2}
                label={{
                  value: `GT: ${groundTruthCount}`,
                  position: 'top',
                  fill: '#059669',
                  fontSize: 10,
                  fontWeight: 600,
                }}
              />
            )}
            <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={56}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Accessible data table for screen readers or when showTable is true */}
      <div className={showTable ? 'block overflow-x-auto mt-2' : 'sr-only'}>
        <table className="min-w-full text-xs text-left border-collapse border border-slate-200">
          <caption className="sr-only">{title}</caption>
          <thead className="bg-slate-50 text-slate-700 border-b border-slate-200">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-semibold">Stage</th>
              <th scope="col" className="px-3 py-1.5 font-semibold text-right">Candidates</th>
              {groundTruthCount != null && (
                <>
                  <th scope="col" className="px-3 py-1.5 font-semibold text-right">TP</th>
                  <th scope="col" className="px-3 py-1.5 font-semibold text-right">FP</th>
                  <th scope="col" className="px-3 py-1.5 font-semibold text-right">Precision</th>
                  <th scope="col" className="px-3 py-1.5 font-semibold text-right">Recall</th>
                </>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {chartData.map((d) => (
              <tr key={d.stageId} className="hover:bg-slate-50">
                <td className="px-3 py-1.5 text-slate-900 font-medium">{d.label}</td>
                <td className="px-3 py-1.5 text-slate-700 text-right font-mono">{d.count}</td>
                {groundTruthCount != null && (
                  <>
                    <td className="px-3 py-1.5 text-slate-600 text-right font-mono">{d.tp ?? '-'}</td>
                    <td className="px-3 py-1.5 text-slate-600 text-right font-mono">{d.fp ?? '-'}</td>
                    <td className="px-3 py-1.5 text-slate-600 text-right font-mono">{d.precision ?? '-'}</td>
                    <td className="px-3 py-1.5 text-slate-600 text-right font-mono">{d.recall ?? '-'}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
