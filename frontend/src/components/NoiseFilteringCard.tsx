import React, { useEffect, useState } from 'react';
import {
  getInspectionNoiseExperiment,
  InspectionNoiseExperimentResponse,
} from '../services/api';
import { NoiseFunnelChart } from './NoiseFunnelChart';

interface NoiseFilteringCardProps {
  inspectionId: number;
}

export const NoiseFilteringCard: React.FC<NoiseFilteringCardProps> = ({ inspectionId }) => {
  const [data, setData] = useState<InspectionNoiseExperimentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);
    setError(null);

    getInspectionNoiseExperiment(inspectionId)
      .then((res) => {
        if (isMounted) setData(res);
      })
      .catch((err: any) => {
        if (isMounted) setError(err?.message || 'Failed to load noise experiment data.');
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [inspectionId]);

  if (isLoading) {
    return (
      <div
        className="p-6 bg-white border border-slate-200 rounded-xl shadow-xs animate-pulse text-center"
        role="status"
        aria-label="Loading noise filtering evaluation"
      >
        <div className="h-4 bg-slate-200 rounded w-1/4 mb-4" />
        <div className="h-40 bg-slate-100 rounded w-full" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-800">
        <span className="font-semibold">Noise Filtering Note:</span> {error || 'No noise evaluation available.'}
      </div>
    );
  }

  const rawCount = data.cases[0]?.candidate_count ?? 0;
  const finalCount = data.cases[3]?.candidate_count ?? 0;
  const removedTotal = Math.max(0, rawCount - finalCount);
  const pctRemoved = rawCount > 0 ? ((removedTotal / rawCount) * 100).toFixed(1) : '0.0';

  return (
    <section
      aria-label="Noise Filtering Evaluation"
      className="p-6 bg-white border border-slate-200 rounded-xl shadow-xs space-y-5"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            Noise Filtering & Candidate Progression
            {data.disclaimer && (
              <span className="text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full">
                {data.disclaimer}
              </span>
            )}
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Four-stage candidate progression from raw thresholded mask to final verified defects
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs font-medium">
          <span className="bg-slate-100 text-slate-700 px-2.5 py-1 rounded-md">
            Raw Candidates: <strong className="font-mono text-slate-900">{rawCount}</strong>
          </span>
          <span className="bg-emerald-50 text-emerald-800 border border-emerald-200 px-2.5 py-1 rounded-md">
            Filtered: <strong className="font-mono text-emerald-900">{removedTotal} ({pctRemoved}%)</strong>
          </span>
        </div>
      </div>

      {/* Dynamic plain-language explanation */}
      <div className="bg-slate-50 border-l-4 border-indigo-500 p-3.5 rounded-r-lg text-xs text-slate-700 leading-relaxed">
        {data.summary}
      </div>

      {/* Funnel Bar Chart */}
      <NoiseFunnelChart
        cases={data.cases}
        groundTruthCount={data.ground_truth_count}
        height={220}
        title="Candidate Count Progression by Stage"
        showTable={false}
      />

      {/* Structured metrics table */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-xs text-left border-collapse border border-slate-200 rounded-lg">
          <caption className="sr-only">Noise filtering stage metrics</caption>
          <thead className="bg-slate-50 text-slate-700 border-b border-slate-200">
            <tr>
              <th scope="col" className="px-3.5 py-2 font-semibold">Stage</th>
              <th scope="col" className="px-3.5 py-2 font-semibold text-right">Candidates</th>
              <th scope="col" className="px-3.5 py-2 font-semibold text-right">Filtered vs Raw</th>
              {data.has_ground_truth && (
                <>
                  <th scope="col" className="px-3.5 py-2 font-semibold text-right">TP</th>
                  <th scope="col" className="px-3.5 py-2 font-semibold text-right">FP</th>
                  <th scope="col" className="px-3.5 py-2 font-semibold text-right">Precision</th>
                  <th scope="col" className="px-3.5 py-2 font-semibold text-right">Recall</th>
                </>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {data.cases.map((c) => {
              const diff = rawCount - c.candidate_count;
              const diffPct = rawCount > 0 ? ((diff / rawCount) * 100).toFixed(1) : '0.0';
              return (
                <tr key={c.stage_id} className="hover:bg-slate-50/75">
                  <td className="px-3.5 py-2 font-medium text-slate-900">{c.label}</td>
                  <td className="px-3.5 py-2 text-right font-mono font-semibold text-slate-800">
                    {c.candidate_count}
                  </td>
                  <td className="px-3.5 py-2 text-right font-mono text-slate-600">
                    {c.stage_id === 1 ? '—' : `-${diff} (${diffPct}%)`}
                  </td>
                  {data.has_ground_truth && (
                    <>
                      <td className="px-3.5 py-2 text-right font-mono text-emerald-700 font-semibold">
                        {c.tp ?? '—'}
                      </td>
                      <td className="px-3.5 py-2 text-right font-mono text-rose-600">
                        {c.fp ?? '—'}
                      </td>
                      <td className="px-3.5 py-2 text-right font-mono font-semibold text-slate-800">
                        {c.precision != null ? `${(c.precision * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-3.5 py-2 text-right font-mono font-semibold text-slate-800">
                        {c.recall != null ? `${(c.recall * 100).toFixed(1)}%` : '—'}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
};
