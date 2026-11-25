import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import {
  NoiseSweepPoint,
  NoiseSweepResponse,
  runNoiseSweep,
} from '../services/api';
import { NoiseFunnelChart } from '../components/NoiseFunnelChart';

const LAYOUT_OPTIONS = [
  { value: 'edge_ring', label: 'Edge Ring (Wafer Bevel Pattern)' },
  { value: 'center_cluster', label: 'Center Cluster' },
  { value: 'donut', label: 'Donut (Annular Distribution)' },
  { value: 'scratch_line', label: 'Scratch Line (Linear Track)' },
  { value: 'local_cluster', label: 'Local Cluster' },
  { value: 'random', label: 'Random Scatter' },
];

const PRESET_NOISE_LEVELS = [
  { label: 'Standard [2, 5, 10, 15, 20]', values: [2.0, 5.0, 10.0, 15.0, 20.0] },
  { label: 'Low Noise [1, 2, 4, 8]', values: [1.0, 2.0, 4.0, 8.0] },
  { label: 'Fine Step [2, 4, 6, 8, 10, 12]', values: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0] },
];

export const NoiseExperimentPage: React.FC = () => {
  const [layout, setLayout] = useState('edge_ring');
  const [path, setPath] = useState<'self_reference' | 'golden'>('self_reference');
  const [noiseLevelsInput, setNoiseLevelsInput] = useState('2, 5, 10, 15, 20');
  const [seedsInput, setSeedsInput] = useState('42, 43, 44, 45, 46');
  const [size, setSize] = useState(512);
  const [defectCount, setDefectCount] = useState(8);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sweepResult, setSweepResult] = useState<NoiseSweepResponse | null>(null);
  const [selectedNoiseLevel, setSelectedNoiseLevel] = useState<number>(10.0);

  const handleRunSweep = async () => {
    setError(null);

    // Parse and validate noise levels
    const levels = noiseLevelsInput
      .split(',')
      .map((s) => parseFloat(s.trim()))
      .filter((n) => !isNaN(n) && n > 0);

    if (levels.length === 0) {
      setError('Please provide at least one valid numeric noise level.');
      return;
    }
    if (levels.length > 8) {
      setError('Maximum 8 noise levels are allowed per sweep.');
      return;
    }

    // Parse and validate seeds
    const seeds = seedsInput
      .split(',')
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n));

    if (seeds.length === 0) {
      setError('Please provide at least one integer seed.');
      return;
    }
    if (seeds.length > 10) {
      setError('Maximum 10 random seeds are allowed per sweep.');
      return;
    }

    if (size > 1024) {
      setError('Wafer dimension cannot exceed 1024.');
      return;
    }

    setIsLoading(true);
    try {
      const res = await runNoiseSweep({
        layout,
        path,
        noise_levels: levels,
        seeds,
        size,
        defect_count: defectCount,
      });
      setSweepResult(res);
      // Select the median level or first level
      if (res.series.length > 0) {
        const mid = Math.floor(res.series.length / 2);
        setSelectedNoiseLevel(res.series[mid].noise_level);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to execute noise sweep experiment.');
    } finally {
      setIsLoading(false);
    }
  };

  // Run initial sweep on mount
  useEffect(() => {
    void handleRunSweep();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Multi-line chart data across noise levels
  const lineChartData = sweepResult
    ? sweepResult.series.map((pt) => ({
        noiseLevel: pt.noise_level,
        sigmaLabel: `σ=${pt.noise_level}`,
        raw: pt.cases[0]?.mean_count ?? 0,
        minArea: pt.cases[1]?.mean_count ?? 0,
        morph: pt.cases[2]?.mean_count ?? 0,
        final: pt.cases[3]?.mean_count ?? 0,
        precision: (pt.cases[3]?.mean_precision ?? 0) * 100,
        recall: (pt.cases[3]?.mean_recall ?? 0) * 100,
      }))
    : [];

  const selectedPoint: NoiseSweepPoint | undefined = sweepResult?.series.find(
    (pt) => Math.abs(pt.noise_level - selectedNoiseLevel) < 0.001
  ) || sweepResult?.series[0];

  const selectedFunnelCases = selectedPoint
    ? selectedPoint.cases.map((c) => ({
        stage_id: c.stage_id,
        stage_name: c.stage_name,
        label: c.label,
        candidate_count: Math.round(c.mean_count),
        tp: Math.round(c.mean_tp),
        fp: Math.round(c.mean_fp),
        fn: Math.round(c.mean_fn),
        precision: c.mean_precision,
        recall: c.mean_recall,
      }))
    : [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
              Noise Filtering Experiment
            </h2>
            <span className="text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 px-2.5 py-0.5 rounded-full">
              Synthetic demonstration data
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Empirical evaluation of hierarchical candidate filtering and precision across sensor noise levels
          </p>
        </div>
      </div>

      {/* Control Card */}
      <section
        aria-label="Experiment Configuration"
        className="p-5 bg-white border border-slate-200 rounded-xl shadow-xs space-y-4"
      >
        <h3 className="text-sm font-bold text-slate-900">Sweep Parameters</h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          <div className="xl:col-span-1">
            <label htmlFor="layout-select" className="block text-xs font-semibold text-slate-700 mb-1">
              Defect Layout
            </label>
            <select
              id="layout-select"
              value={layout}
              onChange={(e) => setLayout(e.target.value)}
              disabled={isLoading}
              className="w-full text-xs rounded-lg border border-slate-300 px-2.5 py-2 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {LAYOUT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <div className="xl:col-span-1">
            <label htmlFor="path-select" className="block text-xs font-semibold text-slate-700 mb-1">
              Detection Mode
            </label>
            <select
              id="path-select"
              value={path}
              onChange={(e) => setPath(e.target.value as any)}
              disabled={isLoading}
              className="w-full text-xs rounded-lg border border-slate-300 px-2.5 py-2 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="self_reference">Self-Reference (Die-to-Die)</option>
              <option value="golden">Golden Reference (Pair)</option>
            </select>
          </div>

          <div className="xl:col-span-1">
            <label htmlFor="noise-input" className="block text-xs font-semibold text-slate-700 mb-1">
              Noise σ Levels (≤ 8)
            </label>
            <input
              id="noise-input"
              type="text"
              value={noiseLevelsInput}
              onChange={(e) => setNoiseLevelsInput(e.target.value)}
              disabled={isLoading}
              placeholder="e.g. 2, 5, 10, 15, 20"
              className="w-full text-xs rounded-lg border border-slate-300 px-2.5 py-2 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <div className="xl:col-span-1">
            <label htmlFor="seeds-input" className="block text-xs font-semibold text-slate-700 mb-1">
              Random Seeds (≤ 10)
            </label>
            <input
              id="seeds-input"
              type="text"
              value={seedsInput}
              onChange={(e) => setSeedsInput(e.target.value)}
              disabled={isLoading}
              placeholder="e.g. 42, 43, 44, 45, 46"
              className="w-full text-xs rounded-lg border border-slate-300 px-2.5 py-2 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <div className="xl:col-span-1">
            <label htmlFor="size-select" className="block text-xs font-semibold text-slate-700 mb-1">
              Wafer Dimension (≤ 1024)
            </label>
            <select
              id="size-select"
              value={size}
              onChange={(e) => setSize(parseInt(e.target.value, 10))}
              disabled={isLoading}
              className="w-full text-xs rounded-lg border border-slate-300 px-2.5 py-2 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value={256}>256 × 256</option>
              <option value={512}>512 × 512</option>
              <option value={1024}>1024 × 1024</option>
            </select>
          </div>

          <div className="xl:col-span-1">
            <label htmlFor="defect-count-input" className="block text-xs font-semibold text-slate-700 mb-1">
              Defects Injected
            </label>
            <input
              id="defect-count-input"
              type="number"
              min={1}
              max={30}
              value={defectCount}
              onChange={(e) => setDefectCount(parseInt(e.target.value, 10) || 8)}
              disabled={isLoading}
              className="w-full text-xs rounded-lg border border-slate-300 px-2.5 py-2 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
        </div>

        {/* Quick Presets & Run Button */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-500 font-medium">Noise Presets:</span>
            {PRESET_NOISE_LEVELS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => setNoiseLevelsInput(preset.values.join(', '))}
                disabled={isLoading}
                className="px-2.5 py-1 text-xs rounded-md border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-700 transition-colors disabled:opacity-50"
              >
                {preset.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleRunSweep}
              disabled={isLoading}
              className="px-5 py-2 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60 flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Running Sweep...
                </>
              ) : (
                'Run Experiment Sweep'
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-rose-50 border border-rose-200 text-rose-800 rounded-lg text-xs font-medium">
            {error}
          </div>
        )}
      </section>

      {/* Dynamic Plain-Language Explanation */}
      {sweepResult && (
        <section
          aria-label="Experiment Summary"
          className="p-5 bg-white border border-slate-200 rounded-xl shadow-xs space-y-2 border-l-4 border-l-indigo-600"
        >
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wide">
            Automated Empirical Findings
          </h3>
          <p className="text-xs text-slate-700 leading-relaxed font-normal">
            {sweepResult.summary}
          </p>
        </section>
      )}

      {/* Visualizations Grid */}
      {sweepResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Multi-Line Chart: Candidates across Noise Levels */}
          <div
            className="p-5 bg-white border border-slate-200 rounded-xl shadow-xs space-y-3"
            role="region"
            aria-label="Candidate Counts Across Noise Levels"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-900">
                Candidate Growth vs. Noise (σ)
              </h3>
              <span className="text-xs text-slate-500 font-mono">
                Target Defect Count: {sweepResult.true_defect_count}
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Raw threshold candidates surge with noise, while Stage 4 stays tightly controlled
            </p>

            <div style={{ width: '100%', height: 260 }} aria-hidden="true">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lineChartData} margin={{ top: 10, right: 16, left: -16, bottom: 0 }}>
                  <XAxis dataKey="sigmaLabel" stroke="#64748b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload || !payload.length) return null;
                      const d = payload[0].payload;
                      return (
                        <div className="bg-slate-900 text-white text-xs rounded-lg p-3 shadow-lg border border-slate-700 space-y-1">
                          <div className="font-semibold text-slate-200">{d.sigmaLabel}</div>
                          <div className="text-slate-300">Stage 1 (Raw): <strong>{d.raw.toFixed(1)}</strong></div>
                          <div className="text-blue-300">Stage 2 (Min-Area): <strong>{d.minArea.toFixed(1)}</strong></div>
                          <div className="text-indigo-300">Stage 3 (Morph): <strong>{d.morph.toFixed(1)}</strong></div>
                          <div className="text-emerald-400 font-bold">Stage 4 (Final): <strong>{d.final.toFixed(1)}</strong></div>
                          <div className="text-slate-400 text-[11px] pt-1 border-t border-slate-700">
                            Precision: {d.precision.toFixed(1)}% | Recall: {d.recall.toFixed(1)}%
                          </div>
                        </div>
                      );
                    }}
                  />
                  <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '11px' }} />
                  <ReferenceLine
                    y={sweepResult.true_defect_count}
                    stroke="#059669"
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                    label={{
                      value: `GT: ${sweepResult.true_defect_count}`,
                      position: 'insideTopRight',
                      fill: '#059669',
                      fontSize: 10,
                      fontWeight: 600,
                    }}
                  />
                  <Line type="monotone" dataKey="raw" name="1. Raw Detection" stroke="#94a3b8" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="minArea" name="2. Min-Area Only" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="morph" name="3. Morph + Area" stroke="#818cf8" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="final" name="4. Final Filtered" stroke="#10b981" strokeWidth={2.5} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Level Funnel Bar Chart */}
          <div
            className="p-5 bg-white border border-slate-200 rounded-xl shadow-xs space-y-3"
            role="region"
            aria-label="Single Level Funnel Inspection"
          >
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-bold text-slate-900">
                Funnel at Selected Noise Level
              </h3>
              <div className="flex items-center gap-1.5">
                <label htmlFor="select-level" className="text-xs text-slate-500 font-medium">
                  Select σ:
                </label>
                <select
                  id="select-level"
                  value={selectedNoiseLevel}
                  onChange={(e) => setSelectedNoiseLevel(parseFloat(e.target.value))}
                  className="text-xs rounded-md border border-slate-300 px-2 py-1 font-mono text-slate-800 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {sweepResult.series.map((pt) => (
                    <option key={pt.noise_level} value={pt.noise_level}>
                      σ = {pt.noise_level}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {selectedPoint?.summary && (
              <p className="text-xs text-indigo-900 bg-indigo-50 p-2.5 rounded-lg border border-indigo-100 font-medium">
                {selectedPoint.summary}
              </p>
            )}

            <NoiseFunnelChart
              cases={selectedFunnelCases}
              groundTruthCount={sweepResult.true_defect_count}
              height={220}
              title={`Stage Funnel (Noise σ=${selectedNoiseLevel})`}
              showTable={false}
            />
          </div>
        </div>
      )}

      {/* Full Multi-Level Metrics Table */}
      {sweepResult && (
        <section
          aria-label="Multi-Level Sweep Metrics Table"
          className="p-5 bg-white border border-slate-200 rounded-xl shadow-xs space-y-4"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900">
              Complete Precision &amp; Recall Performance Table
            </h3>
            <span className="text-xs text-slate-500">
              Evaluated across {sweepResult.seeds.length} seeds per cell
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-xs text-left border-collapse border border-slate-200 rounded-lg">
              <caption className="sr-only">Comprehensive Noise Sweep Metrics</caption>
              <thead className="bg-slate-50 text-slate-700 border-b border-slate-200">
                <tr>
                  <th scope="col" className="px-3 py-2 font-semibold text-center">Noise σ</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">Raw (Stage 1)</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">Min-Area (Stage 2)</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">Morph (Stage 3)</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">Final (Stage 4)</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">TP</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">FP</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">FN</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">Precision</th>
                  <th scope="col" className="px-3 py-2 font-semibold text-right">Recall</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {sweepResult.series.map((pt) => {
                  const c1 = pt.cases[0];
                  const c2 = pt.cases[1];
                  const c3 = pt.cases[2];
                  const c4 = pt.cases[3];
                  const isSelected = Math.abs(pt.noise_level - selectedNoiseLevel) < 0.001;

                  return (
                    <tr
                      key={pt.noise_level}
                      onClick={() => setSelectedNoiseLevel(pt.noise_level)}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? 'bg-indigo-50/70 font-semibold' : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="px-3 py-2 text-center font-mono font-bold text-slate-900">
                        {pt.noise_level.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-slate-600">
                        {c1.mean_count.toFixed(1)} ± {c1.std_count.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-slate-600">
                        {c2.mean_count.toFixed(1)} ± {c2.std_count.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-slate-600">
                        {c3.mean_count.toFixed(1)} ± {c3.std_count.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-emerald-800 font-bold">
                        {c4.mean_count.toFixed(1)} ± {c4.std_count.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-emerald-700">
                        {c4.mean_tp.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-rose-600">
                        {c4.mean_fp.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-amber-600">
                        {c4.mean_fn.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono font-bold text-slate-900">
                        {(c4.mean_precision * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-2 text-right font-mono font-bold text-slate-900">
                        {(c4.mean_recall * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};
