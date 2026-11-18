import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  DashboardStatsResponse,
  getDashboardStats,
  InspectionSeverity,
} from '../services/api';
import { BackendStatus } from '../components/BackendStatus';
import { EmptyState } from '../components/EmptyState';
import { ErrorMessage } from '../components/ErrorMessage';
import { MetricCard } from '../components/MetricCard';
import { SeverityBadge } from '../components/SeverityBadge';

const SEVERITY_COLORS: Record<string, string> = {
  NONE: '#10b981',
  LOW: '#2563eb',
  MEDIUM: '#ea580c',
  HIGH: '#dc2626',
};

const formatDateTime = (iso: string) => {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
};

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<DashboardStatsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await getDashboardStats();
        if (!cancelled) setStats(data);
      } catch (err: unknown) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unable to load dashboard statistics.');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const hasInspections = (stats?.total_inspections ?? 0) > 0;
  const hasDefects = (stats?.total_defects ?? 0) > 0;
  const hasSeverityData = (stats?.severity_distribution ?? []).some((s) => s.count > 0);
  const hasPatternData = (stats?.pattern_distribution ?? []).some((p) => p.count > 0);

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      <section aria-label="System Health">
        <BackendStatus />
      </section>

      {/* Top Header & Primary CTA */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            Inspection Dashboard
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Live SQL-aggregated defect KPIs, spatial pattern trends, and recent wafer inspections.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <Link
            to="/history"
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-white hover:bg-slate-100 text-slate-800 border border-slate-300 shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            View History
          </Link>
          <Link
            to="/inspection/new"
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Start Inspection
          </Link>
        </div>
      </div>

      {error && <ErrorMessage title="Dashboard Error" message={error} />}

      {isLoading ? (
        <div role="status" className="p-10 bg-white border border-slate-200 rounded-xl text-center text-sm text-slate-600">
          Loading dashboard statistics...
        </div>
      ) : !stats ? null : (
        <>
          {/* 4 KPI Metric Cards */}
          <section aria-label="Key Performance Metrics" className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              label="Total Inspections"
              value={stats.total_inspections}
              subtext="Persisted wafer runs"
            />
            <MetricCard
              label="Defects Detected"
              value={stats.total_defects}
              subtext="Total accepted regions"
            />
            <MetricCard
              label="High-Severity Inspections"
              value={stats.high_severity_inspections}
              subtext="Prototype HIGH severity"
            />
            <MetricCard
              label="Average Defect Area"
              value={`${stats.average_defect_area} px`}
              subtext="Mean refined area (uncalibrated)"
            />
          </section>

          {!hasInspections ? (
            <section className="bg-white border border-slate-200 rounded-xl shadow-xs">
              <EmptyState
                title="No inspections recorded yet"
                description="Run your first wafer inspection or try a synthetic demonstration wafer to populate the dashboard charts and history."
                severity="NONE"
              />
              <div className="pb-6 flex justify-center">
                <Link
                  to="/inspection/new"
                  className="px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  Start First Inspection
                </Link>
              </div>
            </section>
          ) : (
            <>
              {/* 4 Real Data Charts (2x2 Grid on Desktop, Stacked on Tablet/Mobile) */}
              <section aria-label="Analytics Charts" className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Chart 1: Defect Count Across Recent Inspections */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wide text-slate-900">
                      Defect Count Across Recent Inspections
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      Last {stats.recent_defect_counts.length} inspection runs (oldest to newest)
                    </p>
                  </div>
                  {stats.recent_defect_counts.length === 0 ? (
                    <EmptyState title="Not enough data" description="Run inspections to view defect count trends." />
                  ) : (
                    <div className="h-56 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={stats.recent_defect_counts.map((pt) => ({
                            ...pt,
                            label: `#${pt.inspection_id}`,
                          }))}
                          margin={{ top: 8, right: 12, left: -12, bottom: 4 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#475569' }} />
                          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#475569' }} />
                          <Tooltip
                            formatter={(val: any) => [`${val} defects`, 'Defects']}
                            labelFormatter={(lbl, payload) =>
                              payload?.[0]?.payload?.filename
                                ? `${lbl} — ${payload[0].payload.filename}`
                                : String(lbl)
                            }
                          />
                          <Bar dataKey="defect_count" name="Defects" fill="#4f46e5" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {/* Chart 2: Severity Distribution */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wide text-slate-900">
                      Severity Distribution
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      Inspection count by prototype severity tier
                    </p>
                  </div>
                  {!hasSeverityData ? (
                    <EmptyState title="Not enough data" description="Complete inspections to populate severity counts." />
                  ) : (
                    <div className="h-56 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={stats.severity_distribution}
                          margin={{ top: 8, right: 12, left: -12, bottom: 4 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="severity" tick={{ fontSize: 11, fill: '#475569' }} />
                          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#475569' }} />
                          <Tooltip />
                          <Bar dataKey="count" name="Inspections" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                            {stats.severity_distribution.map((entry) => (
                              <Cell
                                key={entry.severity}
                                fill={SEVERITY_COLORS[entry.severity] ?? '#64748b'}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {/* Chart 3: Defect Area Distribution (Histogram) */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wide text-slate-900">
                      Defect Area Distribution (Histogram)
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      Refined defect count per area bucket (original image pixels)
                    </p>
                  </div>
                  {!hasDefects ? (
                    <EmptyState title="No defects recorded" description="Defect area histogram appears once defects are detected." />
                  ) : (
                    <div className="h-56 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={stats.defect_area_histogram}
                          margin={{ top: 8, right: 12, left: -12, bottom: 4 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="bin" tick={{ fontSize: 11, fill: '#475569' }} />
                          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#475569' }} />
                          <Tooltip />
                          <Bar dataKey="count" name="Defects" fill="#0ea5e9" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {/* Chart 4: Spatial Pattern Distribution */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wide text-slate-900">
                      Spatial Pattern Distribution
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      Monte Carlo diagnosed wafer pattern types across inspections
                    </p>
                  </div>
                  {!hasPatternData ? (
                    <EmptyState title="Not enough data" description="Pattern distribution appears once inspections complete." />
                  ) : (
                    <div className="h-56 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={stats.pattern_distribution}
                          margin={{ top: 8, right: 16, left: 20, bottom: 28 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis
                            dataKey="pattern_type"
                            tick={{ fontSize: 10, fill: '#475569' }}
                            interval={0}
                            angle={-18}
                            textAnchor="end"
                          />
                          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#475569' }} />
                          <Tooltip />
                          <Bar dataKey="count" name="Inspections" fill="#7c3aed" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </section>

              {/* Recent Inspections Table */}
              <section
                aria-labelledby="recent-inspections-heading"
                className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden"
              >
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                  <h3
                    id="recent-inspections-heading"
                    className="text-xs font-bold uppercase tracking-wide text-slate-900"
                  >
                    Recent Inspections
                  </h3>
                  <Link
                    to="/history"
                    className="text-xs font-semibold text-indigo-600 hover:text-indigo-800"
                  >
                    View all &rarr;
                  </Link>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase text-[10px] font-bold tracking-wider">
                        <th className="py-2.5 px-4">ID</th>
                        <th className="py-2.5 px-4">Filename</th>
                        <th className="py-2.5 px-3">Date &amp; Time</th>
                        <th className="py-2.5 px-3 text-right">Defects</th>
                        <th className="py-2.5 px-3">Severity</th>
                        <th className="py-2.5 px-3">Pattern</th>
                        <th className="py-2.5 px-3 text-right">Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {stats.recent_inspections.map((item) => (
                        <tr
                          key={item.id}
                          tabIndex={0}
                          onClick={() => navigate(`/inspection/${item.id}`)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              navigate(`/inspection/${item.id}`);
                            }
                          }}
                          className="hover:bg-slate-50 cursor-pointer focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
                        >
                          <td className="py-2.5 px-4 font-mono font-semibold text-indigo-700">
                            #{item.id}
                          </td>
                          <td className="py-2.5 px-4 font-mono text-slate-900">
                            {item.original_filename}
                          </td>
                          <td className="py-2.5 px-3 text-slate-600">
                            {formatDateTime(item.created_at)}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-right text-slate-800">
                            {item.defect_count}
                          </td>
                          <td className="py-2.5 px-3">
                            <SeverityBadge severity={(item.severity ?? 'NONE') as InspectionSeverity} />
                          </td>
                          <td className="py-2.5 px-3 font-mono text-[11px] text-slate-700">
                            {item.pattern_type ?? '—'}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-right text-slate-600">
                            {item.processing_time != null ? `${item.processing_time.toFixed(3)} s` : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
};
