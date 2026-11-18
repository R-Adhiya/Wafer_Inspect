import React, { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  deleteInspection,
  getInspectionHistory,
  HistoryItem,
  HistoryQueryParams,
  InspectionSeverity,
} from '../services/api';
import { EmptyState } from '../components/EmptyState';
import { ErrorMessage } from '../components/ErrorMessage';
import { SeverityBadge } from '../components/SeverityBadge';

type SortKey = NonNullable<HistoryQueryParams['sort']>;
type SortOrder = NonNullable<HistoryQueryParams['order']>;

const formatDateTime = (iso: string) => {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
};

export const HistoryPage: React.FC = () => {
  const navigate = useNavigate();

  const [searchInput, setSearchInput] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [severityFilter, setSeverityFilter] = useState<'ALL' | InspectionSeverity>('ALL');
  const [sortKey, setSortKey] = useState<SortKey>('created_at');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [page, setPage] = useState(1);
  const pageSize = 10;

  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Confirmation modal state
  const [pendingDelete, setPendingDelete] = useState<HistoryItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Debounce filename search input (300 ms)
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearch(searchInput);
      setPage(1);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  const fetchHistory = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await getInspectionHistory({
        search: debouncedSearch || undefined,
        severity: severityFilter === 'ALL' ? undefined : severityFilter,
        sort: sortKey,
        order: sortOrder,
        page,
        page_size: pageSize,
      });
      setItems(res.items);
      setTotal(res.total);
      setTotalPages(res.total_pages);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unable to load inspection history.');
    } finally {
      setIsLoading(false);
    }
  }, [debouncedSearch, severityFilter, sortKey, sortOrder, page]);

  useEffect(() => {
    void fetchHistory();
  }, [fetchHistory]);

  const handleSort = (key: SortKey) => {
    setPage(1);
    if (sortKey === key) {
      setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortOrder('desc');
    }
  };

  const handleConfirmDelete = async () => {
    if (!pendingDelete) return;
    setIsDeleting(true);
    setError(null);
    try {
      await deleteInspection(pendingDelete.id);
      setPendingDelete(null);
      await fetchHistory();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete inspection.');
    } finally {
      setIsDeleting(false);
    }
  };

  const renderSortIcon = (key: SortKey) => {
    if (sortKey !== key) return <span className="opacity-30 ml-1">↕</span>;
    return <span className="ml-1 text-indigo-700">{sortOrder === 'asc' ? '↑' : '↓'}</span>;
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            Inspection History
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Search, filter, sort, open, or delete persisted wafer inspection records ({total} total).
          </p>
        </div>

        <Link
          to="/inspection/new"
          className="self-start sm:self-auto px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          Start New Inspection
        </Link>
      </div>

      {error && <ErrorMessage title="History Error" message={error} onDismiss={() => setError(null)} />}

      {/* Search & Severity Filter Controls */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex-1 max-w-md">
          <label htmlFor="history-search" className="sr-only">
            Search inspections by filename
          </label>
          <input
            id="history-search"
            type="search"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Search by filename..."
            className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>

        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter history by severity">
          {(['ALL', 'HIGH', 'MEDIUM', 'LOW', 'NONE'] as const).map((lvl) => (
            <button
              key={lvl}
              type="button"
              aria-pressed={severityFilter === lvl}
              onClick={() => {
                setSeverityFilter(lvl);
                setPage(1);
              }}
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
      </div>

      {/* History Table */}
      <section className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden">
        {isLoading ? (
          <div role="status" className="p-10 text-center text-xs text-slate-600">
            Loading inspection history...
          </div>
        ) : items.length === 0 ? (
          <EmptyState
            title="No matching inspections found"
            description="Adjust your search or severity filter, or run a new wafer inspection."
            severity="NONE"
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase text-[10px] font-bold tracking-wider">
                  <th className="py-2.5 px-3">ID</th>
                  <th className="py-2.5 px-3">Filename</th>
                  <th className="py-2.5 px-3">
                    <button
                      type="button"
                      onClick={() => handleSort('created_at')}
                      className="inline-flex items-center uppercase hover:text-slate-900"
                    >
                      Date &amp; Time {renderSortIcon('created_at')}
                    </button>
                  </th>
                  <th className="py-2.5 px-3 text-right">
                    <button
                      type="button"
                      onClick={() => handleSort('defect_count')}
                      className="inline-flex items-center uppercase hover:text-slate-900"
                    >
                      Defects {renderSortIcon('defect_count')}
                    </button>
                  </th>
                  <th className="py-2.5 px-3">
                    <button
                      type="button"
                      onClick={() => handleSort('severity')}
                      className="inline-flex items-center uppercase hover:text-slate-900"
                    >
                      Severity {renderSortIcon('severity')}
                    </button>
                  </th>
                  <th className="py-2.5 px-3">Pattern</th>
                  <th className="py-2.5 px-3">Detection</th>
                  <th className="py-2.5 px-3 text-right">
                    <button
                      type="button"
                      onClick={() => handleSort('processing_time')}
                      className="inline-flex items-center uppercase hover:text-slate-900"
                    >
                      Time {renderSortIcon('processing_time')}
                    </button>
                  </th>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {items.map((item) => (
                  <tr
                    key={item.id}
                    tabIndex={0}
                    data-testid={`history-row-${item.id}`}
                    onClick={() => navigate(`/inspection/${item.id}`)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        navigate(`/inspection/${item.id}`);
                      }
                    }}
                    className="hover:bg-slate-50 cursor-pointer focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
                  >
                    <td className="py-2.5 px-3 font-mono font-semibold text-indigo-700">
                      #{item.id}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-900 max-w-[200px] truncate" title={item.original_filename}>
                      {item.original_filename}
                    </td>
                    <td className="py-2.5 px-3 text-slate-600 whitespace-nowrap">
                      {formatDateTime(item.created_at)}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-right text-slate-800">
                      {item.defect_count}
                    </td>
                    <td className="py-2.5 px-3">
                      <SeverityBadge severity={(item.severity ?? 'NONE') as InspectionSeverity} />
                    </td>
                    <td className="py-2.5 px-3">
                      {item.pattern_type ? (
                        <span className="inline-flex px-2 py-0.5 rounded bg-indigo-50 text-indigo-800 border border-indigo-200 font-mono text-[10px] font-semibold">
                          {item.pattern_type}
                        </span>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-slate-700 font-mono text-[11px]">
                      {item.detection_path ?? '—'}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-right text-slate-700 whitespace-nowrap">
                      {item.processing_time != null ? `${item.processing_time.toFixed(3)} s` : '—'}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-[11px] text-slate-600 whitespace-nowrap">
                      {item.model_version ? `v${item.model_version}` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <button
                        type="button"
                        aria-label={`Delete inspection #${item.id}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          setPendingDelete(item);
                        }}
                        className="px-2 py-1 text-[11px] font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded focus:outline-none focus:ring-2 focus:ring-rose-500"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Bar */}
        {total > 0 && (
          <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
            <span>
              Page <strong>{page}</strong> of <strong>{totalPages}</strong> ({total} inspections)
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="px-3 py-1 rounded border border-slate-300 bg-white hover:bg-slate-100 disabled:opacity-40 font-semibold text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                Previous
              </button>
              <button
                type="button"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                className="px-3 py-1 rounded border border-slate-300 bg-white hover:bg-slate-100 disabled:opacity-40 font-semibold text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Delete Confirmation Dialog */}
      {pendingDelete && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="delete-dialog-title"
          className="fixed inset-0 z-50 bg-slate-900/50 flex items-center justify-center p-4"
        >
          <div className="bg-white border border-slate-200 rounded-xl max-w-md w-full p-6 shadow-xl space-y-4">
            <h3 id="delete-dialog-title" className="text-base font-bold text-slate-900">
              Delete Inspection #{pendingDelete.id}?
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              This will permanently remove <strong>{pendingDelete.original_filename}</strong>, all{' '}
              <strong>{pendingDelete.defect_count}</strong> defect records, and stored image overlays from disk.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => setPendingDelete(null)}
                className="px-3.5 py-2 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-400"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => void handleConfirmDelete()}
                className="px-3.5 py-2 rounded-lg text-xs font-semibold bg-rose-600 hover:bg-rose-700 text-white focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                {isDeleting ? 'Deleting...' : 'Confirm Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
