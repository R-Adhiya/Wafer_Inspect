import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { DashboardPage } from '../pages/DashboardPage';
import { HistoryPage } from '../pages/HistoryPage';
import * as api from '../services/api';

afterEach(() => {
  vi.restoreAllMocks();
});

const SAMPLE_HISTORY_ITEMS: api.HistoryItem[] = [
  {
    id: 11,
    upload_id: 'aaaa1111-1111-1111-1111-111111111111',
    status: 'completed',
    original_filename: 'lotA_wafer01.png',
    image_width: 512,
    image_height: 512,
    has_reference: false,
    defect_count: 8,
    total_defect_area: 1200,
    largest_defect_area: 160,
    severity: 'HIGH',
    pattern_type: 'EDGE_RING',
    pattern_confidence: 0.9995,
    detection_path: 'self_reference',
    detection_label: 'Die-to-die self-reference detection',
    detection_method: 'Classical CV Baseline',
    model_name: 'WaferInspect-CV',
    model_version: '1.0.0',
    processing_time: 0.22,
    created_at: '2026-11-18T12:00:00Z',
  },
];

describe('Phase 8 Frontend: Dashboard, History queries, and Delete confirmation', () => {
  it('Dashboard renders clear empty state when database has zero inspections', async () => {
    vi.spyOn(api, 'getHealth').mockResolvedValue({ status: 'ok', service: 'waferinspect-backend' });
    vi.spyOn(api, 'getDashboardStats').mockResolvedValue({
      total_inspections: 0,
      total_defects: 0,
      high_severity_inspections: 0,
      average_defect_area: 0,
      recent_inspections: [],
      recent_defect_counts: [],
      severity_distribution: [
        { severity: 'NONE', count: 0 },
        { severity: 'LOW', count: 0 },
        { severity: 'MEDIUM', count: 0 },
        { severity: 'HIGH', count: 0 },
      ],
      defect_area_histogram: [
        { bin: '0-50', count: 0 },
        { bin: '51-150', count: 0 },
      ],
      pattern_distribution: [],
    });

    render(
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/No inspections recorded yet/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Start First Inspection/i)).toBeInTheDocument();
  });

  it('HistoryPage sends correct query parameters for search, severity filter, sort, and page', async () => {
    const historySpy = vi.spyOn(api, 'getInspectionHistory').mockResolvedValue({
      items: SAMPLE_HISTORY_ITEMS,
      total: 25,
      page: 1,
      page_size: 10,
      total_pages: 3,
    });

    render(
      <MemoryRouter>
        <HistoryPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(historySpy).toHaveBeenCalledWith(
        expect.objectContaining({ sort: 'created_at', order: 'desc', page: 1, page_size: 10 })
      );
    });

    // Severity filter -> HIGH
    fireEvent.click(screen.getByRole('button', { name: 'HIGH' }));
    await waitFor(() => {
      expect(historySpy).toHaveBeenLastCalledWith(
        expect.objectContaining({ severity: 'HIGH', page: 1 })
      );
    });

    // Sort by defect_count
    fireEvent.click(screen.getByRole('button', { name: /Defects/i }));
    await waitFor(() => {
      expect(historySpy).toHaveBeenLastCalledWith(
        expect.objectContaining({ sort: 'defect_count', order: 'desc', page: 1 })
      );
    });

    // Pagination -> page 2
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    await waitFor(() => {
      expect(historySpy).toHaveBeenLastCalledWith(
        expect.objectContaining({ page: 2, sort: 'defect_count', severity: 'HIGH' })
      );
    });

    // Debounced search -> resets to page 1 with search param
    const searchInput = screen.getByPlaceholderText(/Search by filename/i);
    fireEvent.change(searchInput, { target: { value: 'lotA' } });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 350));
    });
    await waitFor(() => {
      expect(historySpy).toHaveBeenLastCalledWith(
        expect.objectContaining({ search: 'lotA', page: 1 })
      );
    });
  });

  it('HistoryPage delete confirmation dialog calls deleteInspection API only on confirm', async () => {
    vi.spyOn(api, 'getInspectionHistory').mockResolvedValue({
      items: SAMPLE_HISTORY_ITEMS,
      total: 1,
      page: 1,
      page_size: 10,
      total_pages: 1,
    });
    const deleteSpy = vi.spyOn(api, 'deleteInspection').mockResolvedValue({
      deleted: true,
      inspection_id: 11,
      upload_id: SAMPLE_HISTORY_ITEMS[0].upload_id,
    });

    render(
      <MemoryRouter>
        <HistoryPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByTestId('history-row-11')).toBeInTheDocument();
    });

    // Open delete confirmation dialog
    fireEvent.click(screen.getByRole('button', { name: 'Delete inspection #11' }));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(deleteSpy).not.toHaveBeenCalled();

    // Confirm delete -> calls API
    fireEvent.click(screen.getByRole('button', { name: 'Confirm Delete' }));
    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledWith(11);
    });
  });
});
