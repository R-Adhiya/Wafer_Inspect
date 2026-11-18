import { describe, expect, it, vi, afterEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { InspectionSelectionProvider } from '../context/InspectionSelectionContext';
import { ComparisonViewer } from './ComparisonViewer';
import { DefectTable } from './DefectTable';
import { ProcessingStatus } from './ProcessingStatus';
import { SeverityBadge } from './SeverityBadge';
import {
  AcceptedRegion,
  getInspectionResult,
  InspectionStatusResponse,
  uploadInspection,
} from '../services/api';

const SAMPLE_REGIONS: AcceptedRegion[] = [
  {
    id: 1,
    bbox: [150, 160, 20, 20],
    width_px: 20,
    height_px: 20,
    area_px: 197,
    detected_area_px: 156,
    centroid: [160, 170],
    perimeter_px: 52.3,
    equivalent_diameter_px: 15.8,
    length_px: 16.2,
    thickness_px: 16.2,
    angle_deg: 0,
    elongation: 1.0,
    polarity: 'bright',
    mean_contrast: 60.2,
    peak_contrast: 72.0,
    contrast_to_noise: 8.9,
    touches_edge: false,
    detected_by: 'self_reference',
    why_flagged: 'Bright region, 197 px, contrast 60.2 gray levels = 8.9x the noise level.',
    severity: 'MEDIUM',
    contour: [
      [150, 160],
      [170, 160],
      [170, 180],
      [150, 180],
    ],
  },
  {
    id: 2,
    bbox: [320, 290, 60, 14],
    width_px: 60,
    height_px: 14,
    area_px: 540,
    detected_area_px: 500,
    centroid: [350, 297],
    perimeter_px: 148.0,
    equivalent_diameter_px: 26.2,
    length_px: 61.5,
    thickness_px: 11.2,
    angle_deg: 15,
    elongation: 5.5,
    polarity: 'dark',
    mean_contrast: 64.0,
    peak_contrast: 80.0,
    contrast_to_noise: 10.1,
    touches_edge: true,
    detected_by: 'self_reference',
    why_flagged: 'Dark region, 540 px, contrast 64.0 gray levels = 10.1x the noise level.',
    severity: 'HIGH',
    contour: [
      [320, 290],
      [380, 290],
      [380, 304],
      [320, 304],
    ],
  },
];

afterEach(() => {
  vi.restoreAllMocks();
});

describe('Phase 7 Frontend Components & API Error Mapping', () => {
  it('DefectTable selection toggles the DefectOverlay highlight and vice versa', () => {
    render(
      <InspectionSelectionProvider initialTab="final_overlay">
        <ComparisonViewer
          uploadId="11111111-1111-1111-1111-111111111111"
          imageWidth={512}
          imageHeight={512}
          regions={SAMPLE_REGIONS}
        />
        <DefectTable regions={SAMPLE_REGIONS} inspectionSeverity="HIGH" />
      </InspectionSelectionProvider>
    );

    const row1 = screen.getByTestId('defect-row-1');
    const overlay1 = screen.getByTestId('overlay-defect-1');
    const row2 = screen.getByTestId('defect-row-2');
    const overlay2 = screen.getByTestId('overlay-defect-2');

    expect(row1).toHaveAttribute('data-selected', 'false');
    expect(overlay1).toHaveAttribute('data-selected', 'false');

    // Selecting row 1 highlights overlay 1
    fireEvent.click(row1);
    expect(row1).toHaveAttribute('data-selected', 'true');
    expect(overlay1).toHaveAttribute('data-selected', 'true');
    expect(overlay2).toHaveAttribute('data-selected', 'false');

    // Clicking overlay 2 selects row 2 and unselects row 1
    fireEvent.click(overlay2);
    expect(row2).toHaveAttribute('data-selected', 'true');
    expect(overlay2).toHaveAttribute('data-selected', 'true');
    expect(row1).toHaveAttribute('data-selected', 'false');
    expect(overlay1).toHaveAttribute('data-selected', 'false');

    // Clicking selected row 2 again toggles selection off
    fireEvent.click(row2);
    expect(row2).toHaveAttribute('data-selected', 'false');
    expect(overlay2).toHaveAttribute('data-selected', 'false');
  });

  it('ComparisonViewer tab switching changes the active image src', () => {
    const uploadId = '22222222-2222-2222-2222-222222222222';
    render(
      <InspectionSelectionProvider initialTab="final_overlay">
        <ComparisonViewer
          uploadId={uploadId}
          imageWidth={512}
          imageHeight={512}
          regions={SAMPLE_REGIONS}
        />
      </InspectionSelectionProvider>
    );

    const img = screen.getByTestId('viewer-active-image') as HTMLImageElement;
    expect(img.src).toContain(`/inspection/run/${uploadId}/image/final_overlay`);

    fireEvent.click(screen.getByRole('tab', { name: 'Difference' }));
    expect(img.src).toContain(`/inspection/run/${uploadId}/image/difference`);

    fireEvent.click(screen.getByRole('tab', { name: 'Pattern Overlay' }));
    expect(img.src).toContain(`/inspection/run/${uploadId}/image/pattern_overlay`);
  });

  it('ProcessingStatus renders stages (done, current, pending) from a mocked status sequence', async () => {
    const uploadId = '33333333-3333-3333-3333-333333333333';
    const stages = [
      'preparing_image',
      'preprocessing',
      'detecting',
      'segmenting',
      'filtering_noise',
      'measuring',
      'diagnosing_pattern',
      'generating_report',
    ];

    const sequence: InspectionStatusResponse[] = [
      {
        upload_id: uploadId,
        status: 'running',
        current_stage: 'detecting',
        completed_stages: ['preparing_image', 'preprocessing'],
        stage_history: ['preparing_image', 'preprocessing', 'detecting'],
        stages,
        error: null,
      },
      {
        upload_id: uploadId,
        status: 'running',
        current_stage: 'measuring',
        completed_stages: ['preparing_image', 'preprocessing', 'detecting', 'segmenting', 'filtering_noise'],
        stage_history: ['preparing_image', 'preprocessing', 'detecting', 'segmenting', 'filtering_noise', 'measuring'],
        stages,
        error: null,
      },
    ];

    let callIdx = 0;
    const mockPoll = vi.fn(async () => {
      const next = sequence[Math.min(callIdx, sequence.length - 1)];
      callIdx += 1;
      return next;
    });

    render(
      <ProcessingStatus
        uploadId={uploadId}
        isActive
        pollIntervalMs={50}
        pollFn={mockPoll}
      />
    );

    // First poll: preprocessing done, detecting current, measuring pending
    await waitFor(() => {
      expect(screen.getByText(/Stage: detecting/i)).toBeInTheDocument();
    });
    expect(document.querySelector('[data-stage="preprocessing"]')).toHaveAttribute('data-stage-state', 'done');
    expect(document.querySelector('[data-stage="detecting"]')).toHaveAttribute('data-stage-state', 'current');
    expect(document.querySelector('[data-stage="measuring"]')).toHaveAttribute('data-stage-state', 'pending');

    // Second poll advances stage to measuring
    await act(async () => {
      await new Promise((r) => setTimeout(r, 80));
    });
    await waitFor(() => {
      expect(screen.getByText(/Stage: measuring/i)).toBeInTheDocument();
    });
    expect(document.querySelector('[data-stage="detecting"]')).toHaveAttribute('data-stage-state', 'done');
    expect(document.querySelector('[data-stage="measuring"]')).toHaveAttribute('data-stage-state', 'current');
  });

  it('SeverityBadge renders NONE, LOW, MEDIUM, and HIGH variants with visual tags', () => {
    const { rerender } = render(<SeverityBadge severity="NONE" />);
    expect(screen.getByText('NONE')).toBeInTheDocument();

    rerender(<SeverityBadge severity="LOW" />);
    expect(screen.getByText('[L]')).toBeInTheDocument();
    expect(screen.getByText('LOW')).toBeInTheDocument();

    rerender(<SeverityBadge severity="MEDIUM" />);
    expect(screen.getByText('[M]')).toBeInTheDocument();
    expect(screen.getByText('MEDIUM')).toBeInTheDocument();

    rerender(<SeverityBadge severity="HIGH" />);
    expect(screen.getByText('[H]')).toBeInTheDocument();
    expect(screen.getByText('HIGH')).toBeInTheDocument();
  });

  it('api.ts maps structured backend error responses to Error with code, status, and message', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          detail: {
            code: 'RUN_NOT_FINISHED',
            message: 'Inspection run has not completed yet.',
          },
        }),
        { status: 409, statusText: 'Conflict', headers: { 'Content-Type': 'application/json' } }
      )
    );

    await expect(getInspectionResult('44444444-4444-4444-4444-444444444444')).rejects.toMatchObject({
      message: 'Inspection run has not completed yet.',
      code: 'RUN_NOT_FINISHED',
      status: 409,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // Also test uploadInspection error mapping (415 unsupported format)
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          detail: {
            code: 'UNSUPPORTED_FORMAT',
            message: 'File extension .bmp is not supported.',
          },
        }),
        { status: 415, statusText: 'Unsupported Media Type', headers: { 'Content-Type': 'application/json' } }
      )
    );

    const fakeFile = new File(['abc'], 'bad.bmp', { type: 'image/bmp' });
    await expect(uploadInspection(fakeFile)).rejects.toMatchObject({
      message: 'File extension .bmp is not supported.',
      code: 'UNSUPPORTED_FORMAT',
      status: 415,
    });
  });
});
