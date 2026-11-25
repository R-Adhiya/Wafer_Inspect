import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { NoiseFunnelChart } from './NoiseFunnelChart';
import { NoiseFilteringCard } from './NoiseFilteringCard';
import { NoiseExperimentPage } from '../pages/NoiseExperimentPage';
import * as api from '../services/api';

afterEach(() => {
  vi.restoreAllMocks();
});

const MOCK_CASES: api.NoiseExperimentCase[] = [
  {
    stage_id: 1,
    stage_name: 'raw_threshold',
    label: '1. Raw Detection',
    candidate_count: 24,
    tp: 8,
    fp: 16,
    fn: 0,
    precision: 0.3333,
    recall: 1.0,
  },
  {
    stage_id: 2,
    stage_name: 'min_area_only',
    label: '2. Min-Area Filter Only',
    candidate_count: 14,
    tp: 8,
    fp: 6,
    fn: 0,
    precision: 0.5714,
    recall: 1.0,
  },
  {
    stage_id: 3,
    stage_name: 'after_morphology_and_area',
    label: '3. Morphology + Area',
    candidate_count: 9,
    tp: 8,
    fp: 1,
    fn: 0,
    precision: 0.8889,
    recall: 1.0,
  },
  {
    stage_id: 4,
    stage_name: 'final_defects',
    label: '4. Final Filtered Defects',
    candidate_count: 8,
    tp: 8,
    fp: 0,
    fn: 0,
    precision: 1.0,
    recall: 1.0,
  },
];

const MOCK_EXPERIMENT_RESPONSE: api.InspectionNoiseExperimentResponse = {
  inspection_id: 42,
  upload_id: 'bbbb2222-2222-2222-2222-222222222222',
  has_ground_truth: true,
  ground_truth_count: 8,
  tolerance_px: 15.0,
  cases: MOCK_CASES,
  summary:
    'Noise filtering removed an average of 16.0 false detections (66.7%) at sigma=10.0 while retaining 8.0 of 8 true defects.',
  disclaimer: 'Synthetic demonstration data',
};

const MOCK_SWEEP_RESPONSE: api.NoiseSweepResponse = {
  layout: 'edge_ring',
  path: 'self_reference',
  size: 512,
  noise_levels: [2.0, 5.0, 10.0],
  seeds: [42, 43],
  true_defect_count: 8,
  series: [
    {
      noise_level: 2.0,
      seed_count: 2,
      summary:
        'Noise filtering removed an average of 5.0 false detections (38.5%) at sigma=2.0 while retaining 8.0 of 8 true defects.',
      cases: [
        {
          stage_id: 1,
          stage_name: 'raw_threshold',
          label: '1. Raw Detection',
          mean_count: 13.0,
          std_count: 1.0,
          mean_precision: 0.6154,
          std_precision: 0.05,
          mean_recall: 1.0,
          std_recall: 0.0,
          mean_tp: 8.0,
          mean_fp: 5.0,
          mean_fn: 0.0,
        },
        {
          stage_id: 2,
          stage_name: 'min_area_only',
          label: '2. Min-Area Filter Only',
          mean_count: 11.0,
          std_count: 1.0,
          mean_precision: 0.7273,
          std_precision: 0.06,
          mean_recall: 1.0,
          std_recall: 0.0,
          mean_tp: 8.0,
          mean_fp: 3.0,
          mean_fn: 0.0,
        },
        {
          stage_id: 3,
          stage_name: 'after_morphology_and_area',
          label: '3. Morphology + Area',
          mean_count: 8.0,
          std_count: 0.0,
          mean_precision: 1.0,
          std_precision: 0.0,
          mean_recall: 1.0,
          std_recall: 0.0,
          mean_tp: 8.0,
          mean_fp: 0.0,
          mean_fn: 0.0,
        },
        {
          stage_id: 4,
          stage_name: 'final_defects',
          label: '4. Final Filtered Defects',
          mean_count: 8.0,
          std_count: 0.0,
          mean_precision: 1.0,
          std_precision: 0.0,
          mean_recall: 1.0,
          std_recall: 0.0,
          mean_tp: 8.0,
          mean_fp: 0.0,
          mean_fn: 0.0,
        },
      ],
    },
  ],
  summary:
    'Noise filtering removed an average of 5.0 false detections (38.5%) at sigma=2.0 while retaining 8.0 of 8 true defects.',
  disclaimer: 'Synthetic demonstration data',
};

describe('Phase 9 Frontend: Noise Filtering & Sweep Visualizations', () => {
  it('NoiseFunnelChart renders all 4 stages and accessible data table', () => {
    render(
      <NoiseFunnelChart
        cases={MOCK_CASES}
        groundTruthCount={8}
        title="Test Stage Progression"
        showTable={true}
      />
    );

    expect(screen.getByRole('region', { name: /Test Stage Progression/i })).toBeInTheDocument();
    expect(screen.getByText(/Ground Truth Target: 8 defects/i)).toBeInTheDocument();

    // Check data table rows
    expect(screen.getByText('1. Raw Detection')).toBeInTheDocument();
    expect(screen.getByText('2. Min-Area Filter Only')).toBeInTheDocument();
    expect(screen.getByText('3. Morphology + Area')).toBeInTheDocument();
    expect(screen.getByText('4. Final Filtered Defects')).toBeInTheDocument();
  });

  it('NoiseFilteringCard loads and displays metrics, funnel, and synthetic disclaimer', async () => {
    vi.spyOn(api, 'getInspectionNoiseExperiment').mockResolvedValue(MOCK_EXPERIMENT_RESPONSE);

    render(<NoiseFilteringCard inspectionId={42} />);

    await waitFor(() => {
      expect(screen.getByText(/Noise Filtering & Candidate Progression/i)).toBeInTheDocument();
    });

    expect(screen.getByText('Synthetic demonstration data')).toBeInTheDocument();
    expect(screen.getByText(/Noise filtering removed an average of 16.0 false detections/i)).toBeInTheDocument();
    expect(screen.getByText(/Filtered:/i)).toBeInTheDocument();
  });

  it('NoiseExperimentPage renders controls, triggers sweep, and displays explanation', async () => {
    const sweepSpy = vi.spyOn(api, 'runNoiseSweep').mockResolvedValue(MOCK_SWEEP_RESPONSE);

    render(
      <MemoryRouter>
        <NoiseExperimentPage />
      </MemoryRouter>
    );

    // Initial sweep call on mount
    await waitFor(() => {
      expect(sweepSpy).toHaveBeenCalled();
    });

    expect(screen.getByText('Noise Filtering Experiment')).toBeInTheDocument();
    expect(screen.getByText('Synthetic demonstration data')).toBeInTheDocument();

    // Trigger explicit sweep
    const runBtn = screen.getByRole('button', { name: /Run Experiment Sweep/i });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(sweepSpy).toHaveBeenCalledTimes(2);
    });

    expect(screen.getByText(/Candidate Growth vs. Noise/i)).toBeInTheDocument();
    expect(screen.getByText(/Complete Precision & Recall Performance Table/i)).toBeInTheDocument();
  });

  it('NoiseExperimentPage enforces input validation on excessive noise levels', async () => {
    vi.spyOn(api, 'runNoiseSweep').mockResolvedValue(MOCK_SWEEP_RESPONSE);

    render(
      <MemoryRouter>
        <NoiseExperimentPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Noise Filtering Experiment')).toBeInTheDocument();
    });

    // Enter 9 noise levels (> 8 cap)
    const noiseInput = screen.getByLabelText(/Noise σ Levels/i);
    fireEvent.change(noiseInput, { target: { value: '1, 2, 3, 4, 5, 6, 7, 8, 9' } });

    const runBtn = screen.getByRole('button', { name: /Run Experiment Sweep/i });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(screen.getByText(/Maximum 8 noise levels are allowed per sweep/i)).toBeInTheDocument();
    });
  });
});
