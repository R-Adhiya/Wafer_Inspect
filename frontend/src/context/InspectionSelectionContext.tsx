import React, { createContext, useCallback, useContext, useState } from 'react';
import { RunImageKind } from '../services/api';

export interface ZoomRequest {
  defectId: number;
  bbox: [number, number, number, number];
  nonce: number;
}

export interface InspectionSelectionContextValue {
  selectedDefectId: number | null;
  setSelectedDefectId: (id: number | null) => void;
  selectDefectFromOverlay: (id: number) => void;
  selectDefectFromTable: (id: number | null) => void;
  lastSelectionSource: 'overlay' | 'table' | null;
  activeTab: RunImageKind;
  setActiveTab: (tab: RunImageKind) => void;
  zoomRequest: ZoomRequest | null;
  requestZoomToDefect: (defectId: number, bbox: [number, number, number, number]) => void;
  clearZoomRequest: () => void;
}

const InspectionSelectionContext = createContext<InspectionSelectionContextValue | null>(null);

export interface InspectionSelectionProviderProps {
  initialTab?: RunImageKind;
  children: React.ReactNode;
}

export const InspectionSelectionProvider: React.FC<InspectionSelectionProviderProps> = ({
  initialTab = 'final_overlay',
  children,
}) => {
  const [selectedDefectId, setSelectedDefectIdState] = useState<number | null>(null);
  const [lastSelectionSource, setLastSelectionSource] = useState<'overlay' | 'table' | null>(null);
  const [activeTab, setActiveTab] = useState<RunImageKind>(initialTab);
  const [zoomRequest, setZoomRequest] = useState<ZoomRequest | null>(null);

  const setSelectedDefectId = useCallback((id: number | null) => {
    setSelectedDefectIdState(id);
    setLastSelectionSource(null);
  }, []);

  const selectDefectFromOverlay = useCallback((id: number) => {
    setSelectedDefectIdState(id);
    setLastSelectionSource('overlay');
  }, []);

  const selectDefectFromTable = useCallback((id: number | null) => {
    setSelectedDefectIdState(id);
    setLastSelectionSource('table');
  }, []);

  const requestZoomToDefect = useCallback(
    (defectId: number, bbox: [number, number, number, number]) => {
      setSelectedDefectIdState(defectId);
      setLastSelectionSource('table');
      setZoomRequest({ defectId, bbox, nonce: Date.now() + Math.random() });
    },
    []
  );

  const clearZoomRequest = useCallback(() => {
    setZoomRequest(null);
  }, []);

  return (
    <InspectionSelectionContext.Provider
      value={{
        selectedDefectId,
        setSelectedDefectId,
        selectDefectFromOverlay,
        selectDefectFromTable,
        lastSelectionSource,
        activeTab,
        setActiveTab,
        zoomRequest,
        requestZoomToDefect,
        clearZoomRequest,
      }}
    >
      {children}
    </InspectionSelectionContext.Provider>
  );
};

export function useInspectionSelection(): InspectionSelectionContextValue {
  const ctx = useContext(InspectionSelectionContext);
  if (!ctx) {
    throw new Error('useInspectionSelection must be used within an InspectionSelectionProvider');
  }
  return ctx;
}
