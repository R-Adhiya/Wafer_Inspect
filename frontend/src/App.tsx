import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { HomePage } from './pages/HomePage';
import { NewInspectionPage } from './pages/NewInspectionPage';
import { InspectionReportPage } from './pages/InspectionReportPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col justify-between font-sans antialiased">
        <Navbar />

        <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/inspection/new" element={<NewInspectionPage />} />
            <Route path="/inspection/:uploadId" element={<InspectionReportPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <footer className="border-t border-slate-200 bg-white py-4 px-6 text-center">
          <p className="text-xs text-slate-500 font-normal">
            WaferInspect is a working inspection application built on wafer-like grayscale imagery. It is not an industrial or fab-certified inspection system.
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
}
