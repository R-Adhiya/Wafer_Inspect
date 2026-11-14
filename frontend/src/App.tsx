import { Navbar } from './components/Navbar';
import { BackendStatus } from './components/BackendStatus';
import { NewInspectionPage } from './pages/NewInspectionPage';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col justify-between font-sans antialiased">
      {/* Top Navigation */}
      <Navbar />

      {/* Main Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Backend health status card */}
        <section aria-label="System Health">
          <BackendStatus />
        </section>

        {/* Phase 2: Inspection Upload & Preview Workflow */}
        <NewInspectionPage />
      </main>

      {/* Footer Landmark with updated disclaimer */}
      <footer className="border-t border-slate-200 bg-white py-4 px-6 text-center">
        <p className="text-xs text-slate-500 font-normal">
          WaferInspect is a working inspection application built on wafer-like grayscale imagery. It is not an industrial or fab-certified inspection system.
        </p>
      </footer>
    </div>
  );
}
