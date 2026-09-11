import { Sidebar } from './components/Sidebar';
import { DashboardPage } from './pages/DashboardPage';

export default function App() {
  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden">
      <Sidebar />
      <DashboardPage />
    </div>
  );
}
