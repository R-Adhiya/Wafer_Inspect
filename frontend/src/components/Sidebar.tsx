import { LayoutDashboard, Disc, History, Settings, Cpu } from 'lucide-react';

export const Sidebar = () => {
  const navItems = [
    { name: 'Dashboard', icon: LayoutDashboard, active: true },
    { name: 'New Inspection', icon: Disc, active: false },
    { name: 'Inspection History', icon: History, active: false },
    { name: 'Settings', icon: Settings, active: false },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between h-screen shrink-0">
      <div>
        {/* Brand Logo */}
        <div className="p-6 border-b border-slate-800 flex items-center space-x-3">
          <div className="p-2 bg-sky-500/10 border border-sky-500/30 rounded-lg text-sky-400">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-slate-100 tracking-tight">WaferInspect</h1>
            <span className="text-xs text-sky-400 font-medium">Phase 1 Foundation</span>
          </div>
        </div>

        {/* Navigation items */}
        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.name}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                  item.active
                    ? 'bg-sky-500/10 text-sky-400 border border-sky-500/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{item.name}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Sidebar Footer */}
      <div className="p-4 border-t border-slate-800 text-xs text-slate-500">
        <div>WaferInspect v0.1.0</div>
        <div>Automated Defect Platform</div>
      </div>
    </aside>
  );
};
