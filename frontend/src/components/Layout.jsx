import { NavLink, Outlet } from "react-router-dom";
import { FlaskConical, Home, LayoutGrid, Microscope, Play, Scale } from "lucide-react";
import { useState } from "react";
import DemoTour from "./DemoTour.jsx";

const LINKS = [
  { to: "/", label: "Overview", icon: Home, end: true },
  { to: "/studio", label: "Composition Studio", icon: LayoutGrid },
  { to: "/results", label: "Results", icon: FlaskConical },
  { to: "/lab", label: "Research Lab", icon: Microscope },
  { to: "/compare", label: "Compare", icon: Scale },
];

export function CrystalLogo({ size = 30 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true">
      <g stroke="#0891B2" strokeWidth="2" fill="none">
        <path d="M8 14 L20 8 L32 14 L32 28 L20 34 L8 28 Z" />
        <path d="M8 14 L20 20 L32 14 M20 20 L20 34" />
      </g>
      <circle cx="20" cy="20" r="4" fill="#F59E0B" />
      {[[8, 14], [20, 8], [32, 14], [32, 28], [20, 34], [8, 28]].map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r="2.6" fill="#4F46E5" />
      ))}
    </svg>
  );
}

export default function Layout() {
  const [tour, setTour] = useState(false);
  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b border-line">
        <div className="mx-auto max-w-7xl px-4 h-16 flex items-center gap-6">
          <NavLink to="/" className="flex items-center gap-2 shrink-0">
            <CrystalLogo />
            <div className="leading-tight">
              <div className="font-display text-xl font-bold text-ink">MatSSL Studio</div>
              <div className="text-[11px] text-ink-mute -mt-0.5">semi-supervised materials discovery</div>
            </div>
          </NavLink>
          <nav className="hidden lg:flex items-center gap-1 flex-1">
            {LINKS.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold transition ${
                    isActive ? "bg-brand-50 text-brand" : "text-ink-soft hover:bg-surface hover:text-ink"
                  }`
                }
              >
                <Icon size={16} /> {label}
              </NavLink>
            ))}
          </nav>
          <button className="btn-primary ml-auto text-sm" onClick={() => setTour(true)}>
            <Play size={16} /> Demo tour
          </button>
        </div>
        <nav className="lg:hidden flex gap-1 overflow-x-auto px-4 pb-2">
          {LINKS.map(({ to, label, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) => `whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-semibold ${isActive ? "bg-brand-50 text-brand" : "text-ink-soft"}`}>
              {label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-6">
        <Outlet />
      </main>
      <footer className="border-t border-line bg-white py-4 text-center text-xs text-ink-mute">
        ESI-SBA · Machine Learning mini-project · Laref Ibrahim · Nemouchi Faycal Abdelhadi · Zerradi Abdelbasset · Medelci Djezzar Aymen
      </footer>
      {tour && <DemoTour onClose={() => setTour(false)} />}
    </div>
  );
}
