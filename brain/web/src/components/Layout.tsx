import { NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Settings as SettingsIcon,
  Users,
  Video,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Live", icon: Video, end: true },
  { to: "/people", label: "People", icon: Users, end: false },
  { to: "/events", label: "Events", icon: Activity, end: false },
  { to: "/analytics", label: "Analytics", icon: BarChart3, end: false },
  { to: "/settings", label: "Settings", icon: SettingsIcon, end: false },
];

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-accent/10 ring-1 ring-accent/30">
        <span className="absolute h-3.5 w-3.5 rounded-full border-2 border-accent" />
        <span className="h-1 w-1 rounded-full bg-accent" />
      </span>
      <div className="leading-tight">
        <p className="text-sm font-semibold tracking-tight text-ink">
          Kitchen Vision
        </p>
        <p className="text-[10px] uppercase tracking-[0.18em] text-ink-faint">
          brain
        </p>
      </div>
    </div>
  );
}

function NavItem({
  to,
  label,
  icon: Icon,
  end,
}: {
  to: string;
  label: string;
  icon: typeof Video;
  end: boolean;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn(
          "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
          isActive
            ? "bg-accent/10 text-accent ring-1 ring-inset ring-accent/30"
            : "text-ink-muted hover:bg-white/5 hover:text-ink",
        )
      }
    >
      <Icon className="h-4 w-4" />
      <span className="hidden sm:inline">{label}</span>
    </NavLink>
  );
}

export function Layout() {
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-40 border-b border-line bg-base-900/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Brand />
          <nav className="glass-soft flex items-center gap-1 p-1">
            {NAV.map((n) => (
              <NavItem key={n.to} {...n} />
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 animate-fade-in">
        <Outlet />
      </main>

      <footer className="mx-auto max-w-7xl px-4 pb-8 pt-2 text-center text-[11px] text-ink-faint sm:px-6">
        Kitchen Vision · local-first household perception · served by the brain
      </footer>
    </div>
  );
}
