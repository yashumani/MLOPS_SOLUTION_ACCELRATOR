import { NavLink, Outlet } from "react-router-dom";
import { Activity, BarChart3, Bell, ClipboardList, FileText, Gauge, Home, Play, Settings, SlidersHorizontal, Users } from "lucide-react";
import { useApi } from "../services/ApiContext";
import { useHealthQuery } from "../hooks/usePipelineQueries";
import { StatusBadge } from "./StatusBadge";

const navItems = [
  { to: "/", label: "Home", icon: Home },
  { to: "/submit", label: "Submit", icon: Play },
  { to: "/focus", label: "Focus", icon: Gauge },
  { to: "/configs", label: "Configs", icon: SlidersHorizontal },
  { to: "/auto-retrain", label: "Auto Retrain", icon: Activity },
  { to: "/drift", label: "Drift", icon: BarChart3 },
  { to: "/reports", label: "Reports", icon: FileText },
  { to: "/logs", label: "Logs", icon: ClipboardList },
  { to: "/settings", label: "Settings", icon: Settings },
  { to: "/notifications", label: "Notifications", icon: Bell }
];

export function Layout() {
  const { identity } = useApi();
  const health = useHealthQuery();
  const apiStatus = health.isError ? "Disconnected" : health.data?.status ?? "Checking";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">M3</span>
          <div>
            <strong>MLOps V3</strong>
            <small>Operations UI</small>
          </div>
        </div>
        <nav>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink key={item.to} to={item.to} end={item.to === "/"}>
                <Icon size={17} />
                {item.label}
              </NavLink>
            );
          })}
          {identity.mode === "entra" && identity.roles.includes("admin") && <NavLink to="/users"><Users size={17} />Users</NavLink>}
        </nav>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div>
            <strong>Azure ML workspace</strong>
            <span>{health.data?.workspace ?? "mlops-accelerator"}</span>
          </div>
          <StatusBadge status={apiStatus} />
        </header>
        <Outlet />
      </div>
    </div>
  );
}
