import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Activity,
  Globe,
  Settings,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react";
import { cn } from "../../lib/utils";
import { StatusDot } from "../ui/StatusDot";
import { motion } from "framer-motion";

interface SidebarProps {
  nodeRunning: boolean;
}

const navItems = [
  { path: "/", icon: LayoutDashboard, label: "Dashboard" },
  { path: "/training", icon: Activity, label: "Training" },
  { path: "/network", icon: Globe, label: "Network" },
  { path: "/settings", icon: Settings, label: "Settings" },
];

export function Sidebar({ nodeRunning }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <motion.aside
      animate={{ width: collapsed ? 56 : 200 }}
      transition={{ type: "spring", stiffness: 400, damping: 30 }}
      className="h-full bg-bg-secondary border-r border-border-primary flex flex-col overflow-hidden shrink-0"
    >
      {/* Logo */}
      <div className="h-14 flex items-center px-4 border-b border-border-primary">
        <span className="text-[15px] font-bold tracking-tight text-text-primary">
          {collapsed ? "H" : "HELIX"}
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2 flex flex-col gap-0.5">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md transition-colors duration-150 group relative",
                collapsed ? "justify-center px-2 py-2.5" : "px-3 py-2",
                isActive
                  ? "text-text-primary bg-bg-tertiary"
                  : "text-text-tertiary hover:text-text-secondary hover:bg-bg-tertiary/50"
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-accent rounded-r" />
                )}
                <item.icon size={18} strokeWidth={1.5} />
                {!collapsed && (
                  <span className="text-sm">{item.label}</span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom section */}
      <div className="px-3 py-3 border-t border-border-primary flex items-center gap-2.5">
        <StatusDot
          status={nodeRunning ? "active" : "idle"}
          pulse={nodeRunning}
          size="md"
        />
        {!collapsed && (
          <span className="text-xs text-text-tertiary flex-1">
            {nodeRunning ? "Mining" : "Offline"}
          </span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-text-tertiary hover:text-text-secondary transition-colors duration-150"
        >
          {collapsed ? (
            <PanelLeft size={16} strokeWidth={1.5} />
          ) : (
            <PanelLeftClose size={16} strokeWidth={1.5} />
          )}
        </button>
      </div>
    </motion.aside>
  );
}
