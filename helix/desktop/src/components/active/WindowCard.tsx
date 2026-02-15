import { motion } from "framer-motion";
import { Maximize2, Minus } from "lucide-react";
import type { ReactNode } from "react";
import type { CardId, WindowStatus } from "../../hooks/useWindowManager";
import type { LucideIcon } from "lucide-react";

interface WindowCardProps {
  id: CardId;
  status: WindowStatus;
  title: string;
  icon: LucideIcon;
  onFocus: (id: CardId) => void;
  onMinimize: (id: CardId) => void;
  children: ReactNode;
}

export function WindowCard({
  id,
  status,
  title,
  icon: Icon,
  onFocus,
  onMinimize,
  children,
}: WindowCardProps) {
  if (status === "minimized" || status === "focused") return null;

  return (
    <motion.div
      layout
      className="rounded-xl overflow-hidden border border-border-window bg-bg-card hover:border-border-hover transition-colors"
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      {/* Title bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-bg-window-chrome border-b border-border-window">
        <div className="flex items-center gap-2.5">
          <Icon size={14} className="text-text-tertiary" />
          <span className="text-label text-text-secondary">{title}</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); onFocus(id); }}
            className="p-1 text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer rounded hover:bg-bg-elevated"
          >
            <Maximize2 size={12} />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onMinimize(id); }}
            className="p-1 text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer rounded hover:bg-bg-elevated"
          >
            <Minus size={12} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-5 cursor-pointer" onClick={() => onFocus(id)}>
        {children}
      </div>
    </motion.div>
  );
}
