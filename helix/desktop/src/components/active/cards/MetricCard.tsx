import { motion } from "framer-motion";
import { cn } from "../../../lib/utils";
import type { CardId } from "../../../hooks/useExpandedCard";
import type { ReactNode } from "react";

interface MetricCardProps {
  id: CardId;
  expandedCard: CardId;
  onExpand: (id: CardId) => void;
  value: string;
  label: string;
  expandedContent: ReactNode;
}

export function MetricCard({
  id,
  expandedCard,
  onExpand,
  value,
  label,
  expandedContent,
}: MetricCardProps) {
  const isExpanded = expandedCard === id;
  const isOtherExpanded = expandedCard !== null && expandedCard !== id;

  return (
    <motion.div
      layout
      onClick={() => onExpand(id)}
      className={cn(
        "rounded-xl cursor-pointer overflow-hidden transition-colors",
        isExpanded
          ? "bg-bg-card border border-accent/10 backdrop-blur-sm col-span-2 row-span-2"
          : isOtherExpanded
            ? "bg-bg-card/50"
            : "bg-bg-card hover:bg-bg-card-hover border border-transparent hover:border-border-subtle"
      )}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      {isOtherExpanded ? (
        /* Pill mode: just label */
        <div className="px-3 py-2.5 flex items-center gap-2">
          <span className="text-label text-text-tertiary">{label}</span>
          <span className="font-mono text-text-secondary text-xs">{value}</span>
        </div>
      ) : isExpanded ? (
        /* Expanded mode */
        <div className="p-5">
          <div className="flex items-baseline justify-between mb-4">
            <div>
              <span className="text-metric-lg text-text-primary">{value}</span>
              <span className="text-label text-text-tertiary ml-3">{label}</span>
            </div>
            <span className="text-label-sm text-text-tertiary">ESC to close</span>
          </div>
          <div>{expandedContent}</div>
        </div>
      ) : (
        /* Compact mode: big number + label */
        <div className="p-4 flex flex-col gap-1.5">
          <span className="text-metric-hero text-text-primary">{value}</span>
          <span className="text-label text-text-tertiary">{label}</span>
        </div>
      )}
    </motion.div>
  );
}
