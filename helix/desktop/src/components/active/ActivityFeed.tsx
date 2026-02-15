import { motion, AnimatePresence } from "framer-motion";
import type { ActivityEvent } from "../../lib/types";
import { cn } from "../../lib/utils";

interface ActivityFeedProps {
  events: ActivityEvent[];
}

const typeColor: Record<string, string> = {
  info: "text-text-tertiary",
  success: "text-status-healthy",
  warning: "text-status-warning",
  error: "text-status-error",
  earn: "text-accent",
};

export function ActivityFeed({ events }: ActivityFeedProps) {
  const recent = events.slice(0, 8);

  return (
    <div className="bg-bg-card rounded-xl p-3 overflow-hidden">
      <div className="flex items-center justify-between mb-2 px-1">
        <span className="text-label text-text-tertiary">Activity</span>
        <span className="text-label-sm text-text-tertiary">{events.length} events</span>
      </div>
      <div className="space-y-0.5 max-h-[140px] overflow-y-auto">
        <AnimatePresence mode="popLayout">
          {recent.map((event) => {
            const time = new Date(event.timestamp);
            const timeStr = time.toLocaleTimeString("en-US", {
              hour12: false,
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            });
            return (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
                className="flex items-center gap-3 px-1 py-1 rounded"
              >
                <span className="font-mono text-[10px] text-text-tertiary tabular-nums flex-shrink-0">
                  {timeStr}
                </span>
                <span className="text-xs text-text-secondary truncate flex-1">
                  {event.message}
                </span>
                <span className={cn("text-[10px] uppercase font-medium flex-shrink-0", typeColor[event.event_type] || "text-text-tertiary")}>
                  {event.event_type}
                </span>
              </motion.div>
            );
          })}
        </AnimatePresence>
        {events.length === 0 && (
          <p className="text-xs text-text-tertiary text-center py-3">No activity yet</p>
        )}
      </div>
    </div>
  );
}
