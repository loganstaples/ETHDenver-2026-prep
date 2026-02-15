import { cn } from "../../lib/utils";
import type { ActivityEvent } from "../../lib/types";
import { PanelHeader } from "../ui/Panel";

interface ActivityLogProps {
  events: ActivityEvent[];
}

const typeStyles: Record<string, { badge: string; text: string }> = {
  info: { badge: "text-text-tertiary", text: "text-text-secondary" },
  success: { badge: "text-status-active", text: "text-text-secondary" },
  earn: { badge: "text-status-active font-semibold", text: "text-status-active" },
  warning: { badge: "text-status-warning", text: "text-text-secondary" },
  error: { badge: "text-status-error", text: "text-text-secondary" },
};

export function ActivityLog({ events }: ActivityLogProps) {
  return (
    <div className="h-full bg-bg-panel p-3 flex flex-col overflow-hidden">
      <PanelHeader>ACTIVITY</PanelHeader>

      <div className="flex-1 overflow-y-auto">
        {events.length === 0 ? (
          <div className="text-body text-text-tertiary text-center py-4">
            No activity yet
          </div>
        ) : (
          <div className="flex flex-col">
            {events.map((event, i) => {
              const style = typeStyles[event.event_type] ?? typeStyles.info;
              return (
                <div
                  key={event.id}
                  className={cn(
                    "flex items-start gap-2 py-0.5 border-b border-border-grid",
                    i === 0 && "animate-fade-in-up"
                  )}
                >
                  {/* Timestamp */}
                  <span className="text-mono-data text-text-muted shrink-0 tabular-nums w-[60px]">
                    {formatLogTime(event.timestamp)}
                  </span>
                  {/* Message */}
                  <span className={cn("text-mono-data flex-1", style.text)}>
                    {event.message}
                  </span>
                  {/* Type badge */}
                  <span
                    className={cn(
                      "text-[9px] font-medium uppercase shrink-0 w-[32px] text-right",
                      style.badge
                    )}
                  >
                    {event.event_type === "success" ? "OK" : event.event_type.toUpperCase()}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function formatLogTime(isoTimestamp: string): string {
  try {
    const d = new Date(isoTimestamp);
    return d.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return isoTimestamp;
  }
}
