import { Card, CardHeader } from "../ui/Card";
import type { ActivityEvent } from "../../lib/types";
import { cn } from "../../lib/utils";

interface RecentActivityProps {
  events: ActivityEvent[];
}

export function RecentActivity({ events }: RecentActivityProps) {
  return (
    <Card className="col-span-2">
      <CardHeader title="Recent Activity" />
      {events.length > 0 ? (
        <div className="space-y-0">
          {events.map((event) => (
            <div
              key={event.id}
              className="flex items-start gap-3 py-2 border-b border-border-primary last:border-0"
            >
              <span className="text-[11px] text-text-disabled font-mono tabular-nums shrink-0 mt-0.5">
                {event.timestamp}
              </span>
              <span
                className={cn(
                  "text-sm",
                  event.event_type === "error"
                    ? "text-status-error"
                    : event.event_type === "warning"
                      ? "text-status-warning"
                      : "text-text-secondary"
                )}
              >
                {event.message}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-text-tertiary text-center py-4">
          No recent activity
        </p>
      )}
    </Card>
  );
}
