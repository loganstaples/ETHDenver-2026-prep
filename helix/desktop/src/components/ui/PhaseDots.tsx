import { cn } from "../../lib/utils";

const PHASES = ["Assigned", "Training", "Proving", "Submitted", "Verified"];

interface PhaseDotsProps {
  currentPhase: string;
  className?: string;
}

export function PhaseDots({ currentPhase, className }: PhaseDotsProps) {
  const currentIndex = PHASES.findIndex(
    (p) => p.toLowerCase() === currentPhase.toLowerCase()
  );

  return (
    <div className={cn("flex items-center gap-1", className)}>
      {PHASES.map((phase, i) => {
        const isDone = i < currentIndex;
        const isCurrent = i === currentIndex;
        const isFuture = i > currentIndex;

        return (
          <div key={phase} className="flex items-center">
            <div className="flex flex-col items-center gap-0.5">
              <div
                className={cn(
                  "w-2 h-2 rounded-full border transition-colors",
                  isDone && "bg-status-active border-status-active",
                  isCurrent && "bg-text-bright border-text-bright animate-pulse-dot",
                  isFuture && "bg-transparent border-text-tertiary"
                )}
              />
              <span
                className={cn(
                  "text-[9px] font-medium leading-none whitespace-nowrap",
                  isDone && "text-status-active",
                  isCurrent && "text-text-bright",
                  isFuture && "text-text-tertiary"
                )}
              >
                {phase}
              </span>
            </div>
            {i < PHASES.length - 1 && (
              <div
                className={cn(
                  "w-3 h-px mx-0.5 mt-[-10px]",
                  isDone ? "bg-status-active" : "bg-border-primary"
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
