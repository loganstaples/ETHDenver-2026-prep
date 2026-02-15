import { AnimatePresence } from "framer-motion";
import { useState, useCallback } from "react";
import { IdleView } from "./components/idle/IdleView";
import { ActiveView } from "./components/active/ActiveView";
import { SettingsOverlay } from "./components/settings/SettingsOverlay";
import { useAppState } from "./hooks/useAppState";
import { useNodeStatus } from "./hooks/useNodeStatus";

export default function App() {
  const { phase, beginStart, completeStart, beginStop, completeStop } = useAppState();
  const { start, stop } = useNodeStatus();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const handleActivate = useCallback(async () => {
    beginStart();
    try {
      await start();
    } catch {
      // Backend may not be available (dev mode without Tauri)
    }
    // Wait for orb animation to finish before showing dashboard
    setTimeout(() => {
      completeStart();
    }, 500);
  }, [beginStart, completeStart, start]);

  const handleStop = useCallback(async () => {
    beginStop();
    try {
      await stop();
    } catch {
      // Backend may not be available (dev mode without Tauri)
    }
    setTimeout(() => {
      completeStop();
    }, 600);
  }, [beginStop, completeStop, stop]);

  const isIdle = phase === "idle" || phase === "starting";
  const isActive = phase === "active" || phase === "stopping";

  return (
    <div className="h-full w-full bg-bg-void">
      <AnimatePresence mode="wait">
        {isIdle && (
          <IdleView
            key="idle"
            onActivate={handleActivate}
            phase={phase as "idle" | "starting"}
            onOpenSettings={() => setSettingsOpen(true)}
          />
        )}
        {isActive && (
          <ActiveView
            key="active"
            onStop={handleStop}
            onOpenSettings={() => setSettingsOpen(true)}
          />
        )}
      </AnimatePresence>
      <SettingsOverlay open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
