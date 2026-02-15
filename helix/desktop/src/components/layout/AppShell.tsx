import { useState, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import { useNodeStatus } from "../../hooks/useNodeStatus";
import { LaunchPad } from "../launch/LaunchPad";
import { CommandCenter } from "../dashboard/CommandCenter";
import { SettingsPanel } from "../dashboard/SettingsPanel";
import { SlideOver } from "../ui/SlideOver";

export function AppShell() {
  const { status, start, stop } = useNodeStatus();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const handleStart = useCallback(async () => {
    await start();
  }, [start]);

  const handleStop = useCallback(async () => {
    await stop();
  }, [stop]);

  const openSettings = useCallback(() => setSettingsOpen(true), []);
  const closeSettings = useCallback(() => setSettingsOpen(false), []);

  return (
    <div className="h-full w-full bg-bg-primary overflow-hidden">
      <AnimatePresence mode="wait">
        {!status.running ? (
          <LaunchPad
            key="launchpad"
            onStart={handleStart}
            onOpenSettings={openSettings}
          />
        ) : (
          <CommandCenter
            key="command-center"
            status={status}
            onStop={handleStop}
            onOpenSettings={openSettings}
          />
        )}
      </AnimatePresence>

      <SlideOver
        open={settingsOpen}
        onClose={closeSettings}
        title="SETTINGS"
      >
        <SettingsPanel />
      </SlideOver>
    </div>
  );
}
