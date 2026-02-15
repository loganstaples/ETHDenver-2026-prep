import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Settings } from "lucide-react";
import { StartButton } from "./StartButton";

interface LaunchPadProps {
  onStart: () => void;
  onOpenSettings: () => void;
}

export function LaunchPad({ onStart, onOpenSettings }: LaunchPadProps) {
  const [starting, setStarting] = useState(false);

  const handleStart = useCallback(async () => {
    setStarting(true);
    await onStart();
  }, [onStart]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.05 }}
      transition={{ duration: 0.3 }}
      className="h-full w-full flex flex-col relative"
    >
      {/* Top bar */}
      <div className="flex items-center justify-between px-5 h-10 shrink-0">
        <span className="font-mono text-[14px] font-bold tracking-wider text-text-primary">
          HELIX
        </span>
        <button
          onClick={onOpenSettings}
          className="text-text-tertiary hover:text-text-primary transition-colors p-1.5"
        >
          <Settings size={16} />
        </button>
      </div>

      {/* Center content */}
      <div className="flex-1 flex flex-col items-center justify-center gap-6">
        <StartButton onStart={handleStart} starting={starting} />
        <span className="text-body text-text-tertiary">
          Ready to contribute compute
        </span>
      </div>

      {/* Bottom bar */}
      <div className="flex items-center justify-between px-5 h-8 shrink-0">
        <span className="text-caption text-text-muted">HELIX v0.1.0</span>
        <span className="text-caption text-text-muted">Disconnected</span>
      </div>
    </motion.div>
  );
}
