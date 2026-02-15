import { motion } from "framer-motion";
import { Settings } from "lucide-react";
import { Orb } from "./Orb";

interface IdleViewProps {
  onActivate: () => void;
  phase: "idle" | "starting";
  onOpenSettings: () => void;
}

export function IdleView({ onActivate, phase, onOpenSettings }: IdleViewProps) {
  return (
    <motion.div
      className="h-full w-full flex flex-col items-center justify-center bg-bg-void relative"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Settings button — top right */}
      <button
        onClick={onOpenSettings}
        className="absolute top-5 right-5 p-2 text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer"
      >
        <Settings size={18} />
      </button>

      {/* Orb */}
      <Orb onActivate={onActivate} phase={phase} />

      {/* Wordmark */}
      <motion.div
        className="mt-10 flex flex-col items-center gap-2"
        animate={phase === "starting" ? { opacity: 0, y: 20 } : { opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h1
          className="text-text-primary tracking-[0.3em] text-sm font-light"
          style={{ fontFamily: "'Inter', sans-serif" }}
        >
          HELIX
        </h1>
        <p className="text-text-tertiary text-xs">
          Ready to compute
        </p>
      </motion.div>
    </motion.div>
  );
}
