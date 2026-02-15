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
      {/* Settings button */}
      <button
        onClick={onOpenSettings}
        className="absolute top-5 right-5 p-2 text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer"
      >
        <Settings size={18} />
      </button>

      {/* Orb — wordmark is now rendered inside */}
      <Orb onActivate={onActivate} phase={phase} />
    </motion.div>
  );
}
