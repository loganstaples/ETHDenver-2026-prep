import { motion, AnimatePresence } from "framer-motion";
import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

interface FocusOverlayProps {
  open: boolean;
  onClose: () => void;
  title: string;
  icon: LucideIcon;
  value: string;
  label: string;
  children: ReactNode;
}

export function FocusOverlay({ open, onClose, title, icon: Icon, value, label, children }: FocusOverlayProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-40 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0"
            style={{ backgroundColor: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)" }}
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />

          {/* Card */}
          <motion.div
            className="relative z-50 w-[70vw] max-w-3xl bg-bg-card border border-border-window rounded-xl overflow-hidden"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
          >
            {/* Header bar */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-border-window bg-bg-window-chrome">
              <div className="flex items-center gap-3">
                <Icon size={16} className="text-text-tertiary" />
                <span className="text-label text-text-secondary">{title}</span>
              </div>
              <span className="text-label-sm text-text-tertiary">ESC to close</span>
            </div>

            {/* Value header */}
            <div className="px-6 pt-5 pb-3 flex items-baseline gap-3">
              <span className="text-metric-lg text-text-primary">{value}</span>
              <span className="text-label text-text-tertiary">{label}</span>
            </div>

            {/* Detail content */}
            <div className="px-6 pb-6">
              {children}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
