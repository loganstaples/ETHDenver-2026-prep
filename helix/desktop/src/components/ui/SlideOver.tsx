import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { cn } from "../../lib/utils";

interface SlideOverProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function SlideOver({ open, onClose, title, children, className }: SlideOverProps) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 bg-black/50 z-40"
            onClick={onClose}
          />
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className={cn(
              "fixed right-0 top-0 bottom-0 w-[380px] bg-bg-panel border-l border-border-primary z-50 flex flex-col",
              className
            )}
          >
            <div className="flex items-center justify-between px-4 h-10 border-b border-border-primary bg-bg-header shrink-0">
              <span className="text-label text-text-secondary">{title}</span>
              <button
                onClick={onClose}
                className="text-text-tertiary hover:text-text-primary transition-colors p-1"
              >
                <X size={14} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">{children}</div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
