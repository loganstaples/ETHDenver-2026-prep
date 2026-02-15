import { motion, AnimatePresence } from "framer-motion";
import type { CardId, WindowState } from "../../hooks/useWindowManager";
import type { LucideIcon } from "lucide-react";

interface MinimizedDockProps {
  cards: WindowState[];
  cardMeta: Record<CardId, { title: string; icon: LucideIcon }>;
  onRestore: (id: CardId) => void;
}

export function MinimizedDock({ cards, cardMeta, onRestore }: MinimizedDockProps) {
  if (cards.length === 0) return null;

  return (
    <div className="flex items-center justify-center gap-2 py-2">
      <AnimatePresence>
        {cards.map((card) => {
          const meta = cardMeta[card.id];
          const Icon = meta.icon;
          return (
            <motion.button
              key={card.id}
              onClick={() => onRestore(card.id)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-bg-elevated border border-border-window text-text-secondary hover:text-text-primary hover:border-border-hover transition-colors cursor-pointer"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 25 }}
            >
              <Icon size={12} />
              <span className="text-label-sm">{meta.title}</span>
            </motion.button>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
