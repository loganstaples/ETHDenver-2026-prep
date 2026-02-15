import { useState, useCallback, useEffect } from "react";

export type CardId = "earnings" | "training" | "resources" | "network" | "loss" | "rounds" | null;

export function useExpandedCard() {
  const [expandedCard, setExpandedCard] = useState<CardId>(null);

  const expand = useCallback((id: CardId) => {
    setExpandedCard((prev) => (prev === id ? null : id));
  }, []);

  const collapse = useCallback(() => {
    setExpandedCard(null);
  }, []);

  // Escape key to collapse
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") collapse();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [collapse]);

  return { expandedCard, expand, collapse };
}
