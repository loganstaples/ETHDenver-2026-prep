import { useState, useCallback, useEffect, useMemo } from "react";

export type CardId = "earnings" | "training" | "resources" | "network" | "loss" | "rounds";
export type WindowStatus = "normal" | "focused" | "minimized";

export interface WindowState {
  id: CardId;
  status: WindowStatus;
}

const ALL_CARDS: CardId[] = ["earnings", "training", "resources", "network", "loss", "rounds"];

export function useWindowManager() {
  const [windows, setWindows] = useState<WindowState[]>(
    ALL_CARDS.map((id) => ({ id, status: "normal" as WindowStatus }))
  );

  const focusedCard = useMemo(
    () => windows.find((w) => w.status === "focused")?.id ?? null,
    [windows]
  );

  const minimizedCards = useMemo(
    () => windows.filter((w) => w.status === "minimized"),
    [windows]
  );

  const normalCards = useMemo(
    () => windows.filter((w) => w.status === "normal"),
    [windows]
  );

  const focus = useCallback((id: CardId) => {
    setWindows((prev) =>
      prev.map((w) => (w.id === id ? { ...w, status: "focused" } : w))
    );
  }, []);

  const unfocus = useCallback(() => {
    setWindows((prev) =>
      prev.map((w) => (w.status === "focused" ? { ...w, status: "normal" } : w))
    );
  }, []);

  const minimize = useCallback((id: CardId) => {
    setWindows((prev) =>
      prev.map((w) => (w.id === id ? { ...w, status: "minimized" } : w))
    );
  }, []);

  const restore = useCallback((id: CardId) => {
    setWindows((prev) =>
      prev.map((w) => (w.id === id ? { ...w, status: "normal" } : w))
    );
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") unfocus();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [unfocus]);

  return { windows, focusedCard, minimizedCards, normalCards, focus, unfocus, minimize, restore };
}
