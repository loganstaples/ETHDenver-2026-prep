import { useState, useCallback } from "react";

export type AppPhase = "idle" | "starting" | "active" | "stopping";

export function useAppState() {
  const [phase, setPhase] = useState<AppPhase>("idle");

  const beginStart = useCallback(() => {
    setPhase("starting");
  }, []);

  const completeStart = useCallback(() => {
    setPhase("active");
  }, []);

  const beginStop = useCallback(() => {
    setPhase("stopping");
  }, []);

  const completeStop = useCallback(() => {
    setPhase("idle");
  }, []);

  return { phase, beginStart, completeStart, beginStop, completeStop };
}
