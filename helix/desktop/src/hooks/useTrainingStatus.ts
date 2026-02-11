import { useState, useEffect } from "react";
import type { TrainingStatus } from "../lib/types";
import { getTrainingStatus } from "../lib/tauri";

const POLL_INTERVAL = 2000;

export function useTrainingStatus() {
  const [training, setTraining] = useState<TrainingStatus>({
    active: false,
    current_round: null,
    phase: "Idle",
    progress: 0,
    rounds_completed: 0,
    proofs_generated: 0,
    total_earned: 0,
    session_earned: 0,
  });

  useEffect(() => {
    const fetch = async () => {
      try {
        const t = await getTrainingStatus();
        setTraining(t);
      } catch {
        // Backend not ready
      }
    };

    fetch();
    const interval = setInterval(fetch, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  return training;
}
