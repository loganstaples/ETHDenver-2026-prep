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
    model_name: "",
    current_loss: null,
    current_step: null,
    total_steps: null,
  });

  useEffect(() => {
    const poll = async () => {
      try {
        const t = await getTrainingStatus();
        setTraining(t);
      } catch {
        // Backend not ready
      }
    };

    poll();
    const interval = setInterval(poll, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  return training;
}
