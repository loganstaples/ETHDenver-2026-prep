import { motion } from "framer-motion";
import { Card, CardHeader } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { ProgressBar } from "../components/ui/ProgressBar";
import { useTrainingStatus } from "../hooks/useTrainingStatus";

const phases = ["Assigned", "Training", "Proving", "Submitted", "Verified"];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.03 },
  },
};

const item = {
  hidden: { opacity: 0, y: 4 },
  show: { opacity: 1, y: 0, transition: { duration: 0.15, ease: "easeOut" as const } },
};

export function Training() {
  const training = useTrainingStatus();

  const currentPhaseIndex = phases.findIndex(
    (p) => p.toLowerCase() === training.phase.toLowerCase()
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-text-primary tracking-tight">
          Training
        </h1>
        <p className="text-sm text-text-tertiary">
          Training round detail and history
        </p>
      </div>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-4"
      >
        {/* Current Round Detail */}
        <motion.div variants={item}>
          <Card padding="lg">
            <CardHeader
              title={
                training.current_round !== null
                  ? `Round #${training.current_round}`
                  : "No Active Round"
              }
              action={
                training.active ? (
                  <Badge variant="active">In Progress</Badge>
                ) : (
                  <Badge variant="idle">Idle</Badge>
                )
              }
            />

            {training.active ? (
              <div className="space-y-6">
                {/* Phase Timeline */}
                <div className="flex items-center gap-0">
                  {phases.map((phase, i) => {
                    const isComplete = i < currentPhaseIndex;
                    const isCurrent = i === currentPhaseIndex;
                    return (
                      <div key={phase} className="flex items-center flex-1">
                        <div className="flex flex-col items-center flex-1">
                          <div
                            className={`h-2 w-2 rounded-full ${
                              isComplete
                                ? "bg-accent"
                                : isCurrent
                                  ? "bg-accent"
                                  : "bg-border-secondary"
                            }`}
                          />
                          <span
                            className={`text-[10px] mt-1.5 ${
                              isCurrent
                                ? "text-text-primary font-medium"
                                : isComplete
                                  ? "text-text-secondary"
                                  : "text-text-disabled"
                            }`}
                          >
                            {phase}
                          </span>
                        </div>
                        {i < phases.length - 1 && (
                          <div
                            className={`h-px flex-1 -mt-4 ${
                              isComplete ? "bg-accent" : "bg-border-secondary"
                            }`}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Details */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-[11px] text-text-tertiary uppercase tracking-wide">
                      Phase
                    </span>
                    <p className="text-sm text-text-primary mt-0.5">
                      {training.phase}
                    </p>
                  </div>
                  <div>
                    <span className="text-[11px] text-text-tertiary uppercase tracking-wide">
                      Progress
                    </span>
                    <div className="mt-1">
                      <ProgressBar
                        value={training.progress * 100}
                        size="md"
                        showLabel
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-sm text-text-tertiary">
                  No active training round
                </p>
                <p className="text-xs text-text-disabled mt-1">
                  Rounds begin when the network assigns work to your node
                </p>
              </div>
            )}
          </Card>
        </motion.div>

        {/* Round History */}
        <motion.div variants={item}>
          <Card>
            <CardHeader title="Round History" />
            {training.rounds_completed > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border-primary">
                      <th className="text-left text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Round
                      </th>
                      <th className="text-left text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Status
                      </th>
                      <th className="text-left text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Duration
                      </th>
                      <th className="text-right text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Earned
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border-primary">
                      <td className="py-2.5 font-mono text-text-secondary">
                        --
                      </td>
                      <td className="py-2.5 text-text-tertiary">
                        No history yet
                      </td>
                      <td className="py-2.5 text-text-tertiary">--</td>
                      <td className="py-2.5 text-right text-text-tertiary">
                        --
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-text-tertiary text-center py-6">
                No completed rounds yet
              </p>
            )}
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
}
