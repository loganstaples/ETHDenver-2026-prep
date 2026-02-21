'use client';

import { Play, Pause } from 'lucide-react';
import { useTraining } from '@/hooks/useTraining';
import { Card } from '@/components/ui/Card';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { Skeleton } from '@/components/ui/Skeleton';
import { LossCurve } from '@/components/workspace/LossCurve';
import { RoundHistory } from '@/components/workspace/RoundHistory';
import { WorkerList } from '@/components/workspace/WorkerList';

interface TrainingPanelProps {
  modelId: bigint;
}

interface ConfigItem {
  label: string;
  value: string;
}

export function TrainingPanel({ modelId }: TrainingPanelProps) {
  const { config, progress, isLoading } = useTraining({ modelId });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton height="160px" />
        <Skeleton height="80px" />
        <Skeleton height="400px" />
      </div>
    );
  }

  const configItems: ConfigItem[] = [
    { label: 'Optimizer', value: config.optimizer },
    { label: 'Learning Rate', value: config.learningRate.toExponential(1) },
    { label: 'Loss Function', value: config.lossFunction },
    { label: 'Batch Size', value: config.batchSize.toString() },
    { label: 'Epochs', value: `${progress.currentEpoch} / ${config.totalEpochs}` },
    { label: 'MPC Threshold', value: `${config.mpcThreshold}-of-N` },
  ];

  return (
    <div className="space-y-6">
      {/* Config Summary */}
      <Card>
        <p className="text-sm text-helix-muted mb-4">
          Training Configuration
        </p>
        <div className="grid grid-cols-2 gap-x-8 gap-y-3">
          {configItems.map((item) => (
            <div key={item.label} className="flex items-baseline justify-between">
              <span className="text-sm text-helix-muted">
                {item.label}
              </span>
              <span className="text-sm text-helix-text2 font-mono">
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {/* Progress Section */}
      <Card>
        <p className="text-sm text-helix-muted mb-4">
          Progress
        </p>
        <div className="space-y-4">
          <div>
            <div className="flex items-baseline justify-between mb-1.5">
              <span className="text-2xs text-helix-muted">Epoch Progress</span>
              <span className="text-2xs font-mono text-helix-text2">
                {progress.epochProgress.toFixed(1)}%
              </span>
            </div>
            <ProgressBar value={progress.epochProgress} size="md" />
          </div>
          <div>
            <div className="flex items-baseline justify-between mb-1.5">
              <span className="text-2xs text-helix-muted">Overall Progress</span>
              <span className="text-2xs font-mono text-helix-text2">
                {progress.overallProgress.toFixed(1)}%
              </span>
            </div>
            <ProgressBar value={progress.overallProgress} size="md" />
          </div>
          {progress.estimatedTimeRemaining > 0 && (
            <p className="text-2xs text-helix-dim">
              Estimated time remaining: {formatDuration(progress.estimatedTimeRemaining * 1000)}
            </p>
          )}
        </div>
      </Card>

      {/* Training Controls */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-helix-text2 border border-helix-border rounded-md hover:border-helix-border2 hover:text-helix-text transition-colors"
        >
          <Play size={14} />
          Start Round
        </button>
        <button
          type="button"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-helix-muted border border-helix-border rounded-md hover:border-helix-border2 hover:text-helix-text2 transition-colors"
        >
          <Pause size={14} />
          Pause
        </button>
      </div>

      {/* Sub-sections */}
      <LossCurve modelId={modelId} />
      <RoundHistory modelId={modelId} />
      <WorkerList modelId={modelId} />
    </div>
  );
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return `${minutes}m ${seconds}s`;

  const hours = Math.floor(minutes / 60);
  const remainMinutes = minutes % 60;
  return `${hours}h ${remainMinutes}m`;
}
