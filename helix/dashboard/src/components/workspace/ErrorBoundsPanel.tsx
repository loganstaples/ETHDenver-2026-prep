'use client';

import { useMemo } from 'react';
import { Activity, TrendingUp, AlertTriangle } from 'lucide-react';
import { StatCard } from '@/components/ui/StatCard';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Table } from '@/components/ui/Table';
import { Chart } from '@/components/ui/Chart';
import { Skeleton } from '@/components/ui/Skeleton';
import { useErrorBounds } from '@/hooks/useTraining';
import { cn } from '@/lib/utils';

interface ErrorBoundsPanelProps {
  modelId: bigint;
}

const RISK_STYLES: Record<string, string> = {
  low: 'bg-white/[0.04] text-helix-dim',
  medium: 'bg-white/[0.06] text-helix-muted',
  high: 'bg-white/[0.08] text-helix-text2',
  critical: 'bg-white/[0.10] text-white',
};

function formatScientific(value: number): string {
  if (value === 0) return '0';
  if (Math.abs(value) < 0.0001) return value.toExponential(2);
  if (Math.abs(value) < 1) return value.toFixed(6);
  return value.toFixed(2);
}

export function ErrorBoundsPanel({ modelId }: ErrorBoundsPanelProps) {
  const {
    errorBoundHistory,
    layerErrorBounds,
    totalErrorBound,
    maxAmplification,
    highRiskLayers,
    accumulatedErrorBound,
    isWithinBounds,
    isLoading,
  } = useErrorBounds(modelId);

  // Transform error bound history for chart
  const chartData = useMemo(
    () =>
      errorBoundHistory.map((point) => ({
        epoch: `E${point.epoch}`,
        bound: point.bound,
      })),
    [errorBoundHistory],
  );

  // Transform layer error bounds for table
  const tableColumns = useMemo(
    () => [
      { key: 'layer', label: 'Layer', mono: true },
      { key: 'operation', label: 'Operation' },
      { key: 'inputBound', label: 'Input Bound', align: 'right' as const, mono: true },
      { key: 'outputBound', label: 'Output Bound', align: 'right' as const, mono: true },
      { key: 'amplification', label: 'Amplification', align: 'right' as const, mono: true },
      { key: 'riskLevel', label: 'Risk', align: 'center' as const },
    ],
    [],
  );

  const tableData = useMemo(
    () =>
      layerErrorBounds.map((layer) => ({
        layer: (
          <span className="font-mono text-2xs text-helix-text2">
            {layer.layer}
          </span>
        ),
        operation: (
          <span className="text-sm text-helix-text2">{layer.operation}</span>
        ),
        inputBound: (
          <span className="font-mono text-2xs text-helix-text2">
            {formatScientific(layer.inputBound)}
          </span>
        ),
        outputBound: (
          <span className="font-mono text-2xs text-helix-text2">
            {formatScientific(layer.outputBound)}
          </span>
        ),
        amplification: (
          <span className="font-mono text-2xs text-helix-text2">
            {layer.amplification.toFixed(2)}x
          </span>
        ),
        riskLevel: (
          <Badge className={cn(RISK_STYLES[layer.riskLevel])}>
            {layer.riskLevel}
          </Badge>
        ),
      })),
    [layerErrorBounds],
  );

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} height="5rem" />
          ))}
        </div>
        <Skeleton height="4rem" />
        <Skeleton height="300px" />
        <Skeleton height="16rem" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stat Cards */}
      <div className="grid grid-cols-3 gap-4">
        <StatCard
          label="Accumulated Error"
          value={formatScientific(accumulatedErrorBound)}
          icon={<Activity size={16} />}
        />
        <StatCard
          label="Max Amplification"
          value={`${maxAmplification.toFixed(1)}x`}
          icon={<TrendingUp size={16} />}
        />
        <StatCard
          label="High Risk Layers"
          value={highRiskLayers.length}
          icon={<AlertTriangle size={16} />}
        />
      </div>

      {/* Risk Assessment */}
      <Card variant="default" className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm text-helix-muted">
            Risk Assessment
          </h3>
          {isWithinBounds ? (
            <Badge className="bg-white/[0.06] text-helix-text2">
              Within Bounds
            </Badge>
          ) : (
            <Badge className="bg-white/[0.10] text-white">
              Exceeds Bounds
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-4">
          <div className="space-y-0.5">
            <p className="text-2xs text-helix-muted">Total Error Bound</p>
            <p className="font-mono text-sm text-helix-text">
              {formatScientific(totalErrorBound)}
            </p>
          </div>
          <div className="h-8 w-px bg-helix-border" />
          <div className="space-y-0.5">
            <p className="text-2xs text-helix-muted">Accumulated</p>
            <p className="font-mono text-sm text-helix-text">
              {formatScientific(accumulatedErrorBound)}
            </p>
          </div>
          <div className="h-8 w-px bg-helix-border" />
          <div className="space-y-0.5">
            <p className="text-2xs text-helix-muted">Max Allowed</p>
            <p className="font-mono text-sm text-helix-dim">
              1.00e-2
            </p>
          </div>
        </div>
      </Card>

      {/* Error Bound Chart */}
      <Card variant="default" className="space-y-3">
        <h3 className="text-sm text-helix-muted">
          Error Bound History
        </h3>
        {chartData.length > 0 ? (
          <Chart
            data={chartData}
            xKey="epoch"
            yKeys={[{ key: 'bound', label: 'Error Bound' }]}
            type="area"
            height={300}
          />
        ) : (
          <p className="text-2xs text-helix-dim py-8 text-center">
            No error bound history available
          </p>
        )}
      </Card>

      {/* Layer Propagation Table */}
      <div className="space-y-3">
        <h3 className="text-sm text-helix-muted">
          Layer Error Propagation
        </h3>
        <Card variant="default" className="p-0 overflow-hidden">
          <Table columns={tableColumns} data={tableData} />
        </Card>
      </div>
    </div>
  );
}
