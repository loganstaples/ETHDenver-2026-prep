'use client';

import { Activity, Cpu, GitFork, ShieldCheck } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/utils';

interface NetworkHealthBarProps {
  healthScore: number;
  healthStatus: string;
  issues: string[];
  stats: {
    totalNodes: number;
    activeNodes: number;
    computeNodes: number;
    aggregators: number;
    verifiers: number;
    totalStaked: bigint;
    totalProofs: number;
    averageReputation: number;
    networkUptime: number;
  };
}

const statusBadgeVariant: Record<string, 'pulse' | 'default' | 'outline'> = {
  healthy: 'pulse',
  good: 'pulse',
  degraded: 'default',
  poor: 'outline',
  critical: 'outline',
};

const statusLabel: Record<string, string> = {
  healthy: 'Healthy',
  good: 'Good',
  degraded: 'Degraded',
  poor: 'Poor',
  critical: 'Critical',
};

const statItems = [
  { key: 'activeNodes', label: 'Active Nodes', icon: Activity },
  { key: 'computeNodes', label: 'Compute', icon: Cpu },
  { key: 'aggregators', label: 'Aggregators', icon: GitFork },
  { key: 'verifiers', label: 'Verifiers', icon: ShieldCheck },
] as const;

export function NetworkHealthBar({
  healthScore,
  healthStatus,
  issues,
  stats,
}: NetworkHealthBarProps) {
  return (
    <Card variant="glass" className="p-6">
      <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
        {/* Left: health score + status */}
        <div className="flex items-center gap-5">
          <div className="flex flex-col items-center">
            <span
              className={cn(
                'data-text text-4xl',
                healthScore >= 80 && 'text-white',
                healthScore >= 60 && healthScore < 80 && 'text-helix-text2',
                healthScore < 60 && 'text-helix-muted',
              )}
            >
              {healthScore}
            </span>
            <span className="label-text mt-1">Health Score</span>
          </div>

          <div className="h-10 w-px bg-helix-border hidden sm:block" />

          <Badge variant={statusBadgeVariant[healthStatus] ?? 'default'}>
            {statusLabel[healthStatus] ?? healthStatus}
          </Badge>
        </div>

        {/* Right: stat items */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-8 gap-y-3">
          {statItems.map(({ key, label, icon: Icon }) => (
            <div key={key} className="flex items-center gap-2.5">
              <Icon size={14} className="text-helix-dim flex-shrink-0" strokeWidth={1.5} />
              <div>
                <p className="font-mono text-sm text-white leading-none">
                  {stats[key]}
                </p>
                <p className="text-2xs text-helix-muted mt-0.5">{label}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Issues */}
      {issues.length > 0 && (
        <div className="mt-4 pt-4 border-t border-white/[0.06]">
          <ul className="flex flex-wrap gap-x-4 gap-y-1">
            {issues.map((issue) => (
              <li key={issue} className="text-2xs text-helix-muted">
                {issue}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
