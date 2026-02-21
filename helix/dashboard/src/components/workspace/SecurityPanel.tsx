'use client';

import { useMemo } from 'react';
import { ShieldAlert, AlertOctagon, Clock, Layers } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Table } from '@/components/ui/Table';
import { AddressDisplay } from '@/components/ui/AddressDisplay';
import { TimeAgo } from '@/components/ui/TimeAgo';
import { Skeleton } from '@/components/ui/Skeleton';
import { useAdversarialMonitor } from '@/hooks/useTraining';
import { cn } from '@/lib/utils';

interface SecurityPanelProps {
  modelId: bigint;
}

const THREAT_LEVEL_STYLES: Record<string, { border: string; text: string; bg: string; label: string }> = {
  low: {
    border: 'border-helix-border',
    text: 'text-helix-dim',
    bg: 'bg-transparent',
    label: 'Low',
  },
  medium: {
    border: 'border-helix-border',
    text: 'text-helix-muted',
    bg: 'bg-white/[0.02]',
    label: 'Medium',
  },
  high: {
    border: 'border-white/[0.15]',
    text: 'text-helix-text2',
    bg: 'bg-white/[0.03]',
    label: 'High',
  },
  critical: {
    border: 'border-white/[0.25]',
    text: 'text-white',
    bg: 'bg-white/[0.04]',
    label: 'Critical',
  },
};

const EVENT_TYPE_LABELS: Record<string, string> = {
  invalid_proof: 'Invalid Proof',
  timeout: 'Timeout',
  malicious_gradient: 'Malicious Gradient',
  stake_slashed: 'Stake Slashed',
  challenge_submitted: 'Challenge',
  byzantine_behavior: 'Byzantine',
};

const SEVERITY_STYLES: Record<string, string> = {
  warning: 'bg-white/[0.04] text-helix-muted',
  critical: 'bg-white/[0.10] text-white',
  resolved: 'bg-white/[0.04] text-helix-dim',
};

export function SecurityPanel({ modelId }: SecurityPanelProps) {
  const {
    adversarialEvents,
    criticalEvents,
    recentEvents,
    eventsByType,
    threatLevel,
    acknowledgeAlert,
    isLoading,
  } = useAdversarialMonitor(modelId);

  const threatStyle = THREAT_LEVEL_STYLES[threatLevel] ?? THREAT_LEVEL_STYLES.low;

  const eventTypeBreakdown = useMemo(() => {
    const entries: { type: string; count: number }[] = [];
    eventsByType.forEach((events, type) => {
      entries.push({ type, count: events.length });
    });
    return entries.sort((a, b) => b.count - a.count);
  }, [eventsByType]);

  const tableColumns = useMemo(
    () => [
      { key: 'type', label: 'Type' },
      { key: 'severity', label: 'Severity' },
      { key: 'prover', label: 'Prover' },
      { key: 'description', label: 'Description' },
      { key: 'time', label: 'Time', align: 'right' as const },
      { key: 'status', label: 'Status', align: 'center' as const },
    ],
    [],
  );

  const tableData = useMemo(
    () =>
      adversarialEvents.map((event) => ({
        type: (
          <Badge className="bg-white/[0.06] text-helix-muted">
            {EVENT_TYPE_LABELS[event.type] ?? event.type}
          </Badge>
        ),
        severity: (
          <Badge className={cn(SEVERITY_STYLES[event.severity])}>
            {event.severity}
          </Badge>
        ),
        prover: <AddressDisplay address={event.prover} />,
        description: (
          <span className="text-sm text-helix-text2 line-clamp-1">
            {event.description}
          </span>
        ),
        time: <TimeAgo timestamp={event.timestamp} />,
        status: event.resolved ? (
          <Badge className="bg-white/[0.04] text-helix-dim">Resolved</Badge>
        ) : (
          <button
            type="button"
            onClick={() => acknowledgeAlert(event.id)}
            className="text-2xs text-helix-muted hover:text-white transition-colors cursor-pointer"
          >
            Acknowledge
          </button>
        ),
      })),
    [adversarialEvents, acknowledgeAlert],
  );

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton height="8rem" />
        <div className="flex gap-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} width="8rem" height="2rem" />
          ))}
        </div>
        <Skeleton height="20rem" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Threat Level */}
      <Card
        variant="default"
        className={cn(
          'space-y-3',
          threatStyle.border,
          threatStyle.bg,
        )}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-sm text-helix-muted">
            Threat Level
          </h3>
          <ShieldAlert size={18} className={threatStyle.text} />
        </div>
        <p className={cn('text-3xl font-light tracking-tight', threatStyle.text)}>
          {threatStyle.label}
        </p>
        <p className="text-2xs text-helix-dim">
          {threatLevel === 'low' && 'No significant threats detected'}
          {threatLevel === 'medium' && 'Some anomalous activity detected'}
          {threatLevel === 'high' && 'Active threats require attention'}
          {threatLevel === 'critical' && 'Critical security events in progress'}
        </p>
      </Card>

      {/* Stats Row */}
      <div className="flex items-center gap-3 flex-wrap">
        <Badge variant="outline" className="gap-1.5 px-3 py-1">
          <AlertOctagon size={12} />
          <span>{criticalEvents.length} critical</span>
        </Badge>
        <Badge variant="outline" className="gap-1.5 px-3 py-1">
          <Clock size={12} />
          <span>{recentEvents.length} recent (1h)</span>
        </Badge>
        <Badge variant="outline" className="gap-1.5 px-3 py-1">
          <Layers size={12} />
          <span>{eventTypeBreakdown.length} event types</span>
        </Badge>
        {eventTypeBreakdown.slice(0, 3).map((entry) => (
          <Badge key={entry.type} variant="default" className="px-3 py-1">
            {EVENT_TYPE_LABELS[entry.type] ?? entry.type}: {entry.count}
          </Badge>
        ))}
      </div>

      {/* Critical Alert Banner */}
      {criticalEvents.length > 0 && (
        <Card
          variant="default"
          className="border-white/20 space-y-2"
        >
          <div className="flex items-center gap-2">
            <AlertOctagon size={14} className="text-white shrink-0" />
            <p className="text-sm text-white font-medium">
              Active Threats Detected
            </p>
          </div>
          <p className="text-2xs text-helix-text2">
            {criticalEvents.length} critical security{' '}
            {criticalEvents.length === 1 ? 'event requires' : 'events require'}{' '}
            immediate attention. Potential stake slashing or proof manipulation
            detected.
          </p>
          <div className="flex items-center gap-2 pt-1">
            {criticalEvents.slice(0, 3).map((event) => (
              <Badge key={event.id} className="bg-white/[0.08] text-white">
                {EVENT_TYPE_LABELS[event.type] ?? event.type}
              </Badge>
            ))}
            {criticalEvents.length > 3 && (
              <span className="text-2xs text-helix-muted">
                +{criticalEvents.length - 3} more
              </span>
            )}
          </div>
        </Card>
      )}

      {/* Events Table */}
      <div className="space-y-3">
        <h3 className="text-sm text-helix-muted">
          Security Events
        </h3>
        {adversarialEvents.length > 0 ? (
          <Card variant="default" className="p-0 overflow-hidden">
            <Table columns={tableColumns} data={tableData} />
          </Card>
        ) : (
          <p className="text-2xs text-helix-dim py-8 text-center">
            No security events recorded
          </p>
        )}
      </div>
    </div>
  );
}
