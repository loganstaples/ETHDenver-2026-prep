'use client';

import { useMemo } from 'react';
import { Users } from 'lucide-react';
import { useWorkerHealth } from '@/hooks/useWorkerHealth';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Table } from '@/components/ui/Table';
import { AddressDisplay } from '@/components/ui/AddressDisplay';
import { TimeAgo } from '@/components/ui/TimeAgo';
import { Skeleton } from '@/components/ui/Skeleton';
import { EmptyState } from '@/components/ui/EmptyState';

interface WorkerListProps {
  modelId: bigint;
}

const STATUS_VARIANT: Record<string, 'default' | 'outline' | 'pulse'> = {
  healthy: 'default',
  degraded: 'outline',
  unhealthy: 'outline',
  offline: 'outline',
  unknown: 'outline',
};

const ROLE_VARIANT: Record<string, 'default' | 'outline' | 'pulse'> = {
  compute: 'default',
  aggregator: 'default',
  verifier: 'default',
};

export function WorkerList({ modelId }: WorkerListProps) {
  const { workers, isLoading } = useWorkerHealth({ modelId: Number(modelId) });

  const tableData = useMemo(() => {
    return workers.map((worker) => ({
      address: <AddressDisplay address={worker.address} />,
      role: (
        <Badge variant={ROLE_VARIANT[worker.role] ?? 'default'}>
          {worker.role.toUpperCase()}
        </Badge>
      ),
      status: (
        <Badge variant={STATUS_VARIANT[worker.status] ?? 'outline'}>
          {worker.status.toUpperCase()}
        </Badge>
      ),
      activity: (
        <span className="text-sm text-helix-text2 capitalize">
          {worker.activity}
        </span>
      ),
      proofs: (
        <span className="font-mono text-sm text-helix-text2">
          {worker.performance.proofsSubmitted}
        </span>
      ),
      lastActive: <TimeAgo timestamp={worker.lastSeen} />,
    }));
  }, [workers]);

  if (isLoading) {
    return (
      <Card>
        <Skeleton height="24px" width="120px" className="mb-4" />
        <Skeleton height="200px" />
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-baseline justify-between mb-4">
        <p className="text-sm text-helix-muted">
          Workers
        </p>
        <span className="text-2xs text-helix-dim">
          {workers.length} connected
        </span>
      </div>

      {tableData.length > 0 ? (
        <Table
          columns={[
            { key: 'address', label: 'Address', mono: true },
            { key: 'role', label: 'Role' },
            { key: 'status', label: 'Status' },
            { key: 'activity', label: 'Activity' },
            { key: 'proofs', label: 'Proofs Submitted', align: 'right' },
            { key: 'lastActive', label: 'Last Active', align: 'right' },
          ]}
          data={tableData}
        />
      ) : (
        <EmptyState
          icon={<Users size={24} />}
          title="No workers connected"
          description="Workers will appear here when they join the training network for this model."
        />
      )}
    </Card>
  );
}
