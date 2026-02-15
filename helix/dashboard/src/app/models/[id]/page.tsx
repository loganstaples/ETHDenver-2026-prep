'use client';

import { useState } from 'react';
import { useParams } from 'next/navigation';
import { motion } from 'framer-motion';
import { useTraining } from '@/hooks/useTraining';
import { Badge } from '@/components/ui/Badge';
import { Tabs } from '@/components/ui/Tabs';
import { Skeleton } from '@/components/ui/Skeleton';
import { ModelOverview } from '@/components/workspace/ModelOverview';
import { TrainingPanel } from '@/components/workspace/TrainingPanel';
import { ProofPanel } from '@/components/workspace/ProofPanel';
import { ProofTimeline } from '@/components/workspace/ProofTimeline';
import { StakingPanel } from '@/components/workspace/StakingPanel';
import { ErrorBoundsPanel } from '@/components/workspace/ErrorBoundsPanel';
import { SecurityPanel } from '@/components/workspace/SecurityPanel';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'training', label: 'Training' },
  { id: 'proofs', label: 'Proofs' },
  { id: 'staking', label: 'Staking' },
  { id: 'error-bounds', label: 'Error Bounds' },
  { id: 'security', label: 'Security' },
] as const;

type TabId = (typeof TABS)[number]['id'];

const STATUS_VARIANT: Record<string, 'default' | 'outline' | 'pulse'> = {
  training: 'pulse',
  initializing: 'pulse',
  paused: 'outline',
  completed: 'default',
  failed: 'outline',
};

export default function ModelWorkspacePage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const modelId = BigInt(id);

  const [activeTab, setActiveTab] = useState<TabId>('overview');

  const { state, isLoading } = useTraining({ modelId });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="space-y-2">
          <Skeleton width="240px" height="1.5rem" />
          <Skeleton width="120px" height="1rem" />
        </div>
        <Skeleton width="100%" height="2.5rem" />
        <Skeleton width="100%" height="400px" />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-medium text-helix-text tracking-tight">
          {state?.modelName ?? `Model #${id}`}
        </h1>
        {state?.status && (
          <Badge variant={STATUS_VARIANT[state.status] ?? 'default'}>
            {state.status.toUpperCase()}
          </Badge>
        )}
      </div>

      {/* Tab Navigation */}
      <Tabs
        tabs={[...TABS]}
        activeTab={activeTab}
        onChange={(tabId) => setActiveTab(tabId as TabId)}
        className="border-b border-helix-border pb-px"
      />

      {/* Panel Content */}
      <div className="min-h-[400px]">
        {activeTab === 'overview' && <ModelOverview modelId={modelId} />}
        {activeTab === 'training' && <TrainingPanel modelId={modelId} />}
        {activeTab === 'proofs' && (
          <div className="space-y-6">
            <ProofPanel modelId={modelId} />
            <ProofTimeline modelId={modelId} />
          </div>
        )}
        {activeTab === 'staking' && <StakingPanel modelId={modelId} />}
        {activeTab === 'error-bounds' && <ErrorBoundsPanel modelId={modelId} />}
        {activeTab === 'security' && <SecurityPanel modelId={modelId} />}
      </div>
    </motion.div>
  );
}
