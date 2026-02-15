'use client';

import { motion } from 'framer-motion';
import { useNetworkHealth } from '@/hooks/useNodes';
import { NetworkHealthBar } from '@/components/network/NetworkHealthBar';
import { NodeGrid } from '@/components/network/NodeGrid';
import { Skeleton } from '@/components/ui/Skeleton';

export default function NetworkPage() {
  const { healthScore, healthStatus, issues, networkStats, isLoading } =
    useNetworkHealth();

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-6"
    >
      <h1 className="page-title">Network</h1>

      {isLoading ? (
        <div className="space-y-6">
          <Skeleton height="140px" />
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} height="220px" />
            ))}
          </div>
        </div>
      ) : (
        <>
          <NetworkHealthBar
            healthScore={healthScore}
            healthStatus={healthStatus}
            issues={issues}
            stats={networkStats}
          />
          <NodeGrid />
        </>
      )}
    </motion.div>
  );
}
