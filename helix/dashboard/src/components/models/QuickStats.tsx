'use client';

import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Boxes, Activity, Server, ShieldCheck } from 'lucide-react';
import { StatCard } from '@/components/ui/StatCard';
import { useContractState } from '@/hooks/useContractState';
import { useNodes } from '@/hooks/useNodes';

export function QuickStats() {
  const { networkState, models, isLoading: contractLoading } = useContractState();
  const { networkStats } = useNodes({ autoRefresh: true });

  const totalRounds = useMemo(() => {
    if (contractLoading || models.length === 0) return null;
    return models.reduce(
      (sum, m) => sum + Number(m.currentRound),
      0,
    );
  }, [models, contractLoading]);

  const items = [
    {
      label: 'Active Models',
      value: contractLoading
        ? '--'
        : String(networkState?.totalModels ?? 0),
      icon: <Boxes size={22} strokeWidth={1.5} />,
    },
    {
      label: 'Total Rounds',
      value: totalRounds !== null ? totalRounds.toLocaleString() : '--',
      icon: <Activity size={22} strokeWidth={1.5} />,
    },
    {
      label: 'Network Nodes',
      value: `${networkStats.activeNodes}/${networkStats.totalNodes}`,
      icon: <Server size={22} strokeWidth={1.5} />,
    },
    {
      label: 'Proofs Verified',
      value: networkStats.totalProofs.toLocaleString(),
      icon: <ShieldCheck size={22} strokeWidth={1.5} />,
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {items.map((item, i) => (
        <motion.div
          key={item.label}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05, duration: 0.2 }}
        >
          <StatCard
            label={item.label}
            value={item.value}
            icon={item.icon}
          />
        </motion.div>
      ))}
    </div>
  );
}
