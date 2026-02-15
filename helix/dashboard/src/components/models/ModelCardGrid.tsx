'use client';

import { motion } from 'framer-motion';
import { Boxes } from 'lucide-react';
import { Skeleton } from '@/components/ui/Skeleton';
import { EmptyState } from '@/components/ui/EmptyState';
import { ModelCard } from './ModelCard';
import type { ModelDetails } from '@/hooks/useContractState';

interface ModelCardGridProps {
  models: ModelDetails[];
  isLoading?: boolean;
}

function SkeletonCard() {
  return (
    <div className="bg-white/[0.02] backdrop-blur-md border border-white/[0.06] rounded-lg p-5 space-y-4">
      <div className="flex items-start justify-between">
        <Skeleton width="60%" height="1rem" />
        <Skeleton width="4rem" height="1.25rem" />
      </div>
      <Skeleton width="50%" height="0.75rem" />
      <div className="grid grid-cols-3 gap-3">
        <div className="space-y-1">
          <Skeleton width="100%" height="0.625rem" />
          <Skeleton width="70%" height="0.875rem" />
        </div>
        <div className="space-y-1">
          <Skeleton width="100%" height="0.625rem" />
          <Skeleton width="70%" height="0.875rem" />
        </div>
        <div className="space-y-1">
          <Skeleton width="100%" height="0.625rem" />
          <Skeleton width="70%" height="0.875rem" />
        </div>
      </div>
      <div className="space-y-2">
        <Skeleton width="100%" height="0.25rem" />
        <div className="flex justify-between">
          <Skeleton width="5rem" height="0.625rem" />
          <Skeleton width="4rem" height="0.625rem" />
        </div>
      </div>
    </div>
  );
}

export function ModelCardGrid({ models, isLoading }: ModelCardGridProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <EmptyState
        icon={<Boxes size={24} strokeWidth={1.5} />}
        title="No models registered"
        description="Register your first model to begin training"
      />
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {models.map((model, i) => (
        <motion.div
          key={model.id.toString()}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.04, duration: 0.2 }}
        >
          <ModelCard model={model} />
        </motion.div>
      ))}
    </div>
  );
}
