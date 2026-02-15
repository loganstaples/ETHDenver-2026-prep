'use client';

import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/Badge';
import { TimeAgo } from '@/components/ui/TimeAgo';
import { Skeleton } from '@/components/ui/Skeleton';
import { useProofTimeline } from '@/hooks/useProofs';
import { cn } from '@/lib/utils';

interface ProofTimelineProps {
  modelId: bigint;
}

const EVENT_TYPE_STYLES: Record<string, string> = {
  started: 'bg-white/[0.04] text-helix-muted',
  witness_generated: 'bg-white/[0.06] text-helix-text2',
  setup_complete: 'bg-white/[0.06] text-helix-text2',
  proving: 'bg-white/[0.06] text-helix-muted',
  proof_generated: 'bg-white/[0.08] text-helix-text2',
  submitted: 'bg-white/[0.08] text-white',
  verified: 'bg-white/[0.10] text-white',
  failed: 'bg-white/[0.04] text-helix-dim',
  challenged: 'bg-white/[0.06] text-helix-text2',
};

const EVENT_TYPE_LABELS: Record<string, string> = {
  started: 'Started',
  witness_generated: 'Witness',
  setup_complete: 'Setup',
  proving: 'Proving',
  proof_generated: 'Generated',
  submitted: 'Submitted',
  verified: 'Verified',
  failed: 'Failed',
  challenged: 'Challenged',
};

function truncateProofId(id: string): string {
  if (id.length <= 16) return id;
  return `${id.slice(0, 10)}...${id.slice(-4)}`;
}

function formatDuration(ms: number | undefined): string | null {
  if (ms === undefined || ms <= 0) return null;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms / 60000)}m`;
}

const staggerContainer = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.04,
    },
  },
};

const staggerItem = {
  hidden: { opacity: 0, x: -8 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.2 },
  },
};

export function ProofTimeline({ modelId }: ProofTimelineProps) {
  const { groupedTimeline, isLoading } = useProofTimeline(modelId);

  const formattedGroups = useMemo(
    () =>
      groupedTimeline.map((group) => ({
        ...group,
        formattedDate: formatGroupDate(group.date),
      })),
    [groupedTimeline],
  );

  if (isLoading) {
    return (
      <div className="space-y-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="space-y-3">
            <Skeleton width="6rem" height="0.75rem" />
            <Skeleton height="4rem" />
            <Skeleton height="4rem" />
          </div>
        ))}
      </div>
    );
  }

  if (formattedGroups.length === 0) {
    return (
      <p className="text-2xs text-helix-dim py-8 text-center">
        No timeline events yet
      </p>
    );
  }

  return (
    <div className="space-y-8">
      {formattedGroups.map((group) => (
        <div key={group.date} className="space-y-3">
          {/* Date Header */}
          <p className="text-2xs text-helix-muted uppercase tracking-wider font-mono">
            {group.formattedDate}
          </p>

          {/* Events */}
          <motion.div
            className="relative pl-6"
            variants={staggerContainer}
            initial="hidden"
            animate="visible"
          >
            {/* Connecting line */}
            <div className="absolute left-[5px] top-2 bottom-2 w-px bg-helix-border" />

            {group.events.map((event) => {
              const isVerified = event.type === 'verified';
              const duration = formatDuration(event.duration);

              return (
                <motion.div
                  key={event.id}
                  variants={staggerItem}
                  className="relative pb-4 last:pb-0"
                >
                  {/* Dot */}
                  <div
                    className={cn(
                      'absolute -left-6 top-1.5 h-[10px] w-[10px] rounded-full border-2',
                      isVerified
                        ? 'bg-white border-white'
                        : 'bg-transparent border-helix-dim',
                    )}
                  />

                  {/* Event Content */}
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2 min-w-0">
                      <Badge className={cn(EVENT_TYPE_STYLES[event.type])}>
                        {EVENT_TYPE_LABELS[event.type] ?? event.type}
                      </Badge>
                      <span className="font-mono text-2xs text-helix-text2 truncate">
                        {truncateProofId(event.proofId)}
                      </span>
                      {duration && (
                        <span className="text-2xs text-helix-dim font-mono shrink-0">
                          {duration}
                        </span>
                      )}
                    </div>
                    <TimeAgo timestamp={event.timestamp} className="shrink-0" />
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      ))}
    </div>
  );
}

function formatGroupDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffDays = Math.floor(
    (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24),
  );

  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';

  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
  });
}
