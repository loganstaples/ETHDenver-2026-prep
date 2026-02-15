'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

type ProgressSize = 'sm' | 'md';

interface ProgressBarProps {
  value: number;
  className?: string;
  size?: ProgressSize;
}

const sizeStyles: Record<ProgressSize, string> = {
  sm: 'h-1',
  md: 'h-1.5',
};

export function ProgressBar({
  value,
  className,
  size = 'sm',
}: ProgressBarProps) {
  const clampedValue = Math.min(100, Math.max(0, value));

  return (
    <div
      className={cn(
        'w-full overflow-hidden rounded-full bg-helix-border',
        sizeStyles[size],
        className,
      )}
    >
      <motion.div
        className="h-full rounded-full bg-white"
        initial={{ width: 0 }}
        animate={{ width: `${clampedValue}%` }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      />
    </div>
  );
}
