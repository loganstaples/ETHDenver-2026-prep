'use client';

import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  width?: string;
  height?: string;
}

export function Skeleton({
  className,
  width = '100%',
  height = '1rem',
}: SkeletonProps) {
  return (
    <div
      className={cn('animate-pulse rounded bg-helix-border', className)}
      style={{ width, height }}
    />
  );
}
