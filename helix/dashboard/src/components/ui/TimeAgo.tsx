'use client';

import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface TimeAgoProps {
  timestamp: number;
  className?: string;
}

function formatTimeAgo(timestamp: number): string {
  const now = Date.now();
  const diff = Math.max(0, Math.floor((now - timestamp) / 1000));

  if (diff < 60) return `${diff}s ago`;

  const minutes = Math.floor(diff / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function TimeAgo({ timestamp, className }: TimeAgoProps) {
  const [display, setDisplay] = useState(() => formatTimeAgo(timestamp));

  useEffect(() => {
    setDisplay(formatTimeAgo(timestamp));

    const interval = setInterval(() => {
      setDisplay(formatTimeAgo(timestamp));
    }, 10_000);

    return () => clearInterval(interval);
  }, [timestamp]);

  return (
    <time
      dateTime={new Date(timestamp).toISOString()}
      className={cn('text-2xs text-helix-muted font-mono', className)}
    >
      {display}
    </time>
  );
}
