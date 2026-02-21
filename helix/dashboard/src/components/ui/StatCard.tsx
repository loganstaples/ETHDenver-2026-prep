'use client';

import { type ReactNode, type CSSProperties } from 'react';
import { cn } from '@/lib/utils';
import { Card } from './Card';

interface StatCardProps {
  label: string;
  value: string | number;
  delta?: string;
  deltaDirection?: 'up' | 'down' | 'neutral';
  icon?: ReactNode;
  className?: string;
  style?: CSSProperties;
}

const deltaColors: Record<string, string> = {
  up: 'text-helix-text2',
  down: 'text-helix-muted',
  neutral: 'text-helix-dim',
};

const deltaSymbols: Record<string, string> = {
  up: '\u2191',
  down: '\u2193',
  neutral: '',
};

export function StatCard({
  label,
  value,
  delta,
  deltaDirection = 'neutral',
  icon,
  className,
  style,
}: StatCardProps) {
  return (
    <Card variant="default" className={cn('relative', className)} style={style}>
      {icon && (
        <div className="absolute top-5 right-5 text-helix-dim">
          {icon}
        </div>
      )}
      <div className="space-y-1.5">
        <p className="text-sm text-helix-muted">
          {label}
        </p>
        <p className="text-3xl font-light tracking-tight text-helix-text">
          {value}
        </p>
        {delta && (
          <p className={cn('text-sm', deltaColors[deltaDirection])}>
            {deltaSymbols[deltaDirection]}
            {deltaSymbols[deltaDirection] ? ' ' : ''}
            {delta}
          </p>
        )}
      </div>
    </Card>
  );
}
