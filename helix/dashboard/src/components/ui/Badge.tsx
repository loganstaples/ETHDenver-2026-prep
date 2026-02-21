'use client';

import { type ReactNode } from 'react';
import { cn } from '@/lib/utils';

type BadgeVariant = 'default' | 'outline' | 'pulse';

interface BadgeProps {
  variant?: BadgeVariant;
  className?: string;
  children: ReactNode;
}

const baseStyles = 'inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg';

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-white/[0.06] text-helix-muted',
  outline: 'border border-helix-border bg-transparent text-helix-muted',
  pulse: 'bg-white/[0.06] text-helix-muted',
};

export function Badge({
  variant = 'default',
  className,
  children,
}: BadgeProps) {
  return (
    <span className={cn(baseStyles, variantStyles[variant], className)}>
      {variant === 'pulse' && (
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-white" />
        </span>
      )}
      {children}
    </span>
  );
}
