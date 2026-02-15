'use client';

import { type ReactNode } from 'react';
import { motion, type HTMLMotionProps } from 'framer-motion';
import { cn } from '@/lib/utils';

type CardVariant = 'default' | 'glass' | 'ghost';

interface CardProps extends Omit<HTMLMotionProps<'div'>, 'children'> {
  variant?: CardVariant;
  className?: string;
  children: ReactNode;
  hover?: boolean;
}

const variantStyles: Record<CardVariant, string> = {
  default: 'bg-helix-surface border border-helix-border rounded-lg p-5',
  glass: 'bg-white/[0.02] backdrop-blur-md border border-white/[0.06] rounded-lg p-5 relative overflow-hidden',
  ghost: 'rounded-lg p-5',
};

export function Card({
  variant = 'default',
  className,
  children,
  hover = false,
  ...props
}: CardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(
        variantStyles[variant],
        hover && 'hover:border-helix-border2 transition-colors',
        className,
      )}
      {...props}
    >
      {variant === 'glass' && (
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
      )}
      {children}
    </motion.div>
  );
}
