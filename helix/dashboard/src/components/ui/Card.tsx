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
  default: 'bg-helix-surface/50 border border-helix-border rounded-2xl p-6',
  glass: 'bg-white/[0.03] backdrop-blur-md border border-white/[0.07] rounded-2xl p-6 relative overflow-hidden',
  ghost: 'rounded-2xl p-6',
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
        hover && 'hover:border-white/[0.15] hover:bg-white/[0.04] transition-all duration-300',
        className,
      )}
      {...props}
    >
      {variant === 'glass' && (
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/25 to-transparent" />
      )}
      {children}
    </motion.div>
  );
}
