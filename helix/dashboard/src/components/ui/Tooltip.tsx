'use client';

import { type ReactNode, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';

interface TooltipProps {
  content: string;
  children: ReactNode;
  side?: 'top' | 'bottom';
  className?: string;
}

export function Tooltip({
  content,
  children,
  side = 'top',
  className,
}: TooltipProps) {
  const [visible, setVisible] = useState(false);

  return (
    <div
      className={cn('relative inline-flex', className)}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      <AnimatePresence>
        {visible && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            className={cn(
              'absolute left-1/2 -translate-x-1/2 z-50 pointer-events-none',
              'bg-helix-surface2 border border-helix-border text-2xs text-helix-text2',
              'px-2 py-1 rounded shadow-lg whitespace-nowrap',
              side === 'top' && 'bottom-full mb-1.5',
              side === 'bottom' && 'top-full mt-1.5',
            )}
          >
            {content}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
