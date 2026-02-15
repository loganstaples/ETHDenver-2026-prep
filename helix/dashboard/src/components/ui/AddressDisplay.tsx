'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AddressDisplayProps {
  address: string;
  truncate?: boolean;
  className?: string;
}

export function AddressDisplay({
  address,
  truncate = true,
  className,
}: AddressDisplayProps) {
  const [copied, setCopied] = useState(false);

  const displayAddress = truncate
    ? `${address.slice(0, 6)}...${address.slice(-4)}`
    : address;

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(address);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API may not be available in all contexts
    }
  }, [address]);

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={cn(
        'relative inline-flex items-center gap-1.5 font-mono text-sm text-helix-text2',
        'hover:text-helix-text cursor-pointer transition-colors',
        className,
      )}
    >
      <span>{displayAddress}</span>
      <Copy size={12} className="shrink-0 opacity-50" />
      <AnimatePresence>
        {copied && (
          <motion.span
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute -top-6 left-1/2 -translate-x-1/2 text-2xs text-helix-text whitespace-nowrap bg-helix-surface2 border border-helix-border px-1.5 py-0.5 rounded"
          >
            Copied
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}
