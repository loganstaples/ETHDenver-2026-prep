'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, X, ShieldAlert, ExternalLink, CheckCircle2 } from 'lucide-react';
import type { CheaterInfo } from '@/hooks/useMpcTraining';

const ADI_EXPLORER = 'https://explorer.ab.testnet.adifoundation.ai';

interface CheaterToastProps {
  /** Detected cheater info — null means no cheater. Toast appears on transition to non-null. */
  cheater: CheaterInfo | null;
}

/**
 * Animated toast notification that appears when a cheater is detected during
 * MPC training. Shows progression: detected → slashed on-chain → recovered.
 * Slides in from the top-right and auto-dismisses after 20s.
 */
export function CheaterToast({ cheater }: CheaterToastProps) {
  const [visible, setVisible] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [currentCheater, setCurrentCheater] = useState<CheaterInfo | null>(null);

  // Show toast when cheater transitions from null to non-null
  useEffect(() => {
    if (cheater && !dismissed) {
      setCurrentCheater(cheater);
      setVisible(true);

      // Auto-dismiss after 20s (longer to show full progression)
      const timer = setTimeout(() => setVisible(false), 20_000);
      return () => clearTimeout(timer);
    }
  }, [cheater, dismissed]);

  // Keep currentCheater updated as slash/recovery info arrives
  useEffect(() => {
    if (cheater) {
      setCurrentCheater(cheater);
    }
  }, [cheater]);

  // Reset dismissed state when cheater changes (new detection)
  useEffect(() => {
    if (!cheater) {
      setDismissed(false);
    }
  }, [cheater]);

  const handleDismiss = () => {
    setVisible(false);
    setDismissed(true);
  };

  const explorerUrl = currentCheater?.slash_tx_hash
    ? `${ADI_EXPLORER}/tx/${currentCheater.slash_tx_hash}`
    : null;

  return (
    <AnimatePresence>
      {visible && currentCheater && (
        <motion.div
          initial={{ opacity: 0, y: -20, x: 20 }}
          animate={{ opacity: 1, y: 0, x: 0 }}
          exit={{ opacity: 0, y: -20, x: 20 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="fixed top-6 right-6 z-[9999] max-w-sm w-full pointer-events-auto"
        >
          <div className="relative overflow-hidden rounded-2xl border border-red-500/40 bg-[#1a0a0a]/95 backdrop-blur-xl shadow-2xl shadow-red-500/20">
            {/* Animated accent bar */}
            <motion.div
              className="absolute top-0 left-0 h-1 bg-gradient-to-r from-red-500 via-orange-500 to-red-500"
              initial={{ width: '100%' }}
              animate={{ width: '0%' }}
              transition={{ duration: 20, ease: 'linear' }}
            />

            <div className="p-4">
              <div className="flex items-start gap-3">
                {/* Pulsing icon */}
                <div className="relative shrink-0 mt-0.5">
                  <motion.div
                    className="absolute inset-0 rounded-full bg-red-500/30"
                    animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                  <div className="relative w-10 h-10 rounded-full bg-red-500/20 border border-red-500/40 flex items-center justify-center">
                    <ShieldAlert size={20} className="text-red-400" />
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-semibold text-red-300">
                      Byzantine Fault Detected
                    </h4>
                    <AlertTriangle size={14} className="text-red-400 animate-pulse" />
                  </div>
                  <p className="mt-1 text-xs text-red-200/70 leading-relaxed">
                    <span className="font-mono font-medium text-red-300">
                      Worker {currentCheater.party_index}
                    </span>{' '}
                    submitted corrupted weight shares at{' '}
                    <span className="font-mono font-medium text-red-300">
                      step {currentCheater.step}
                    </span>
                    . SPDZ MAC verification caught the tampering.
                  </p>

                  {/* Progression steps */}
                  <div className="mt-2 space-y-1">
                    <StepIndicator done label="Cheater identified via pairwise MAC verification" />
                    <StepIndicator
                      done={currentCheater.slashed}
                      label={
                        currentCheater.slashed
                          ? 'Stake slashed on-chain'
                          : 'Submitting blame report on-chain...'
                      }
                    />
                    {currentCheater.slashed && explorerUrl && (
                      <a
                        href={explorerUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-5 flex items-center gap-1 text-[10px] text-orange-400/80 hover:text-orange-300 transition-colors font-mono"
                      >
                        {currentCheater.slash_tx_hash!.slice(0, 10)}...{currentCheater.slash_tx_hash!.slice(-6)}
                        <ExternalLink size={10} />
                      </a>
                    )}
                    <StepIndicator
                      done={currentCheater.recovered}
                      label={
                        currentCheater.recovered
                          ? `Training resumed with ${currentCheater.recovery_workers} workers`
                          : currentCheater.slashed
                            ? 'Redistributing shares...'
                            : 'Awaiting slashing'
                      }
                    />
                  </div>
                </div>

                {/* Dismiss */}
                <button
                  onClick={handleDismiss}
                  className="shrink-0 p-1 rounded-lg hover:bg-red-500/20 transition-colors"
                >
                  <X size={14} className="text-red-400/60" />
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function StepIndicator({ done, label }: { done: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      {done ? (
        <CheckCircle2 size={12} className="text-green-400 shrink-0" />
      ) : (
        <motion.div
          className="w-3 h-3 rounded-full border border-red-400/60 shrink-0"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
      <span className={`text-[11px] ${done ? 'text-green-300/70' : 'text-red-200/50'}`}>
        {label}
      </span>
    </div>
  );
}
