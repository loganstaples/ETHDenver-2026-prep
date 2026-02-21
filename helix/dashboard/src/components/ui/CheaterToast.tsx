'use client';

import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform, animate } from 'framer-motion';
import { ShieldAlert, ExternalLink, ShieldCheck, Zap, RefreshCw, CheckCircle2 } from 'lucide-react';
import type { CheaterInfo } from '@/hooks/useMpcTraining';

const ADI_EXPLORER = 'https://explorer.ab.testnet.adifoundation.ai';

interface CheaterToastProps {
  cheater: CheaterInfo | null;
}

// Progression phases for the cinematic sequence
type Phase = 'detected' | 'identified' | 'slashed' | 'recovered' | 'dismissed';

/**
 * Cheater detection widget. When a cheater is caught:
 * 1. Widget slides in from the right with the detection sequence
 * 2. Steps light up as events arrive: Detected → Identified → Slashed → Recovered
 * 3. On recovery, widget slides out after a delay
 * The progress bar on the training page turns red independently.
 */
export function CheaterToast({ cheater }: CheaterToastProps) {
  const [phase, setPhase] = useState<Phase>('dismissed');
  const [showWidget, setShowWidget] = useState(false);
  const [currentCheater, setCurrentCheater] = useState<CheaterInfo | null>(null);
  const prevCheaterRef = useRef<CheaterInfo | null>(null);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Detect new cheater (transition from null → non-null)
  useEffect(() => {
    if (cheater && !prevCheaterRef.current) {
      setCurrentCheater(cheater);
      setPhase('detected');
      setShowWidget(true);

      // Move to "identified" after a brief beat (pairwise MAC identification is instant)
      setTimeout(() => setPhase('identified'), 800);
    }
    prevCheaterRef.current = cheater;
  }, [cheater]);

  // Track slash and recovery updates
  useEffect(() => {
    if (!cheater) return;
    setCurrentCheater(cheater);

    if (cheater.slashed && phase === 'identified') {
      setPhase('slashed');
    }
    if (cheater.recovered && (phase === 'slashed' || phase === 'identified')) {
      setPhase('recovered');
      // Auto-dismiss after seeing the success state
      if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = setTimeout(() => {
        setShowWidget(false);
        setTimeout(() => setPhase('dismissed'), 500);
      }, 6000);
    }
  }, [cheater, phase]);

  // If no chain is running, slashing won't fire — go straight to recovered from identified
  useEffect(() => {
    if (cheater?.recovered && phase === 'identified') {
      setPhase('recovered');
      if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = setTimeout(() => {
        setShowWidget(false);
        setTimeout(() => setPhase('dismissed'), 500);
      }, 6000);
    }
  }, [cheater?.recovered, phase]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    };
  }, []);

  // Reset when cheater clears
  useEffect(() => {
    if (!cheater && currentCheater) {
      setShowWidget(false);
      setTimeout(() => {
        setCurrentCheater(null);
        setPhase('dismissed');
      }, 500);
    }
  }, [cheater, currentCheater]);

  const handleDismiss = () => {
    if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    setShowWidget(false);
    setTimeout(() => setPhase('dismissed'), 500);
  };

  const explorerUrl = currentCheater?.slash_tx_hash
    ? `${ADI_EXPLORER}/tx/${currentCheater.slash_tx_hash}`
    : null;

  const steps: { key: Phase; icon: typeof ShieldAlert; label: string; sublabel?: string }[] = [
    {
      key: 'detected',
      icon: ShieldAlert,
      label: 'Corruption Detected',
      sublabel: currentCheater
        ? `Worker ${currentCheater.party_index} · Step ${currentCheater.step}`
        : undefined,
    },
    {
      key: 'identified',
      icon: Zap,
      label: 'Cheater Identified',
      sublabel: 'Pairwise SPDZ MAC verification',
    },
    {
      key: 'slashed',
      icon: ShieldCheck,
      label: 'Stake Slashed',
      sublabel: explorerUrl ? undefined : 'On-chain punishment',
    },
    {
      key: 'recovered',
      icon: RefreshCw,
      label: 'Recovered Safely',
      sublabel: currentCheater?.recovery_workers
        ? `${currentCheater.recovery_workers} honest workers from checkpoint`
        : 'Rolled back to last verified checkpoint',
    },
  ];

  const phaseOrder: Phase[] = ['detected', 'identified', 'slashed', 'recovered'];
  const currentPhaseIdx = phaseOrder.indexOf(phase);

  return (
    <>
      <AnimatePresence>
        {showWidget && currentCheater && phase !== 'dismissed' && (
          <motion.div
            key="widget"
            initial={{ opacity: 0, x: 80, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 80, scale: 0.95 }}
            transition={{
              type: 'spring',
              damping: 28,
              stiffness: 260,
              mass: 0.8,
            }}
            className="fixed top-6 right-6 z-[9999] w-[340px] pointer-events-auto"
          >
            <div className="relative overflow-hidden rounded-2xl border border-red-500/30 shadow-2xl shadow-red-950/40">
              {/* Background with animated noise */}
              <div
                className="absolute inset-0"
                style={{
                  background: 'linear-gradient(160deg, #1a0808 0%, #0d0404 50%, #0a0202 100%)',
                }}
              />

              {/* Top scanline accent */}
              <motion.div
                className="absolute top-0 left-0 right-0 h-px"
                style={{
                  background: 'linear-gradient(90deg, transparent, #ef4444, #f97316, #ef4444, transparent)',
                }}
                animate={{ opacity: [0.4, 0.8, 0.4] }}
                transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
              />

              <div className="relative p-5">
                {/* Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="relative">
                      <motion.div
                        className="absolute -inset-1 rounded-lg bg-red-500/20"
                        animate={phase !== 'recovered' ? {
                          scale: [1, 1.3, 1],
                          opacity: [0.3, 0, 0.3],
                        } : { scale: 1, opacity: 0 }}
                        transition={{ duration: 1.5, repeat: phase !== 'recovered' ? Infinity : 0 }}
                      />
                      <div className={`relative w-8 h-8 rounded-lg flex items-center justify-center transition-colors duration-500 ${
                        phase === 'recovered'
                          ? 'bg-emerald-500/15 border border-emerald-500/30'
                          : 'bg-red-500/15 border border-red-500/30'
                      }`}>
                        {phase === 'recovered' ? (
                          <CheckCircle2 size={16} className="text-emerald-400" />
                        ) : (
                          <ShieldAlert size={16} className="text-red-400" />
                        )}
                      </div>
                    </div>
                    <div>
                      <h4 className={`text-[13px] font-semibold tracking-tight transition-colors duration-500 ${
                        phase === 'recovered' ? 'text-emerald-300' : 'text-red-300'
                      }`}>
                        {phase === 'recovered' ? 'Protocol Secured' : 'Byzantine Fault'}
                      </h4>
                      <p className="text-xs text-zinc-500 font-mono">
                        SPDZ-MAC · MPC TRAINING
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={handleDismiss}
                    className="w-6 h-6 rounded-md flex items-center justify-center text-zinc-600 hover:text-zinc-400 hover:bg-white/5 transition-colors"
                  >
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                      <path d="M1 1l8 8M9 1l-8 8" />
                    </svg>
                  </button>
                </div>

                {/* Step progression */}
                <div className="space-y-0.5">
                  {steps.map((step, idx) => {
                    const isActive = phaseOrder.indexOf(step.key) <= currentPhaseIdx;
                    const isCurrent = step.key === phase;
                    // Skip slashed step if no chain (slashed never fires)
                    const shouldSkip = step.key === 'slashed' && !currentCheater.slashed && phase === 'recovered';

                    if (shouldSkip) return null;

                    return (
                      <motion.div
                        key={step.key}
                        initial={{ opacity: 0, x: 12 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.12, duration: 0.3 }}
                        className="relative"
                      >
                        <div className={`flex items-start gap-3 py-2.5 px-3 rounded-xl transition-all duration-500 ${
                          isCurrent && phase !== 'recovered'
                            ? 'bg-red-500/[0.06]'
                            : isCurrent && phase === 'recovered'
                              ? 'bg-emerald-500/[0.06]'
                              : ''
                        }`}>
                          {/* Timeline dot/icon */}
                          <div className="relative shrink-0 mt-0.5">
                            {isActive ? (
                              <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ type: 'spring', damping: 15, stiffness: 300 }}
                                className={`w-5 h-5 rounded-md flex items-center justify-center ${
                                  step.key === 'recovered'
                                    ? 'bg-emerald-500/20'
                                    : 'bg-red-500/20'
                                }`}
                              >
                                <step.icon size={11} className={
                                  step.key === 'recovered' ? 'text-emerald-400' : 'text-red-400'
                                } />
                              </motion.div>
                            ) : (
                              <div className="w-5 h-5 rounded-md border border-zinc-800 flex items-center justify-center">
                                <div className="w-1.5 h-1.5 rounded-full bg-zinc-700" />
                              </div>
                            )}
                            {/* Connecting line */}
                            {idx < steps.length - 1 && !(step.key === 'slashed' && !currentCheater.slashed && phase === 'recovered') && (
                              <div className={`absolute left-[9px] top-[22px] w-px h-3 transition-colors duration-500 ${
                                isActive ? (step.key === 'recovered' ? 'bg-emerald-500/20' : 'bg-red-500/20') : 'bg-zinc-800'
                              }`} />
                            )}
                          </div>

                          {/* Label */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className={`text-[12px] font-medium transition-colors duration-500 ${
                                isActive
                                  ? step.key === 'recovered' ? 'text-emerald-300' : 'text-red-300'
                                  : 'text-zinc-600'
                              }`}>
                                {step.label}
                              </span>
                              {isCurrent && phase !== 'recovered' && (
                                <motion.div
                                  className="w-1 h-1 rounded-full bg-red-400"
                                  animate={{ opacity: [1, 0.3, 1] }}
                                  transition={{ duration: 1, repeat: Infinity }}
                                />
                              )}
                            </div>
                            {step.sublabel && isActive && (
                              <motion.p
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                transition={{ duration: 0.2 }}
                                className={`text-xs mt-0.5 font-mono ${
                                  isActive ? 'text-zinc-500' : 'text-zinc-700'
                                }`}
                              >
                                {step.sublabel}
                              </motion.p>
                            )}
                            {/* Explorer link for slashed step */}
                            {step.key === 'slashed' && isActive && explorerUrl && (
                              <motion.a
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                href={explorerUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 mt-1 text-xs text-orange-400/70 hover:text-orange-300 transition-colors font-mono"
                              >
                                {currentCheater.slash_tx_hash!.slice(0, 10)}...{currentCheater.slash_tx_hash!.slice(-6)}
                                <ExternalLink size={9} />
                              </motion.a>
                            )}
                          </div>

                          {/* Status indicator */}
                          <div className="shrink-0 mt-1">
                            {isActive ? (
                              <motion.div
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ type: 'spring', damping: 12 }}
                              >
                                <CheckCircle2 size={12} className={
                                  step.key === 'recovered' ? 'text-emerald-400/60' : 'text-red-400/40'
                                } />
                              </motion.div>
                            ) : (
                              <div className="w-3 h-3" />
                            )}
                          </div>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>

                {/* Bottom accent for recovered state */}
                <AnimatePresence>
                  {phase === 'recovered' && (
                    <motion.div
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="mt-3 pt-3 border-t border-emerald-500/10"
                    >
                      <p className="text-xs text-emerald-400/40 text-center font-mono tracking-wider uppercase">
                        Training continues · Protocol integrity verified
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Bottom progress bar — time until auto-dismiss */}
              {phase === 'recovered' && (
                <motion.div
                  className="h-px bg-gradient-to-r from-emerald-500/40 via-emerald-400/60 to-emerald-500/40"
                  initial={{ width: '100%' }}
                  animate={{ width: '0%' }}
                  transition={{ duration: 6, ease: 'linear' }}
                />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
