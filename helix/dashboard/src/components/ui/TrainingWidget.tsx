'use client';

import { useState, useEffect, useMemo } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, ChevronUp, ArrowUpRight, Zap } from 'lucide-react';
import { useDashboardSessions } from '@/hooks/useDashboardSessions';
import type { LossDataPoint } from '@/hooks/useDashboardSessions';

// ─── Tiny sparkline ──────────────────────────────────────────────────────────

function Sparkline({ data, width = 200, height = 48 }: {
  data: LossDataPoint[];
  width?: number;
  height?: number;
}) {
  const points = useMemo(() => {
    if (data.length < 2) return '';
    // Downsample to ~60 points max for the sparkline
    const maxPts = 60;
    const step = Math.max(1, Math.floor(data.length / maxPts));
    const sampled = data.filter((_, i) => i % step === 0 || i === data.length - 1);

    const losses = sampled.map((d) => d.loss);
    const minL = Math.min(...losses);
    const maxL = Math.max(...losses);
    const range = maxL - minL || 1;

    return sampled
      .map((d, i) => {
        const x = (i / (sampled.length - 1)) * width;
        const y = height - ((d.loss - minL) / range) * (height - 4) - 2;
        return `${x},${y}`;
      })
      .join(' ');
  }, [data, width, height]);

  if (data.length < 2) return null;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="overflow-visible"
    >
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="white" stopOpacity={0.08} />
          <stop offset="100%" stopColor="white" stopOpacity={0} />
        </linearGradient>
      </defs>
      {/* Fill area */}
      <polygon
        points={`0,${height} ${points} ${width},${height}`}
        fill="url(#sparkFill)"
      />
      {/* Line */}
      <polyline
        points={points}
        fill="none"
        stroke="white"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ filter: 'drop-shadow(0 0 3px rgba(255,255,255,0.3))' }}
      />
      {/* End dot */}
      {points && (
        <circle
          cx={width}
          cy={parseFloat(points.split(' ').pop()?.split(',')[1] || '0')}
          r={2.5}
          fill="white"
          style={{ filter: 'drop-shadow(0 0 4px rgba(255,255,255,0.5))' }}
        />
      )}
    </svg>
  );
}

// ─── Elapsed time formatter ──────────────────────────────────────────────────

function formatElapsed(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  if (m === 0) return `${s}s`;
  return `${m}m ${s.toString().padStart(2, '0')}s`;
}

// ─── Main widget ─────────────────────────────────────────────────────────────

export function TrainingWidget() {
  const pathname = usePathname();
  const { activeSession, losses } = useDashboardSessions();
  const [expanded, setExpanded] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  // Hide on train page and dashboard — those already show full training UI
  const hidden = pathname === '/train' || pathname === '/dashboard';

  // Only show for active (non-terminal) sessions
  const isActive =
    activeSession &&
    (activeSession.status === 'starting' || activeSession.status === 'running');

  // Live elapsed timer
  useEffect(() => {
    if (!activeSession || !isActive) return;
    const tick = () => {
      if (activeSession.started_at > 0) {
        setElapsed(Math.floor(Date.now() / 1000 - activeSession.started_at));
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [activeSession, isActive]);

  if (hidden || !isActive) return null;

  const progress =
    activeSession.total_steps > 0
      ? activeSession.current_step / activeSession.total_steps
      : 0;
  const pct = Math.round(progress * 100);

  return (
    <div className="fixed bottom-6 right-6 z-30" style={{ maxWidth: 320 }}>
      <AnimatePresence mode="wait">
        {!expanded ? (
          /* ── Collapsed pill ──────────────────────────────────── */
          <motion.button
            key="pill"
            onClick={() => setExpanded(true)}
            initial={{ opacity: 0, y: 12, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 500, damping: 35 }}
            className="group flex items-center gap-3 rounded-2xl border border-white/[0.08] bg-helix-surface/95 backdrop-blur-xl px-4 py-3 shadow-2xl shadow-black/50 cursor-pointer hover:border-white/[0.14] transition-colors duration-200"
          >
            {/* Pulsing indicator */}
            <div className="relative shrink-0">
              <motion.div
                className="absolute inset-0 rounded-full bg-white/20"
                animate={{ scale: [1, 1.8, 1], opacity: [0.4, 0, 0.4] }}
                transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
              />
              <div className="relative w-8 h-8 rounded-full bg-white/[0.06] border border-white/[0.1] flex items-center justify-center">
                <Cpu size={14} className="text-white/80" />
              </div>
            </div>

            {/* Info */}
            <div className="flex flex-col items-start min-w-0">
              <span className="text-xs font-medium text-white truncate max-w-[160px]">
                {activeSession.model_name || 'Training'}
              </span>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-2xs font-mono text-helix-text2 tabular-nums">
                  {pct}%
                </span>
                {/* Mini progress track */}
                <div className="w-16 h-1 rounded-full bg-white/[0.06] overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-white/60"
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ type: 'spring', stiffness: 120, damping: 20 }}
                  />
                </div>
                <span className="text-2xs font-mono text-helix-muted tabular-nums">
                  {activeSession.current_step}/{activeSession.total_steps}
                </span>
              </div>
            </div>

            {/* Expand chevron */}
            <ChevronUp
              size={14}
              className="text-helix-muted group-hover:text-helix-text2 transition-colors shrink-0 ml-1"
            />
          </motion.button>
        ) : (
          /* ── Expanded card ──────────────────────────────────── */
          <motion.div
            key="card"
            initial={{ opacity: 0, y: 12, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            className="rounded-2xl border border-white/[0.08] bg-helix-surface/95 backdrop-blur-xl shadow-2xl shadow-black/50 overflow-hidden"
          >
            {/* Top edge glow */}
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />

            {/* Header — click to collapse */}
            <button
              onClick={() => setExpanded(false)}
              className="w-full flex items-center justify-between px-4 pt-3.5 pb-2 cursor-pointer"
            >
              <div className="flex items-center gap-2.5">
                <div className="relative">
                  <motion.div
                    className="absolute inset-0 rounded-full bg-white/20"
                    animate={{ scale: [1, 1.6, 1], opacity: [0.3, 0, 0.3] }}
                    transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
                  />
                  <div className="relative w-7 h-7 rounded-full bg-white/[0.06] border border-white/[0.1] flex items-center justify-center">
                    <Cpu size={13} className="text-white/80" />
                  </div>
                </div>
                <div className="text-left">
                  <span className="text-xs font-semibold text-white tracking-tight">
                    {activeSession.model_name || 'Training'}
                  </span>
                  <span className="block text-2xs text-helix-muted font-mono">
                    {activeSession.phase_description?.replace(/Training step \d+\/\d+ — /, '') || activeSession.status}
                  </span>
                </div>
              </div>
              <motion.div
                animate={{ rotate: 180 }}
                transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              >
                <ChevronUp size={14} className="text-helix-muted" />
              </motion.div>
            </button>

            {/* Sparkline */}
            <div className="px-4 pt-1 pb-2">
              <div className="h-12 w-full">
                <Sparkline data={losses} width={288} height={48} />
              </div>
            </div>

            {/* Progress bar */}
            <div className="px-4 pb-3">
              <div className="w-full h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{
                    background: 'linear-gradient(90deg, rgba(255,255,255,0.25), rgba(255,255,255,0.7))',
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ type: 'spring', stiffness: 100, damping: 20 }}
                />
              </div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-3 gap-px bg-white/[0.04] border-t border-white/[0.06]">
              <div className="bg-helix-surface px-3 py-2.5">
                <span className="block text-2xs text-helix-muted uppercase tracking-wider">Step</span>
                <span className="block text-sm font-mono font-semibold text-white tabular-nums tracking-tight mt-0.5">
                  {activeSession.current_step}
                  <span className="text-helix-dim">/{activeSession.total_steps}</span>
                </span>
              </div>
              <div className="bg-helix-surface px-3 py-2.5">
                <span className="block text-2xs text-helix-muted uppercase tracking-wider">Loss</span>
                <span className="block text-sm font-mono font-semibold text-white tabular-nums tracking-tight mt-0.5">
                  {activeSession.current_loss > 0 ? activeSession.current_loss.toFixed(4) : '—'}
                </span>
              </div>
              <div className="bg-helix-surface px-3 py-2.5">
                <span className="block text-2xs text-helix-muted uppercase tracking-wider">Time</span>
                <span className="block text-sm font-mono font-semibold text-white tabular-nums tracking-tight mt-0.5">
                  {formatElapsed(elapsed)}
                </span>
              </div>
            </div>

            {/* Footer actions */}
            <div className="flex items-center justify-between px-4 py-2.5 border-t border-white/[0.04]">
              <div className="flex items-center gap-1.5">
                <Zap size={11} className="text-helix-muted" />
                <span className="text-2xs text-helix-muted font-mono tabular-nums">
                  {activeSession.mac_checks_passed} MACs
                </span>
                {activeSession.checkpoints_submitted > 0 && (
                  <>
                    <span className="text-helix-dim mx-1">&middot;</span>
                    <span className="text-2xs text-helix-muted font-mono tabular-nums">
                      {activeSession.checkpoints_submitted} ckpt
                    </span>
                  </>
                )}
              </div>
              <Link
                href="/train"
                className="flex items-center gap-1 text-2xs text-helix-text2 hover:text-white transition-colors"
              >
                Open
                <ArrowUpRight size={11} />
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
