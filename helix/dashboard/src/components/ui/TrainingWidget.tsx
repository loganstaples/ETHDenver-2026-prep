'use client';

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { motion, AnimatePresence, useDragControls } from 'framer-motion';
import {
  Cpu,
  CheckCircle,
  XCircle,
  ArrowUpRight,
  Minimize2,
  Maximize2,
  GripVertical,
  Shield,
} from 'lucide-react';
import { useDashboardSessions } from '@/hooks/useDashboardSessions';
import type { LossDataPoint } from '@/hooks/useDashboardSessions';

// ─── Tiny sparkline ──────────────────────────────────────────────────────────

function Sparkline({ data, width = 200, height = 40 }: {
  data: LossDataPoint[];
  width?: number;
  height?: number;
}) {
  const points = useMemo(() => {
    if (data.length < 2) return '';
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
        <linearGradient id="wSparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="white" stopOpacity={0.06} />
          <stop offset="100%" stopColor="white" stopOpacity={0} />
        </linearGradient>
      </defs>
      <polygon
        points={`0,${height} ${points} ${width},${height}`}
        fill="url(#wSparkFill)"
      />
      <polyline
        points={points}
        fill="none"
        stroke="rgba(255,255,255,0.5)"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {points && (
        <circle
          cx={width}
          cy={parseFloat(points.split(' ').pop()?.split(',')[1] || '0')}
          r={2}
          fill="white"
          style={{ filter: 'drop-shadow(0 0 3px rgba(255,255,255,0.4))' }}
        />
      )}
    </svg>
  );
}

// ─── Mini horseshoe phase ring ───────────────────────────────────────────────

function MiniPhaseRing({ phase, totalPhases, size = 48 }: {
  phase: number;
  totalPhases: number;
  size?: number;
}) {
  const strokeWidth = 3;
  const radius = (size - strokeWidth) / 2;
  const center = size / 2;
  const arcDeg = 270;
  const circumference = 2 * Math.PI * radius;
  const arcLength = (arcDeg / 360) * circumference;
  const progress = Math.min(100, (phase / totalPhases) * 100);
  const filledLength = (progress / 100) * arcLength;
  const rotation = 135;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <defs>
        <filter id={`mini-glow-${size}`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <circle
        cx={center} cy={center} r={radius}
        fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth}
        strokeDasharray={`${arcLength} ${circumference}`}
        strokeLinecap="round"
        transform={`rotate(${rotation} ${center} ${center})`}
      />
      {progress > 0 && (
        <circle
          cx={center} cy={center} r={radius}
          fill="none" stroke="white" strokeWidth={strokeWidth}
          strokeDasharray={`${filledLength} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(${rotation} ${center} ${center})`}
          filter={`url(#mini-glow-${size})`}
          style={{ transition: 'stroke-dasharray 0.6s ease-out' }}
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

// ─── Constants ───────────────────────────────────────────────────────────────

const TOTAL_PHASES = 13;

const PHASE_DESCRIPTIONS: Record<number, string> = {
  1: 'Loading data',
  2: 'Init weights',
  3: 'Deploy contracts',
  4: 'Deploy coordinator',
  5: 'Register job',
  6: 'Staking',
  7: 'Preparing workers',
  8: 'MPC training',
  9: 'Checkpoints',
  10: 'Verifying MACs',
  11: 'Completing',
  12: 'Evaluating',
  13: 'Complete',
};

// ─── Terminal state type ─────────────────────────────────────────────────────

type TerminalState = {
  kind: 'complete';
  modelName: string;
  accuracy: number | null;
  steps: number;
  totalSteps: number;
  elapsed: number;
} | {
  kind: 'failed';
  modelName: string;
  steps: number;
  totalSteps: number;
  elapsed: number;
} | {
  kind: 'paused';
  modelName: string;
  steps: number;
  totalSteps: number;
  elapsed: number;
} | {
  kind: 'stopped';
  modelName: string;
  steps: number;
  totalSteps: number;
  elapsed: number;
};

const DISMISS_DELAY = 6000;

type WidgetMode = 'minimized' | 'normal' | 'expanded';

// ─── Main widget ─────────────────────────────────────────────────────────────

export function TrainingWidget() {
  const pathname = usePathname();
  const router = useRouter();
  const { activeSession, losses } = useDashboardSessions();
  const [elapsed, setElapsed] = useState(0);
  const [terminal, setTerminal] = useState<TerminalState | null>(null);
  const dismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevStatusRef = useRef<string | null>(null);
  const dragControls = useDragControls();
  const constraintsRef = useRef<HTMLDivElement>(null);

  // Widget state
  const [mode, setMode] = useState<WidgetMode>('normal');

  const isActive =
    activeSession &&
    (activeSession.status === 'starting' || activeSession.status === 'running');

  // Detect transitions to terminal states
  useEffect(() => {
    if (!activeSession) return;
    const prev = prevStatusRef.current;
    const curr = activeSession.status;
    const terminalStatuses = ['complete', 'failed', 'paused', 'stopped'] as const;

    if (prev && prev !== curr && (terminalStatuses as readonly string[]).includes(curr)) {
      let state: TerminalState;
      const baseName = activeSession.model_name || 'Training';
      const baseSteps = activeSession.current_step;
      const baseTotalSteps = activeSession.total_steps;

      if (curr === 'complete') {
        state = { kind: 'complete', modelName: baseName, accuracy: activeSession.accuracy, steps: baseSteps, totalSteps: baseTotalSteps, elapsed };
      } else if (curr === 'paused') {
        state = { kind: 'paused', modelName: baseName, steps: baseSteps, totalSteps: baseTotalSteps, elapsed };
      } else if (curr === 'stopped') {
        state = { kind: 'stopped', modelName: baseName, steps: baseSteps, totalSteps: baseTotalSteps, elapsed };
      } else {
        state = { kind: 'failed', modelName: baseName, steps: baseSteps, totalSteps: baseTotalSteps, elapsed };
      }

      setTerminal(state);
      if (dismissTimer.current) clearTimeout(dismissTimer.current);
      dismissTimer.current = setTimeout(() => setTerminal(null), DISMISS_DELAY);
    }

    prevStatusRef.current = curr;
  }, [activeSession, elapsed]);

  // Cleanup timer
  useEffect(() => {
    return () => {
      if (dismissTimer.current) clearTimeout(dismissTimer.current);
    };
  }, []);

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

  const handleGoToTrain = useCallback(() => {
    if (pathname !== '/train') {
      router.push('/train');
    }
  }, [pathname, router]);

  // Nothing to show
  if (!isActive && !terminal) return null;

  const progress =
    isActive && activeSession.total_steps > 0
      ? activeSession.current_step / activeSession.total_steps
      : 0;
  const pct = Math.round(progress * 100);

  const phaseLabel = isActive
    ? PHASE_DESCRIPTIONS[activeSession.phase] || 'Processing...'
    : '';

  // ─── glass styles ────────────────────────────────────────────────────────
  const glass = 'bg-[#18181b]/85 backdrop-blur-2xl border border-white/[0.08] shadow-2xl shadow-black/70';

  // ─── Terminal state color helpers ────────────────────────────────────────
  const terminalColor = terminal?.kind === 'complete' ? 'green'
    : terminal?.kind === 'failed' ? 'red'
    : terminal?.kind === 'paused' ? 'yellow'
    : 'white';
  const terminalEdge = `from-transparent via-${terminalColor}-400/30 to-transparent`;
  const terminalIcon = terminal?.kind === 'complete' ? <CheckCircle size={14} className="text-green-400" />
    : terminal?.kind === 'failed' ? <XCircle size={14} className="text-red-400" />
    : terminal?.kind === 'paused' ? <span className="text-yellow-400 text-xs">&#9208;</span>
    : <span className="text-white/50 text-xs">&#9209;</span>;
  const terminalTitle = terminal?.kind === 'complete' ? 'Training Complete'
    : terminal?.kind === 'failed' ? 'Training Failed'
    : terminal?.kind === 'paused' ? 'Training Paused'
    : 'Training Stopped';

  return (
    <>
      {/* Drag constraints — full viewport */}
      <div ref={constraintsRef} className="fixed inset-0 pointer-events-none z-40" />

      <motion.div
        drag
        dragControls={dragControls}
        dragMomentum={false}
        dragConstraints={constraintsRef}
        dragElastic={0.05}
        className="fixed bottom-5 right-5 z-50"
      >
        <AnimatePresence mode="wait">

          {/* ═══════════ TERMINAL STATES ═══════════ */}
          {terminal && (
            <motion.div
              key={`terminal-${terminal.kind}`}
              initial={{ opacity: 0, y: 16, scale: 0.92 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.95 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              className={`rounded-2xl ${glass} overflow-hidden`}
              style={{ width: 300 }}
            >
              <div className={`absolute inset-x-0 top-0 h-px bg-gradient-to-r ${terminalEdge}`} />

              <div className="px-4 pt-4 pb-3 flex items-start gap-3">
                <div className="w-8 h-8 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center shrink-0 mt-0.5">
                  {terminalIcon}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold text-white tracking-tight truncate">
                    {terminalTitle}
                  </p>
                  <p className="text-xs text-white/40 mt-0.5 truncate">
                    {terminal.modelName}
                  </p>
                </div>
              </div>

              <div className="px-4 pb-4 flex items-center gap-4">
                {terminal.kind === 'complete' && terminal.accuracy != null && (
                  <div>
                    <span className="block text-xs text-white/40">Accuracy</span>
                    <span className="block text-sm font-mono font-semibold text-green-400 tabular-nums mt-0.5">
                      {(terminal.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                <div>
                  <span className="block text-xs text-white/40">Steps</span>
                  <span className="block text-sm font-mono font-semibold text-white tabular-nums mt-0.5">
                    {terminal.steps}
                  </span>
                </div>
                <div>
                  <span className="block text-xs text-white/40">Time</span>
                  <span className="block text-sm font-mono font-semibold text-white tabular-nums mt-0.5">
                    {formatElapsed(terminal.elapsed)}
                  </span>
                </div>
              </div>

              <div className="h-0.5 bg-white/[0.03]">
                <motion.div
                  className={`h-full ${
                    terminal.kind === 'complete' ? 'bg-green-400/30'
                    : terminal.kind === 'failed' ? 'bg-red-400/30'
                    : terminal.kind === 'paused' ? 'bg-yellow-400/30'
                    : 'bg-white/10'
                  }`}
                  initial={{ width: '100%' }}
                  animate={{ width: '0%' }}
                  transition={{ duration: DISMISS_DELAY / 1000, ease: 'linear' }}
                />
              </div>
            </motion.div>
          )}

          {/* ═══════════ ACTIVE: MINIMIZED ═══════════ */}
          {!terminal && isActive && mode === 'minimized' && (
            <motion.div
              key="minimized"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              className={`rounded-xl ${glass} overflow-hidden cursor-pointer select-none`}
              style={{ width: 260 }}
              onClick={handleGoToTrain}
            >
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/[0.10] to-transparent" />

              <div className="px-3 py-2.5 flex items-center gap-3">
                {/* Drag handle */}
                <div
                  className="shrink-0 cursor-grab active:cursor-grabbing text-white/15 hover:text-white/30 transition-colors"
                  onPointerDown={(e) => dragControls.start(e)}
                >
                  <GripVertical size={12} />
                </div>

                {/* Pulse dot */}
                <div className="relative shrink-0">
                  <motion.div
                    className="absolute inset-0 rounded-full bg-white/20"
                    animate={{ scale: [1, 1.8, 1], opacity: [0.3, 0, 0.3] }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                  />
                  <div className="w-2 h-2 rounded-full bg-white/60" />
                </div>

                {/* Key info */}
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span className="text-xs font-semibold text-white truncate">
                    {activeSession.model_name || 'Training'}
                  </span>
                  <span className="text-xs font-mono text-white/40 tabular-nums shrink-0">
                    {activeSession.current_step}/{activeSession.total_steps}
                  </span>
                  <span className="text-xs font-mono text-white/25 tabular-nums shrink-0">
                    {pct}%
                  </span>
                </div>

                {/* Expand button */}
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setMode('normal'); }}
                  className="shrink-0 p-1 rounded text-white/20 hover:text-white/50 transition-colors"
                >
                  <Maximize2 size={11} />
                </button>
              </div>

              {/* Thin progress bar */}
              <div className="h-[2px] bg-white/[0.03]">
                <motion.div
                  className="h-full bg-white/40"
                  animate={{ width: `${pct}%` }}
                  transition={{ type: 'spring', stiffness: 100, damping: 20 }}
                />
              </div>
            </motion.div>
          )}

          {/* ═══════════ ACTIVE: NORMAL ═══════════ */}
          {!terminal && isActive && mode === 'normal' && (
            <motion.div
              key="normal"
              initial={{ opacity: 0, y: 16, scale: 0.92 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.95 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              className={`rounded-2xl ${glass} overflow-hidden select-none`}
              style={{ width: 300 }}
            >
              {/* Top edge glow */}
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/[0.12] to-transparent" />

              {/* Header */}
              <div className="px-4 pt-3 pb-2 flex items-center justify-between">
                <div className="flex items-center gap-2 min-w-0">
                  {/* Drag handle */}
                  <div
                    className="shrink-0 cursor-grab active:cursor-grabbing text-white/15 hover:text-white/30 transition-colors"
                    onPointerDown={(e) => dragControls.start(e)}
                  >
                    <GripVertical size={12} />
                  </div>

                  {/* Pulsing indicator */}
                  <div className="relative shrink-0">
                    <motion.div
                      className="absolute inset-0 rounded-full bg-white/15"
                      animate={{ scale: [1, 1.7, 1], opacity: [0.3, 0, 0.3] }}
                      transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
                    />
                    <div className="w-6 h-6 rounded-full bg-white/[0.06] border border-white/[0.1] flex items-center justify-center">
                      <Cpu size={11} className="text-white/70" />
                    </div>
                  </div>

                  <div className="min-w-0">
                    <p className="text-xs font-semibold text-white tracking-tight truncate">
                      {activeSession.model_name || 'Training'}
                    </p>
                    <p className="text-xs text-white/35 font-mono truncate">
                      {formatElapsed(elapsed)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  <button
                    type="button"
                    onClick={() => setMode('minimized')}
                    className="p-1.5 rounded-lg text-white/20 hover:text-white/50 hover:bg-white/[0.04] transition-colors"
                    title="Minimize"
                  >
                    <Minimize2 size={12} />
                  </button>
                  <button
                    type="button"
                    onClick={() => setMode('expanded')}
                    className="p-1.5 rounded-lg text-white/20 hover:text-white/50 hover:bg-white/[0.04] transition-colors"
                    title="Expand"
                  >
                    <Maximize2 size={12} />
                  </button>
                  <button
                    type="button"
                    onClick={handleGoToTrain}
                    className="p-1.5 rounded-lg text-white/20 hover:text-white/50 hover:bg-white/[0.04] transition-colors"
                    title="Go to train page"
                  >
                    <ArrowUpRight size={12} />
                  </button>
                </div>
              </div>

              {/* Phase + Step compact view */}
              <div className="px-4 pb-2">
                <div className="flex items-center gap-3">
                  {/* Mini phase ring */}
                  <div className="relative shrink-0 w-12 h-12 flex items-center justify-center">
                    <MiniPhaseRing phase={activeSession.phase} totalPhases={TOTAL_PHASES} size={48} />
                    <span className="absolute text-xs font-bold text-white tabular-nums">{activeSession.phase}</span>
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-medium text-white truncate">
                      {phaseLabel}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs font-mono text-white/50 tabular-nums">
                        {activeSession.current_step}/{activeSession.total_steps} steps
                      </span>
                      <span className="text-xs font-mono text-white/25 tabular-nums">
                        {pct}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Progress bar */}
              <div className="px-4 pb-3">
                <div className="w-full h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-white/50"
                    animate={{ width: `${pct}%` }}
                    transition={{ type: 'spring', stiffness: 100, damping: 20 }}
                    style={{
                      boxShadow: pct > 0 && pct < 100
                        ? '0 0 8px rgba(255,255,255,0.3)'
                        : 'none',
                    }}
                  />
                </div>
              </div>

              {/* Loss readout */}
              <div className="px-4 pb-3 flex items-center gap-4 border-t border-white/[0.04] pt-2.5">
                <div>
                  <span className="block text-xs text-white/35">Loss</span>
                  <span className="block text-xs font-mono font-semibold text-white tabular-nums mt-0.5">
                    {activeSession.current_loss > 0 ? activeSession.current_loss.toFixed(4) : '\u2014'}
                  </span>
                </div>
                {activeSession.mac_checks_passed > 0 && (
                  <div>
                    <span className="block text-xs text-white/35">MACs</span>
                    <span className="block text-xs font-mono font-semibold text-white tabular-nums mt-0.5">
                      {activeSession.mac_checks_passed}
                    </span>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* ═══════════ ACTIVE: EXPANDED ═══════════ */}
          {!terminal && isActive && mode === 'expanded' && (
            <motion.div
              key="expanded"
              initial={{ opacity: 0, y: 16, scale: 0.92 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.95 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              className={`rounded-2xl ${glass} overflow-hidden select-none`}
              style={{ width: 340 }}
            >
              {/* Top edge glow */}
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/[0.15] to-transparent" />

              {/* Header */}
              <div className="px-5 pt-4 pb-3 flex items-center justify-between">
                <div className="flex items-center gap-2.5 min-w-0">
                  {/* Drag handle */}
                  <div
                    className="shrink-0 cursor-grab active:cursor-grabbing text-white/15 hover:text-white/30 transition-colors"
                    onPointerDown={(e) => dragControls.start(e)}
                  >
                    <GripVertical size={14} />
                  </div>

                  <div className="relative shrink-0">
                    <motion.div
                      className="absolute inset-0 rounded-full bg-white/15"
                      animate={{ scale: [1, 1.7, 1], opacity: [0.3, 0, 0.3] }}
                      transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
                    />
                    <div className="w-7 h-7 rounded-full bg-white/[0.06] border border-white/[0.1] flex items-center justify-center">
                      <Cpu size={13} className="text-white/70" />
                    </div>
                  </div>

                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-white tracking-tight truncate">
                      {activeSession.model_name || 'Training'}
                    </p>
                    <p className="text-xs text-white/35 font-mono truncate">
                      {formatElapsed(elapsed)} &middot; {activeSession.session_id.slice(0, 8)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  <button
                    type="button"
                    onClick={() => setMode('minimized')}
                    className="p-1.5 rounded-lg text-white/20 hover:text-white/50 hover:bg-white/[0.04] transition-colors"
                    title="Minimize"
                  >
                    <Minimize2 size={12} />
                  </button>
                  <button
                    type="button"
                    onClick={() => setMode('normal')}
                    className="p-1.5 rounded-lg text-white/20 hover:text-white/50 hover:bg-white/[0.04] transition-colors"
                    title="Collapse"
                  >
                    <Maximize2 size={12} />
                  </button>
                  <button
                    type="button"
                    onClick={handleGoToTrain}
                    className="p-1.5 rounded-lg text-white/20 hover:text-white/50 hover:bg-white/[0.04] transition-colors"
                    title="Go to train page"
                  >
                    <ArrowUpRight size={12} />
                  </button>
                </div>
              </div>

              {/* Phase ring + status */}
              <div className="px-5 pb-3">
                <div className="flex items-center gap-4">
                  <div className="relative shrink-0 w-16 h-16 flex items-center justify-center">
                    <MiniPhaseRing phase={activeSession.phase} totalPhases={TOTAL_PHASES} size={64} />
                    <div className="absolute flex flex-col items-center">
                      <span className="text-sm font-bold text-white tabular-nums">{activeSession.phase}</span>
                      <span className="text-xs text-white/30">of {TOTAL_PHASES}</span>
                    </div>
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-medium text-white truncate">
                      {phaseLabel}
                    </p>
                    {activeSession.sub_step && (
                      <p className="text-xs text-white/40 mt-0.5 truncate">
                        {activeSession.sub_step}
                      </p>
                    )}
                    <div className="flex items-baseline gap-2 mt-2">
                      <span className="text-lg font-bold font-mono text-white tabular-nums tracking-tighter">
                        {activeSession.current_step}
                      </span>
                      <span className="text-xs font-mono text-white/25 tabular-nums">
                        / {activeSession.total_steps} steps
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Progress bar */}
              <div className="px-5 pb-3">
                <div className="w-full h-2 rounded-full bg-white/[0.04] overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-white/50"
                    animate={{ width: `${pct}%` }}
                    transition={{ type: 'spring', stiffness: 100, damping: 20 }}
                    style={{
                      boxShadow: pct > 0 && pct < 100
                        ? '0 0 10px rgba(255,255,255,0.35)'
                        : 'none',
                    }}
                  />
                </div>
                <div className="flex items-center justify-between mt-1.5">
                  <span className="text-xs font-mono text-white/25 tabular-nums">{pct}%</span>
                  <span className="text-xs text-white/30">
                    {activeSession.elapsed_secs > 0
                      ? formatElapsed(Math.round(activeSession.elapsed_secs))
                      : formatElapsed(elapsed)} elapsed
                  </span>
                </div>
              </div>

              {/* Sparkline */}
              {losses.length >= 2 && (
                <div className="px-5 pb-2">
                  <div className="h-11 w-full">
                    <Sparkline data={losses} width={298} height={44} />
                  </div>
                </div>
              )}

              {/* Stats grid */}
              <div className="px-5 pb-4 grid grid-cols-3 gap-3 border-t border-white/[0.04] pt-3">
                <div>
                  <span className="block text-xs text-white/35">Loss</span>
                  <span className="block text-sm font-mono font-semibold text-white tabular-nums mt-0.5">
                    {activeSession.current_loss > 0 ? activeSession.current_loss.toFixed(4) : '\u2014'}
                  </span>
                </div>
                <div>
                  <span className="block text-xs text-white/35">Accuracy</span>
                  <span className="block text-sm font-mono font-semibold text-white tabular-nums mt-0.5">
                    {(activeSession.accuracy ?? 0) > 0
                      ? `${((activeSession.accuracy ?? 0) * 100).toFixed(1)}%`
                      : '\u2014'}
                  </span>
                </div>
                <div>
                  <span className="block text-xs text-white/35">MACs</span>
                  <span className="block text-sm font-mono font-semibold text-white tabular-nums mt-0.5 flex items-center gap-1">
                    <Shield size={10} className="text-green-400/50" />
                    {activeSession.mac_checks_passed}
                  </span>
                </div>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </motion.div>
    </>
  );
}
