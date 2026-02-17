'use client';

import { useState, useMemo, useEffect, useCallback, useRef, Suspense } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { motion, AnimatePresence, useMotionValue, useSpring } from 'framer-motion';
import {
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Shield,
  Zap,
  Activity,
  Wifi,
  WifiOff,
  Upload,
  Download,
  HardDrive,
  ExternalLink,
  Copy,
  History,
  ChevronDown,
  Link2,
  Layers,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
} from 'recharts';
import { useSignMessage } from 'wagmi';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/utils';
import {
  useMpcTraining,
  type TrainingJobConfig,
  type UploadedData,
  type UploadedWeights,
  type ZeroGStorageResult,
} from '@/hooks/useMpcTraining';
import { useModelRegistry, type ModelWithVersions } from '@/hooks/useModelRegistry';
import { deriveModelKey, decryptWeights } from '@/lib/model-encryption';

// ============================================================================
// Constants
// ============================================================================

const PHASE_DESCRIPTIONS: Record<number, string> = {
  1: 'Deploying contracts',
  2: 'Registering model',
  3: 'Staking tokens',
  4: 'Starting training round',
  5: 'Spawning MPC workers',
  6: 'Distributing data',
  7: 'Generating Beaver triples',
  8: 'Running MPC training',
  9: 'Submitting checkpoint',
  10: 'Verifying MACs',
  11: 'Generating ZK proof',
  12: 'On-chain settlement',
  13: 'Complete',
};

const TOTAL_PHASES = 13;

type WeightFetchStatus = 'idle' | 'fetching' | 'decrypting' | 'uploading' | 'done' | 'error';

// ============================================================================
// Version & Training History
// ============================================================================

const VERSION_REGEX = /^\d+(\.\d+){0,2}$/;
function isValidVersion(v: string): boolean { return VERSION_REGEX.test(v.trim()); }

interface TrainingHistoryEntry {
  sessionId: string;
  version: string;
  accuracy: number | null;
  steps: number;
  totalSteps: number;
  date: string;
  storedOn0G: boolean;
  rootHash?: string;
  status: 'complete' | 'failed';
}

const HISTORY_KEY = 'helix-training-history';

function getTrainingHistory(): TrainingHistoryEntry[] {
  if (typeof window === 'undefined') return [];
  try { const raw = localStorage.getItem(HISTORY_KEY); return raw ? JSON.parse(raw) : []; }
  catch { return []; }
}

function saveTrainingHistory(entries: TrainingHistoryEntry[]): void {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(entries)); }
  catch { /* noop */ }
}

function getNextVersion(history: TrainingHistoryEntry[]): string {
  if (history.length === 0) return '1.0.0';
  let maxMajor = 0;
  for (const entry of history) {
    const major = parseInt(entry.version.split('.')[0], 10);
    if (!isNaN(major) && major > maxMajor) maxMajor = major;
  }
  return `${maxMajor + 1}.0.0`;
}

// ============================================================================
// Shared micro-components
// ============================================================================

function Toggle({ on, onToggle, disabled }: { on: boolean; onToggle: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={cn(
        'relative w-11 h-6 rounded-full transition-colors shrink-0',
        on ? 'bg-white' : 'bg-helix-border2',
        disabled && 'opacity-40 cursor-not-allowed',
      )}
    >
      <motion.div
        className={cn('absolute top-1 w-4 h-4 rounded-full', on ? 'bg-black' : 'bg-helix-muted')}
        animate={{ left: on ? 24 : 4 }}
        transition={{ type: 'spring', stiffness: 500, damping: 35 }}
      />
    </button>
  );
}

function NumberInput({ value, onChange, ...rest }: { value: number; onChange: (v: number) => void } & Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'>) {
  return (
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full bg-transparent text-right text-base text-white outline-none placeholder:text-helix-dim [-moz-appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      {...rest}
    />
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-helix-surface2 border border-helix-border2 rounded-xl px-3 py-2 text-xs shadow-xl">
      <p className="text-helix-dim mb-0.5">Step {label}</p>
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      {payload.map((entry: any, i: number) => (
        <p key={i} className="text-white font-medium">
          {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
}

// ============================================================================
// SettingRow — clean list row
// ============================================================================

function SettingRow({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-5 py-4">
      <div>
        <span className="text-base text-helix-text2">{label}</span>
        {hint && <p className="text-xs text-red-400 mt-0.5">{hint}</p>}
      </div>
      <div className="w-32 flex justify-end">{children}</div>
    </div>
  );
}

// ============================================================================
// UploadRow — minimal file upload
// ============================================================================

function UploadRow({ label, accept, onUpload, uploaded, optional }: {
  label: string;
  accept: string;
  onUpload: (file: File) => Promise<void>;
  uploaded: string | null;
  optional?: boolean;
}) {
  return (
    <div className={cn(
      'flex items-center justify-between px-5 py-3.5 rounded-2xl border transition-colors',
      uploaded
        ? 'bg-green-500/[0.04] border-green-500/15'
        : 'bg-helix-surface border-helix-border',
    )}>
      <div className="flex items-center gap-3 min-w-0">
        {uploaded ? (
          <CheckCircle size={16} className="text-green-400 shrink-0" />
        ) : (
          <Upload size={16} className="text-helix-muted shrink-0" />
        )}
        <div className="min-w-0">
          <p className={cn('text-base truncate', uploaded ? 'text-green-300' : 'text-helix-text2')}>
            {label}
            {optional && !uploaded && <span className="text-helix-dim ml-1.5">optional</span>}
          </p>
          {uploaded && (
            <p className="text-xs text-green-400/60 truncate">{uploaded}</p>
          )}
        </div>
      </div>
      <label className="shrink-0 px-3.5 py-1.5 rounded-xl bg-white/[0.06] text-xs text-helix-text2 hover:text-white hover:bg-white/[0.1] transition-colors cursor-pointer">
        {uploaded ? 'Replace' : 'Upload'}
        <input type="file" accept={accept} className="hidden" onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
        }} />
      </label>
    </div>
  );
}

// ============================================================================
// PaymentNumber — Cash App style animated payment display
// ============================================================================

function PaymentNumber({
  value,
  color,
  editable,
  onChange,
}: {
  value: number;
  color: 'green' | 'yellow' | 'red';
  editable: boolean;
  onChange?: (v: number) => void;
}) {
  const [display, setDisplay] = useState(value);
  const [editText, setEditText] = useState(value.toFixed(4));
  const motionVal = useMotionValue(value);
  const spring = useSpring(motionVal, { stiffness: 80, damping: 20, mass: 0.5 });
  const wasEditable = useRef(editable);

  useEffect(() => {
    if (!editable) motionVal.set(value);
  }, [value, editable, motionVal]);

  useEffect(() => {
    if (editable) return;
    return spring.on('change', (v) => setDisplay(v));
  }, [spring, editable]);

  // When entering edit mode, seed editText with the formatted value
  useEffect(() => {
    if (editable && !wasEditable.current) {
      setEditText(value.toFixed(4));
    }
    wasEditable.current = editable;
  }, [editable, value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const text = e.target.value;
    setEditText(text);
    const num = parseFloat(text);
    if (!isNaN(num) && num >= 0) {
      onChange?.(num);
    }
  };

  const colorClass = color === 'green' ? 'text-green-400'
    : color === 'yellow' ? 'text-yellow-400'
    : 'text-red-400';

  const numberClass = cn(
    'text-7xl font-bold tracking-tighter tabular-nums transition-colors duration-500',
    colorClass,
  );

  return (
    <div className="flex items-baseline justify-center gap-4">
      {editable ? (
        <input
          type="text"
          inputMode="decimal"
          value={editText}
          onChange={handleChange}
          className={cn(numberClass, 'bg-transparent text-center outline-none p-0 border-0 w-56')}
        />
      ) : (
        <span className={numberClass}>
          {display.toFixed(4)}
        </span>
      )}
      <span className={cn('text-2xl font-semibold transition-colors duration-500', colorClass)}>
        ETH
      </span>
    </div>
  );
}

// ============================================================================
// Config Form — two-column full-width layout
// ============================================================================

interface ConfigFormProps {
  onStart: (config: TrainingJobConfig, opts: { storeOn0G: boolean; version: string }) => void;
  isStarting: boolean;
  onUploadData: (file: File) => Promise<void>;
  onUploadWeights: (file: File) => Promise<void>;
  uploadedData: UploadedData | null;
  uploadedWeights: UploadedWeights | null;
  workersOnline: number;
  defaultVersion: string;
  models: ModelWithVersions[];
  selectedModelId: number | null;
  onSelectModel: (tokenId: number | null) => void;
  isFetchingWeights: boolean;
  fetchedModelName: string | null;
}

function ConfigForm({
  onStart, isStarting, onUploadData, onUploadWeights,
  uploadedData, uploadedWeights, workersOnline, defaultVersion,
  models, selectedModelId, onSelectModel, isFetchingWeights, fetchedModelName,
}: ConfigFormProps) {
  const [numSteps, setNumSteps] = useState(500);
  const [learningRate, setLearningRate] = useState(0.001);
  const [checkpointFreq] = useState(50);
  const [zkMode, setZkMode] = useState<'off' | 'always' | 'risk'>('off');
  const [zkCheckpointFreq, setZkCheckpointFreq] = useState(5);
  const [minWorkersForMpc, setMinWorkersForMpc] = useState(2);
  const [paymentEth, setPaymentEth] = useState(1.0);
  const [stakePerWorkerEth, setStakePerWorkerEth] = useState(0.1);
  const [storeOn0G, setStoreOn0G] = useState(false);
  const [version, setVersion] = useState(defaultVersion);
  const [versionError, setVersionError] = useState<string | null>(null);
  const [autoPropose, setAutoPropose] = useState(true);

  // Auto-compute recommended payment: base rate × steps × workers × ZK overhead
  const recommendedPayment = useMemo(() => {
    const workers = workersOnline > 0 ? workersOnline : 3;
    const zkMult = zkMode === 'always' ? 1.75 : zkMode === 'risk' ? 1.25 : 1.0;
    return parseFloat((0.0002 * numSteps * workers * zkMult).toFixed(4));
  }, [numSteps, workersOnline, zkMode]);

  const effectivePayment = autoPropose ? recommendedPayment : paymentEth;

  const paymentColor = (() => {
    if (recommendedPayment <= 0) return 'green' as const;
    const ratio = effectivePayment / recommendedPayment;
    if (ratio >= 0.9) return 'green' as const;
    if (ratio >= 0.6) return 'yellow' as const;
    return 'red' as const;
  })();

  const handleToggleAutoPropose = () => {
    if (autoPropose) setPaymentEth(recommendedPayment);
    setAutoPropose(!autoPropose);
  };

  const handleVersionChange = (v: string) => {
    setVersion(v);
    setVersionError(v.trim() && !isValidVersion(v) ? 'Must be x.y.z format' : null);
  };

  const handleSubmit = () => {
    const v = version.trim() || defaultVersion;
    if (!isValidVersion(v)) return;
    onStart({
      architecture: [784, 128, 10],
      num_workers: workersOnline > 0 ? workersOnline : 3,
      num_steps: numSteps,
      learning_rate: learningRate,
      checkpoint_freq: checkpointFreq,
      mac_interval: 1,
      zk_mode: zkMode,
      zk_checkpoint_freq: zkCheckpointFreq,
      min_workers_for_mpc: minWorkersForMpc,
      train_size: 1000,
      test_size: 200,
      use_real_mnist: true,
      payment_eth: effectivePayment,
      stake_per_worker_eth: stakePerWorkerEth,
      simulate_cheater: false,
      seed: 42,
    }, { storeOn0G, version: v });
  };

  const selectedModel = models.find((m) => m.tokenId === selectedModelId) ?? null;
  const latestVersion = selectedModel?.versions?.length
    ? selectedModel.versions[selectedModel.versions.length - 1]
    : null;

  const canStart = !isStarting && !versionError && !isFetchingWeights;

  return (
    <div className="space-y-6">
      {/* ── Header ────────────────────────────────────────────── */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight text-white">
            New Training Run
          </h1>
          <p className="text-base text-helix-muted mt-1">
            MNIST 784 → 128 → 10 · ~102K params · Verifiable MPC
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.06]">
          <span className={cn('w-2 h-2 rounded-full', workersOnline >= 2 ? 'bg-green-400 animate-pulse' : 'bg-helix-dim')} />
          <span className="text-sm text-helix-text2">
            {workersOnline} worker{workersOnline !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* ── Two-column grid ──────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── LEFT COLUMN ──────────────────────────────────────── */}
        <div className="flex flex-col gap-5">

          {/* Continue from existing model */}
          {models.length > 0 && (
            <div className="space-y-3">
              <div className="relative">
                <Layers size={14} className="absolute left-4 top-1/2 -translate-y-1/2 text-helix-muted pointer-events-none" />
                <select
                  value={selectedModelId ?? ''}
                  onChange={(e) => onSelectModel(e.target.value ? Number(e.target.value) : null)}
                  className="w-full appearance-none pl-10 pr-10 py-3.5 bg-helix-surface border border-helix-border rounded-2xl text-base text-white focus:outline-none focus:border-helix-border2 transition-colors cursor-pointer"
                >
                  <option value="">Start from scratch</option>
                  {models.map((m) => {
                    const latest = m.versions.length ? m.versions[m.versions.length - 1] : null;
                    return (
                      <option key={m.tokenId} value={m.tokenId}>
                        {m.name} {latest ? `v${latest.semver}` : '(no versions)'}
                      </option>
                    );
                  })}
                </select>
                <ChevronDown size={14} className="absolute right-4 top-1/2 -translate-y-1/2 text-helix-muted pointer-events-none" />
              </div>

              <AnimatePresence mode="wait">
                {isFetchingWeights && (
                  <motion.div
                    key="fetching"
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]"
                  >
                    <Loader2 size={14} className="animate-spin text-white" />
                    <span className="text-sm text-helix-text2">Fetching weights from 0G...</span>
                  </motion.div>
                )}
                {!isFetchingWeights && fetchedModelName && uploadedWeights && (
                  <motion.div
                    key="loaded"
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-green-500/[0.06] border border-green-500/20"
                  >
                    <CheckCircle size={14} className="text-green-400" />
                    <span className="text-sm text-green-300">
                      <span className="font-medium">{fetchedModelName}</span>
                      <span className="text-green-400/60 ml-1.5">
                        {uploadedWeights.totalParams.toLocaleString()} params
                      </span>
                    </span>
                  </motion.div>
                )}
                {selectedModel && latestVersion && !latestVersion.weightsStored && !isFetchingWeights && (
                  <motion.div
                    key="no-weights"
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-yellow-500/[0.06] border border-yellow-500/20"
                  >
                    <AlertTriangle size={14} className="text-yellow-400" />
                    <span className="text-sm text-yellow-300/80">No stored weights — upload manually or start fresh</span>
                  </motion.div>
                )}
                {selectedModel && !latestVersion && !isFetchingWeights && (
                  <motion.div
                    key="no-versions"
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-yellow-500/[0.06] border border-yellow-500/20"
                  >
                    <AlertTriangle size={14} className="text-yellow-400" />
                    <span className="text-sm text-yellow-300/80">No versions yet — upload weights or start fresh</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Core training settings */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden divide-y divide-helix-border">
            <SettingRow label="Version" hint={versionError ?? undefined}>
              <input
                type="text"
                value={version}
                onChange={(e) => handleVersionChange(e.target.value)}
                placeholder={defaultVersion}
                className={cn(
                  'bg-transparent text-right text-base outline-none placeholder:text-helix-dim w-24',
                  versionError ? 'text-red-400' : 'text-white',
                )}
              />
            </SettingRow>

            <SettingRow label="Training Steps">
              <NumberInput value={numSteps} onChange={setNumSteps} min={10} max={10000} />
            </SettingRow>

            <SettingRow label="Learning Rate">
              <NumberInput value={learningRate} onChange={setLearningRate} step={0.0001} min={0.00001} max={1} />
            </SettingRow>

            <SettingRow label="Stake / Worker">
              <div className="flex items-center gap-1.5">
                <NumberInput value={stakePerWorkerEth} onChange={setStakePerWorkerEth} step={0.01} min={0} />
                <span className="text-sm text-helix-dim shrink-0">ETH</span>
              </div>
            </SettingRow>
          </div>

          {/* Uploads */}
          <div className="space-y-3">
            <UploadRow
              label="Training Data"
              accept=".json"
              onUpload={onUploadData}
              uploaded={uploadedData ? `${uploadedData.samples} samples · ${uploadedData.inputDim}D` : null}
            />
            <UploadRow
              label={fetchedModelName ? `Weights · ${fetchedModelName}` : 'Initial Weights'}
              accept=".json"
              onUpload={onUploadWeights}
              uploaded={uploadedWeights ? `${uploadedWeights.totalParams.toLocaleString()} params` : null}
              optional
            />
          </div>
        </div>

        {/* ── RIGHT COLUMN ─────────────────────────────────────── */}
        <div className="flex flex-col gap-5">

          {/* Payment + Start — Cash App style, connected */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
            <div className="p-6">
              {/* Header with auto toggle */}
              <div className="flex items-center justify-between mb-5">
                <span className="text-base font-medium text-helix-text2">Payment</span>
                <div className="flex items-center gap-2.5">
                  <span className="text-sm text-helix-dim">Auto</span>
                  <Toggle on={autoPropose} onToggle={handleToggleAutoPropose} />
                </div>
              </div>

              {/* Big number — same component for both modes */}
              <div className="py-3">
                <PaymentNumber
                  value={autoPropose ? recommendedPayment : paymentEth}
                  color={paymentColor}
                  editable={!autoPropose}
                  onChange={setPaymentEth}
                />
              </div>

              {/* Footer — fixed height, breakdown slides horizontally, rec fades in */}
              <div className="relative h-5 mt-4">
                <motion.span
                  className="absolute top-0 text-sm text-helix-dim whitespace-nowrap"
                  animate={{ left: autoPropose ? '50%' : 0, x: autoPropose ? '-50%' : 0 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                >
                  {workersOnline > 0 ? workersOnline : 3} workers × {numSteps} steps
                  {zkMode !== 'off' && ` × ${zkMode === 'always' ? '1.75' : '1.25'}× ZK`}
                </motion.span>
                <AnimatePresence>
                  {!autoPropose && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-0 right-0 text-sm text-helix-dim whitespace-nowrap"
                    >
                      Rec: {recommendedPayment.toFixed(4)} ETH
                    </motion.span>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Start button — attached to payment card */}
            <motion.button
              type="button"
              onClick={handleSubmit}
              disabled={!canStart}
              whileTap={canStart ? { scale: 0.98 } : {}}
              className={cn(
                'w-full py-5 text-xl font-bold tracking-tight transition-all',
                'flex items-center justify-center gap-3 border-t',
                canStart
                  ? 'bg-zinc-200 text-black border-zinc-200 hover:bg-zinc-300 active:bg-zinc-400'
                  : 'bg-helix-border text-helix-muted border-helix-border cursor-not-allowed',
              )}
            >
              {isStarting ? (
                <><Loader2 size={22} className="animate-spin" /> Starting...</>
              ) : isFetchingWeights ? (
                <><Loader2 size={22} className="animate-spin" /> Loading Weights...</>
              ) : (
                <><Play size={22} /> Start Training</>
              )}
            </motion.button>
          </div>

          {/* ZK Mode */}
          <div className="space-y-3">
            <span className="text-base font-medium text-helix-text2">ZK Proofs</span>
            <div className="grid grid-cols-3 gap-2">
              {(['off', 'always', 'risk'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setZkMode(mode)}
                  className={cn(
                    'py-2.5 rounded-xl text-base font-medium transition-all',
                    zkMode === mode
                      ? 'bg-white text-black shadow-lg shadow-white/5'
                      : 'bg-helix-surface border border-helix-border text-helix-muted hover:text-white hover:border-helix-border2',
                  )}
                >
                  {mode === 'off' ? 'Off' : mode === 'always' ? 'Always' : 'Risk-Based'}
                </button>
              ))}
            </div>

            <AnimatePresence>
              {zkMode === 'always' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="flex items-center justify-between py-3.5 px-4 bg-helix-surface border border-helix-border rounded-xl">
                    <span className="text-base text-helix-text2">ZK Checkpoint Freq</span>
                    <NumberInput value={zkCheckpointFreq} onChange={setZkCheckpointFreq} min={1} max={100} />
                  </div>
                </motion.div>
              )}
              {zkMode === 'risk' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="flex items-center justify-between py-3.5 px-4 bg-helix-surface border border-helix-border rounded-xl">
                    <span className="text-base text-helix-text2">Min Workers for MPC</span>
                    <NumberInput value={minWorkersForMpc} onChange={setMinWorkersForMpc} min={2} max={workersOnline > 2 ? workersOnline : 7} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Store on 0G */}
          <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
            <div className="flex items-center justify-between px-5 py-4">
              <div>
                <p className="text-base text-white">Store on 0G</p>
                <p className="text-xs text-helix-dim mt-0.5">Save model to decentralized storage</p>
              </div>
              <Toggle on={storeOn0G} onToggle={() => setStoreOn0G(!storeOn0G)} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Live Progress — full-width two-column training view
// ============================================================================

interface LiveProgressProps {
  session: NonNullable<ReturnType<typeof useMpcTraining>['session']>;
  losses: { step: number; loss: number }[];
  isConnected: boolean;
  error: string | null;
  version: string;
  onDownloadModel: (sessionId: string) => Promise<void>;
  onStoreOnZeroG: (sessionId: string, version?: string) => Promise<void>;
  isStoringOnZeroG: boolean;
  zeroGResult: ZeroGStorageResult | null;
  showStoreOn0G: boolean;
}

function LiveProgress({ session, losses, isConnected, error, version, onDownloadModel, onStoreOnZeroG, isStoringOnZeroG, zeroGResult, showStoreOn0G }: LiveProgressProps) {
  const stepProgress = session.total_steps > 0
    ? (session.current_step / session.total_steps) * 100
    : 0;
  const isTerminal = session.status === 'complete' || session.status === 'failed';
  const isComplete = session.status === 'complete';

  return (
    <div className="space-y-6">
      {/* ── Status bar ──────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isConnected ? (
            <Wifi size={14} className="text-green-400" />
          ) : (
            <WifiOff size={14} className="text-red-400" />
          )}
          <span className="text-sm text-helix-dim">
            {session.session_id.slice(0, 8)}
          </span>
          <Badge variant="default">v{version}</Badge>
        </div>
        <div className={cn(
          'flex items-center gap-2 px-3.5 py-1.5 rounded-full text-sm font-medium',
          isComplete ? 'bg-green-500/10 text-green-400'
            : session.status === 'failed' ? 'bg-red-500/10 text-red-400'
            : 'bg-white/[0.06] text-white',
        )}>
          {!isTerminal && <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />}
          {session.status === 'running' ? 'Training' : session.status}
        </div>
      </div>

      {/* ── Error ───────────────────────────────────────────── */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="px-4 py-3 rounded-2xl bg-red-500/[0.06] border border-red-500/20"
        >
          <p className="text-sm text-red-300">{error}</p>
        </motion.div>
      )}

      {/* ── Two-column layout ─────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* LEFT: Hero metric + chart */}
        <div className="space-y-6">
          {/* Hero metric */}
          <div className="flex items-center justify-center py-8 rounded-2xl bg-helix-surface border border-helix-border">
            {isComplete && session.accuracy !== null ? (
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', stiffness: 200, damping: 20 }}
                className="text-center"
              >
                <p className="text-sm text-helix-muted uppercase tracking-wider mb-2">Final Accuracy</p>
                <p className="text-7xl font-semibold tracking-tighter text-white">
                  {(session.accuracy * 100).toFixed(1)}
                  <span className="text-3xl text-helix-muted ml-1">%</span>
                </p>
              </motion.div>
            ) : session.status === 'failed' ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center">
                <XCircle size={48} className="text-red-400 mx-auto mb-3" />
                <p className="text-xl font-medium text-red-300">Training Failed</p>
              </motion.div>
            ) : (
              <div className="text-center">
                <p className="text-sm text-helix-muted uppercase tracking-wider mb-2">Current Loss</p>
                <p className="text-7xl font-semibold tracking-tighter text-white tabular-nums">
                  {session.current_loss > 0 ? session.current_loss.toFixed(4) : '—'}
                </p>
              </div>
            )}
          </div>

          {/* Loss curve chart */}
          <LossCurve data={losses} />
        </div>

        {/* RIGHT: Progress, phases, stats */}
        <div className="space-y-5">

          {/* Progress bar */}
          {!isTerminal && (
            <div className="space-y-2">
              <div className="h-2 w-full rounded-full bg-helix-border overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-white"
                  animate={{ width: `${stepProgress}%` }}
                  transition={{ duration: 0.4, ease: 'easeOut' }}
                />
              </div>
              <div className="flex items-center justify-between text-sm text-helix-dim">
                <span>Step {session.current_step}</span>
                <span>{session.total_steps}</span>
              </div>
            </div>
          )}

          {/* Phase indicator */}
          {!isTerminal && (
            <div className="flex items-center gap-3 px-4 py-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.05]">
              <div className="flex gap-[3px]">
                {Array.from({ length: TOTAL_PHASES }, (_, i) => (
                  <div
                    key={i}
                    className={cn(
                      'w-1.5 h-3 rounded-sm transition-all duration-300',
                      i + 1 < session.phase ? 'bg-white'
                        : i + 1 === session.phase ? 'bg-white/60'
                        : 'bg-helix-border',
                    )}
                  />
                ))}
              </div>
              <span className="text-sm text-helix-text2">
                {session.phase_description || PHASE_DESCRIPTIONS[session.phase] || 'Processing...'}
              </span>
              <span className="ml-auto text-sm text-helix-dim">
                {session.phase}/{TOTAL_PHASES}
              </span>
            </div>
          )}

          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="MAC Checks" value={session.mac_checks_passed} icon={<Shield size={14} />} />
            <StatCard label="Checkpoints" value={session.checkpoints_submitted} icon={<Zap size={14} />} />
            <StatCard label="ZK Proofs" value={session.zk_proofs_generated} icon={<Activity size={14} />} />
            <StatCard label="Elapsed" value={
              session.elapsed_secs > 0
                ? `${session.elapsed_secs.toFixed(0)}s`
                : `${Math.floor((Date.now() / 1000) - session.started_at)}s`
            } icon={null} />
          </div>

          {/* Cheater alert */}
          {session.cheater_detected && (
            <motion.div
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3 px-4 py-3.5 rounded-2xl bg-red-500/[0.08] border border-red-500/25"
            >
              <AlertTriangle size={16} className="text-red-400 shrink-0" />
              <div>
                <p className="text-sm font-medium text-red-300">Cheater Detected</p>
                <p className="text-xs text-red-300/60 mt-0.5">
                  Worker {session.cheater_detected.party_index} · Step {session.cheater_detected.step} · Stake slashed
                </p>
              </div>
            </motion.div>
          )}

          {/* Results actions */}
          {isTerminal && (
            <ResultsActions
              session={session}
              version={version}
              onDownloadModel={onDownloadModel}
              onStoreOnZeroG={onStoreOnZeroG}
              isStoringOnZeroG={isStoringOnZeroG}
              zeroGResult={zeroGResult}
              showStoreOn0G={showStoreOn0G}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// StatCard — clean metric card
// ============================================================================

function StatCard({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode | null }) {
  return (
    <div className="bg-helix-surface border border-helix-border rounded-2xl px-4 py-3.5">
      <div className="flex items-center gap-1.5 mb-1.5">
        {icon && <span className="text-helix-dim">{icon}</span>}
        <span className="text-xs text-helix-dim uppercase tracking-wider">{label}</span>
      </div>
      <p className="text-2xl font-semibold text-white tabular-nums">{value}</p>
    </div>
  );
}

// ============================================================================
// Loss Curve — area chart
// ============================================================================

function LossCurve({ data }: { data: { step: number; loss: number }[] }) {
  const chartData = useMemo(() => {
    if (data.length <= 200) return data;
    const step = Math.ceil(data.length / 200);
    return data.filter((_, i) => i % step === 0 || i === data.length - 1);
  }, [data]);

  if (chartData.length < 2) {
    return (
      <div className="flex items-center justify-center h-48 rounded-2xl bg-helix-surface border border-helix-border text-helix-dim text-sm">
        Waiting for data...
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-helix-surface border border-helix-border p-5 pt-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-helix-muted">Loss</span>
        <span className="text-sm text-helix-dim tabular-nums">
          {chartData[chartData.length - 1].loss.toFixed(4)}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(255,255,255,0.12)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0)" />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="step"
            stroke="transparent"
            tick={{ fill: '#3e3e44', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke="transparent"
            tick={{ fill: '#3e3e44', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={40}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <RechartsTooltip content={<ChartTooltip />} cursor={{ stroke: '#2a2a2e', strokeWidth: 1 }} />
          <Area
            type="monotone"
            dataKey="loss"
            name="Loss"
            stroke="#ffffff"
            strokeWidth={1.5}
            fill="url(#lossGrad)"
            dot={false}
            activeDot={{ r: 3, fill: '#ffffff', stroke: '#111113', strokeWidth: 2 }}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ============================================================================
// Results Actions — post-training buttons
// ============================================================================

function ResultsActions({ session, version, onDownloadModel, onStoreOnZeroG, isStoringOnZeroG, zeroGResult, showStoreOn0G }: {
  session: LiveProgressProps['session'];
  version: string;
  onDownloadModel: (sessionId: string) => Promise<void>;
  onStoreOnZeroG: (sessionId: string, version?: string) => Promise<void>;
  isStoringOnZeroG: boolean;
  zeroGResult: ZeroGStorageResult | null;
  showStoreOn0G: boolean;
}) {
  const isSuccess = session.status === 'complete';
  const [copied, setCopied] = useState(false);

  const copyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isSuccess) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-3"
    >
      {/* Summary stats */}
      <div className="flex items-center gap-6 py-3 text-sm text-helix-muted">
        <span>{session.current_step}/{session.total_steps} steps</span>
        <span className="w-px h-3 bg-helix-border" />
        <span>
          {session.elapsed_secs > 0
            ? `${session.elapsed_secs.toFixed(1)}s`
            : session.started_at > 0
              ? `${((Date.now() / 1000) - session.started_at).toFixed(1)}s`
              : '--'}
        </span>
        <span className="w-px h-3 bg-helix-border" />
        <span>{session.checkpoints_submitted} checkpoints</span>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={() => onDownloadModel(session.session_id)}
          className="flex-1 flex items-center justify-center gap-2 py-3.5 rounded-2xl bg-white text-black text-sm font-semibold hover:shadow-lg hover:shadow-white/10 transition-all"
        >
          <Download size={16} />
          Download Model
        </button>

        {showStoreOn0G && !zeroGResult && (
          <button
            type="button"
            onClick={() => onStoreOnZeroG(session.session_id, version)}
            disabled={isStoringOnZeroG}
            className={cn(
              'flex-1 flex items-center justify-center gap-2 py-3.5 rounded-2xl border text-sm font-semibold transition-all',
              isStoringOnZeroG
                ? 'bg-helix-surface border-helix-border text-helix-muted cursor-wait'
                : 'bg-helix-surface border-helix-border text-white hover:border-helix-border2',
            )}
          >
            {isStoringOnZeroG ? (
              <><Loader2 size={16} className="animate-spin" /> Uploading...</>
            ) : (
              <><HardDrive size={16} /> Store on 0G</>
            )}
          </button>
        )}
      </div>

      {/* 0G result */}
      {zeroGResult && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl bg-green-500/[0.04] border border-green-500/15 p-4 space-y-3"
        >
          <div className="flex items-center gap-2">
            <CheckCircle size={14} className="text-green-400" />
            <span className="text-sm text-green-300 font-medium">Stored on 0G</span>
          </div>

          <div className="flex items-center gap-2">
            <code className="flex-1 text-xs text-green-300/70 bg-green-500/[0.06] px-3 py-2 rounded-xl truncate">
              {zeroGResult.rootHash}
            </code>
            <button
              type="button"
              onClick={() => copyHash(zeroGResult.rootHash)}
              className="shrink-0 p-2 rounded-xl bg-green-500/[0.06] text-green-400 hover:text-green-300 transition-colors"
            >
              {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
            </button>
          </div>

          <div className="flex items-center gap-4">
            <a
              href={zeroGResult.explorerUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-green-400/60 hover:text-green-400 transition-colors flex items-center gap-1"
            >
              View on Explorer <ExternalLink size={10} />
            </a>

            <Link
              href={`/my-models?action=add-version&session=${session.session_id}&hash=${zeroGResult.rootHash}&version=${version}&accuracy=${session.accuracy !== null ? session.accuracy * 100 : ''}`}
              className="text-xs text-green-400/60 hover:text-green-400 transition-colors flex items-center gap-1"
            >
              Save to Registry <Link2 size={10} />
            </Link>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

// ============================================================================
// Training History
// ============================================================================

function TrainingHistoryList({ history, onClearHistory }: { history: TrainingHistoryEntry[]; onClearHistory: () => void }) {
  if (history.length === 0) return null;

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <History size={14} className="text-helix-muted" />
          <span className="text-sm font-medium text-helix-text2">History</span>
          <span className="text-sm text-helix-dim">{history.length}</span>
        </div>
        <button
          type="button"
          onClick={onClearHistory}
          className="text-xs text-helix-dim hover:text-red-400 transition-colors"
        >
          Clear
        </button>
      </div>

      <div className="space-y-1.5">
        {history.slice().reverse().map((entry) => (
          <div
            key={entry.sessionId}
            className="flex items-center gap-3 px-4 py-3 rounded-xl bg-helix-surface border border-helix-border"
          >
            {entry.status === 'complete' ? (
              <CheckCircle size={12} className="text-green-400 shrink-0" />
            ) : (
              <XCircle size={12} className="text-red-400 shrink-0" />
            )}

            <span className="text-sm text-helix-muted">v{entry.version}</span>

            <div className="flex-1" />

            {entry.accuracy !== null && (
              <span className="text-base font-medium text-white tabular-nums">
                {(entry.accuracy * 100).toFixed(1)}%
              </span>
            )}

            <span className="text-sm text-helix-dim tabular-nums">
              {entry.steps}/{entry.totalSteps}
            </span>

            {entry.storedOn0G && (
              <HardDrive size={10} className="text-green-400/60" />
            )}

            <span className="text-sm text-helix-dim">
              {new Date(entry.date).toLocaleDateString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Inner page (needs useSearchParams, so must be inside Suspense)
// ============================================================================

function TrainPageInner() {
  const searchParams = useSearchParams();
  const queryModelId = searchParams.get('model');

  const {
    startTraining, uploadData, uploadWeights, downloadModel, storeOnZeroG,
    session, losses, isConnected, isStarting, error: trainingError,
    uploadedData, uploadedWeights, workersOnline, zeroGResult, isStoringOnZeroG,
  } = useMpcTraining();

  const { models, isLoading: isLoadingModels } = useModelRegistry();
  const { signMessageAsync } = useSignMessage();

  const hasSession = session !== null;
  const [wantsStoreOn0G, setWantsStoreOn0G] = useState(false);
  const [currentVersion, setCurrentVersion] = useState('1.0.0');
  const [history, setHistory] = useState<TrainingHistoryEntry[]>([]);
  const historyRecordedRef = useRef(false);

  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [weightFetchStatus, setWeightFetchStatus] = useState<WeightFetchStatus>('idle');
  const [weightFetchError, setWeightFetchError] = useState<string | null>(null);
  const [fetchedModelName, setFetchedModelName] = useState<string | null>(null);
  const fetchingForRef = useRef<number | null>(null);
  const autoSelectAppliedRef = useRef(false);

  // Auto-select from query param
  useEffect(() => {
    if (autoSelectAppliedRef.current) return;
    if (!queryModelId || isLoadingModels) return;
    const tokenId = Number(queryModelId);
    if (isNaN(tokenId)) return;
    const found = models.find((m) => m.tokenId === tokenId);
    if (found) {
      autoSelectAppliedRef.current = true;
      setSelectedModelId(tokenId);
    }
  }, [queryModelId, models, isLoadingModels]);

  // Fetch weights when selectedModelId changes
  useEffect(() => {
    if (selectedModelId === null) {
      if (fetchingForRef.current !== null) fetchingForRef.current = null;
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      setFetchedModelName(null);
      return;
    }

    const model = models.find((m) => m.tokenId === selectedModelId);
    if (!model) return;

    const lv = model.versions.length ? model.versions[model.versions.length - 1] : null;
    if (!lv || !lv.weightsStored || !lv.rootHash) {
      setWeightFetchStatus('idle');
      setWeightFetchError(null);
      setFetchedModelName(null);
      return;
    }

    if (fetchingForRef.current === selectedModelId) return;
    fetchingForRef.current = selectedModelId;

    const fetchWeights = async () => {
      setWeightFetchStatus('fetching');
      setWeightFetchError(null);
      setFetchedModelName(null);

      try {
        const res = await fetch('/api/fetch-from-0g', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rootHash: lv.rootHash }),
        });

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.error || `Failed to fetch from 0G (HTTP ${res.status})`);
        }

        const result = await res.json();
        let weightsData: unknown;

        if (result.encoding === 'base64') {
          setWeightFetchStatus('decrypting');
          const rawBytes = Uint8Array.from(atob(result.data), (c) => c.charCodeAt(0));
          const signMsg = async (message: string): Promise<string> => signMessageAsync({ message });
          const key = await deriveModelKey(signMsg, selectedModelId);
          const decryptedJson = await decryptWeights(key, rawBytes);
          weightsData = JSON.parse(decryptedJson);
        } else {
          weightsData = result.data;
        }

        setWeightFetchStatus('uploading');
        const weightsJson = JSON.stringify(weightsData);
        const blob = new Blob([weightsJson], { type: 'application/json' });
        const syntheticFile = new File([blob], `weights-from-${model.slug}.json`, { type: 'application/json' });
        await uploadWeights(syntheticFile);

        setFetchedModelName(model.name);
        setWeightFetchStatus('done');
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch weights';
        setWeightFetchError(message);
        setWeightFetchStatus('error');
        fetchingForRef.current = null;
      }
    };

    fetchWeights();
  }, [selectedModelId, models, signMessageAsync, uploadWeights]);

  const handleSelectModel = useCallback((tokenId: number | null) => {
    fetchingForRef.current = null;
    setSelectedModelId(tokenId);
  }, []);

  // Load history on mount
  useEffect(() => {
    const h = getTrainingHistory();
    setHistory(h);
    setCurrentVersion(getNextVersion(h));
  }, []);

  // Record to history when session reaches terminal state
  useEffect(() => {
    if (!session) { historyRecordedRef.current = false; return; }
    if (historyRecordedRef.current) return;
    if (session.status !== 'complete' && session.status !== 'failed') return;

    historyRecordedRef.current = true;
    const entry: TrainingHistoryEntry = {
      sessionId: session.session_id,
      version: currentVersion,
      accuracy: session.accuracy,
      steps: session.current_step,
      totalSteps: session.total_steps,
      date: new Date().toISOString(),
      storedOn0G: false,
      status: session.status as 'complete' | 'failed',
    };

    setHistory((prev) => {
      const updated = [...prev, entry];
      saveTrainingHistory(updated);
      return updated;
    });
  }, [session, currentVersion]);

  // Update history when 0G storage completes
  useEffect(() => {
    if (!zeroGResult || !session) return;
    setHistory((prev) => {
      const updated = prev.map((e) =>
        e.sessionId === session.session_id
          ? { ...e, storedOn0G: true, rootHash: zeroGResult.rootHash }
          : e,
      );
      saveTrainingHistory(updated);
      return updated;
    });
  }, [zeroGResult, session]);

  const handleStart = (config: TrainingJobConfig, opts: { storeOn0G: boolean; version: string }) => {
    setWantsStoreOn0G(opts.storeOn0G);
    setCurrentVersion(opts.version);
    startTraining(config);
  };

  const handleClearHistory = useCallback(() => {
    setHistory([]);
    saveTrainingHistory([]);
    setCurrentVersion('1.0.0');
  }, []);

  const defaultVersion = useMemo(() => getNextVersion(history), [history]);
  const isFetchingWeights = weightFetchStatus === 'fetching' || weightFetchStatus === 'decrypting' || weightFetchStatus === 'uploading';

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="py-4 space-y-8"
    >
      {/* Weight fetch error */}
      {weightFetchStatus === 'error' && weightFetchError && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2.5 px-4 py-3 rounded-2xl bg-red-500/[0.06] border border-red-500/20"
        >
          <XCircle size={14} className="text-red-400 shrink-0" />
          <p className="text-sm text-red-300">{weightFetchError}</p>
        </motion.div>
      )}

      {!hasSession ? (
        <ConfigForm
          onStart={handleStart}
          isStarting={isStarting}
          onUploadData={uploadData}
          onUploadWeights={uploadWeights}
          uploadedData={uploadedData}
          uploadedWeights={uploadedWeights}
          workersOnline={workersOnline}
          defaultVersion={defaultVersion}
          models={models}
          selectedModelId={selectedModelId}
          onSelectModel={handleSelectModel}
          isFetchingWeights={isFetchingWeights}
          fetchedModelName={fetchedModelName}
        />
      ) : (
        <LiveProgress
          session={session}
          losses={losses}
          isConnected={isConnected}
          error={trainingError}
          version={currentVersion}
          onDownloadModel={downloadModel}
          onStoreOnZeroG={storeOnZeroG}
          isStoringOnZeroG={isStoringOnZeroG}
          zeroGResult={zeroGResult}
          showStoreOn0G={wantsStoreOn0G}
        />
      )}

      <TrainingHistoryList history={history} onClearHistory={handleClearHistory} />
    </motion.div>
  );
}

// ============================================================================
// Main Page (wrapped in Suspense for useSearchParams)
// ============================================================================

export default function TrainPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-20">
          <Loader2 size={24} className="animate-spin text-helix-muted" />
        </div>
      }
    >
      <TrainPageInner />
    </Suspense>
  );
}
