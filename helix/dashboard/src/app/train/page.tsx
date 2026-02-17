'use client';

import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Shield,
  Zap,
  Activity,
  Clock,
  Wifi,
  WifiOff,
  Upload,
  Download,
  HardDrive,
  ExternalLink,
  Copy,
  History,
  Tag,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
} from 'recharts';
import { Card } from '@/components/ui/Card';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/utils';
import {
  useMpcTraining,
  type TrainingJobConfig,
  type UploadedData,
  type UploadedWeights,
  type ZeroGStorageResult,
} from '@/hooks/useMpcTraining';

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

// ============================================================================
// Version & Training History
// ============================================================================

const VERSION_REGEX = /^\d+(\.\d+){0,2}$/;

function isValidVersion(v: string): boolean {
  return VERSION_REGEX.test(v.trim());
}

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
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveTrainingHistory(entries: TrainingHistoryEntry[]): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
  } catch {
    // localStorage full or unavailable
  }
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
// Chart tooltip
// ============================================================================

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-helix-surface2 border border-helix-border rounded-lg p-2 text-xs shadow-lg">
      <p className="text-helix-muted font-mono mb-1">Step {label}</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} className="text-helix-text">
          <span className="text-helix-text2">{entry.name}: </span>
          {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
}

// ============================================================================
// Config Form Component
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
}

function ConfigForm({ onStart, isStarting, onUploadData, onUploadWeights, uploadedData, uploadedWeights, workersOnline, defaultVersion }: ConfigFormProps) {
  const [numSteps, setNumSteps] = useState(500);
  const [learningRate, setLearningRate] = useState(0.001);
  const [checkpointFreq, setCheckpointFreq] = useState(50);
  const [zkMode, setZkMode] = useState<'off' | 'always' | 'risk'>('off');
  const [zkCheckpointFreq, setZkCheckpointFreq] = useState(5);
  const [minWorkersForMpc, setMinWorkersForMpc] = useState(2);
  const [paymentEth, setPaymentEth] = useState(1.0);
  const [stakePerWorkerEth, setStakePerWorkerEth] = useState(0.1);
  const [simulateCheater, setSimulateCheater] = useState(false);
  const [storeOn0G, setStoreOn0G] = useState(false);
  const [version, setVersion] = useState(defaultVersion);
  const [versionError, setVersionError] = useState<string | null>(null);

  const handleVersionChange = (v: string) => {
    setVersion(v);
    if (v.trim() && !isValidVersion(v)) {
      setVersionError('Version must be x, x.y, or x.y.z (e.g. 1, 1.0, 1.0.0)');
    } else {
      setVersionError(null);
    }
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
      payment_eth: paymentEth,
      stake_per_worker_eth: stakePerWorkerEth,
      simulate_cheater: simulateCheater,
      seed: 42,
    }, { storeOn0G, version: v });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Model Architecture */}
      <Card variant="default">
        <h3 className="text-sm font-medium text-white mb-4">Model Architecture</h3>
        <div className="space-y-4">
          <div>
            <label className="label-text block mb-1.5">Architecture</label>
            <div className="flex items-center gap-2 px-3 py-2 bg-helix-bg border border-helix-border rounded-md">
              <span className="text-sm text-helix-text font-mono">MNIST 784 → 128 → 10</span>
              <Badge variant="default" className="ml-auto">~102K params</Badge>
            </div>
          </div>

          <div>
            <label className="label-text block mb-1.5">
              <span className="flex items-center gap-1.5">
                <Tag size={12} />
                Model Version
              </span>
            </label>
            <input
              type="text"
              value={version}
              onChange={(e) => handleVersionChange(e.target.value)}
              placeholder={defaultVersion}
              className={cn(
                'w-full px-3 py-2 bg-helix-bg border rounded-md text-sm text-helix-text font-mono focus:outline-none transition-colors',
                versionError ? 'border-red-500/50 focus:border-red-500' : 'border-helix-border focus:border-helix-border2',
              )}
            />
            {versionError && (
              <p className="text-2xs text-red-400 mt-1">{versionError}</p>
            )}
          </div>

          <div>
            <label className="label-text block mb-1.5">MPC Workers</label>
            <div className="flex items-center gap-2 px-3 py-2 bg-helix-bg border border-helix-border rounded-md">
              <span className={cn(
                'inline-block w-2 h-2 rounded-full',
                workersOnline >= 2 ? 'bg-green-400 animate-pulse' : 'bg-red-400',
              )} />
              <span className="text-sm text-helix-text font-mono">
                {workersOnline} worker{workersOnline !== 1 ? 's' : ''} online
              </span>
              {workersOnline < 2 && (
                <span className="text-2xs text-red-400 ml-auto">Need 2+</span>
              )}
            </div>
          </div>

          <div>
            <label className="label-text block mb-1.5">Training Steps</label>
            <input
              type="number"
              value={numSteps}
              onChange={(e) => setNumSteps(Number(e.target.value))}
              min={10}
              max={10000}
              className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
            />
          </div>

          <div>
            <label className="label-text block mb-1.5">Learning Rate</label>
            <input
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              step={0.0001}
              min={0.00001}
              max={1}
              className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
            />
          </div>

          <div>
            <label className="label-text block mb-1.5">Checkpoint Frequency</label>
            <input
              type="number"
              value={checkpointFreq}
              onChange={(e) => setCheckpointFreq(Number(e.target.value))}
              min={1}
              max={1000}
              className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
            />
          </div>
        </div>
      </Card>

      {/* Data & Weights */}
      <Card variant="default">
        <h3 className="text-sm font-medium text-white mb-4">Data & Weights</h3>
        <div className="space-y-4">
          <div>
            <label className="label-text block mb-1.5">Training Data (JSON)</label>
            <div className="flex items-center gap-3">
              <label
                className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded-md text-sm cursor-pointer transition-colors',
                  'bg-helix-bg border border-helix-border hover:border-helix-border2 text-helix-text',
                )}
              >
                <Upload size={14} />
                <span>Upload</span>
                <input
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) onUploadData(file);
                  }}
                />
              </label>
              {uploadedData && (
                <div className="flex items-center gap-2">
                  <CheckCircle size={14} className="text-green-400" />
                  <span className="text-sm text-helix-text font-mono">
                    {uploadedData.samples} samples ({uploadedData.inputDim}D)
                  </span>
                </div>
              )}
            </div>
          </div>

          <div>
            <label className="label-text block mb-1.5">Initial Weights (JSON, optional)</label>
            <div className="flex items-center gap-3">
              <label
                className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded-md text-sm cursor-pointer transition-colors',
                  'bg-helix-bg border border-helix-border hover:border-helix-border2 text-helix-text',
                )}
              >
                <Upload size={14} />
                <span>Upload</span>
                <input
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) onUploadWeights(file);
                  }}
                />
              </label>
              {uploadedWeights && (
                <div className="flex items-center gap-2">
                  <CheckCircle size={14} className="text-green-400" />
                  <span className="text-sm text-helix-text font-mono">
                    {uploadedWeights.totalParams.toLocaleString()} params
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>

      {/* Security & Economics */}
      <Card variant="default">
        <h3 className="text-sm font-medium text-white mb-4">Security & Economics</h3>
        <div className="space-y-4">
          {/* ZK Mode */}
          <div>
            <label className="label-text block mb-2">ZK Proofs</label>
            <div className="flex gap-2">
              {(['off', 'always', 'risk'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setZkMode(mode)}
                  className={cn(
                    'flex-1 px-3 py-2 rounded-md text-xs font-mono uppercase tracking-wider transition-all',
                    zkMode === mode
                      ? 'bg-white text-black'
                      : 'bg-helix-bg border border-helix-border text-helix-muted hover:border-helix-border2',
                  )}
                >
                  {mode === 'off' ? 'Off' : mode === 'always' ? 'Always' : 'Risk-Based'}
                </button>
              ))}
            </div>
          </div>

          {/* Conditional ZK fields */}
          <AnimatePresence>
            {zkMode === 'always' && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
              >
                <label className="label-text block mb-1.5">ZK Checkpoint Frequency</label>
                <input
                  type="number"
                  value={zkCheckpointFreq}
                  onChange={(e) => setZkCheckpointFreq(Number(e.target.value))}
                  min={1}
                  max={100}
                  className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
                />
              </motion.div>
            )}
            {zkMode === 'risk' && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
              >
                <label className="label-text block mb-1.5">Min Workers for MPC</label>
                <input
                  type="number"
                  value={minWorkersForMpc}
                  onChange={(e) => setMinWorkersForMpc(Number(e.target.value))}
                  min={2}
                  max={workersOnline > 2 ? workersOnline : 7}
                  className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Payment fields */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label-text block mb-1.5">Payment (ETH)</label>
              <input
                type="number"
                value={paymentEth}
                onChange={(e) => setPaymentEth(Number(e.target.value))}
                step={0.1}
                min={0}
                className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
              />
            </div>
            <div>
              <label className="label-text block mb-1.5">Stake / Worker (ETH)</label>
              <input
                type="number"
                value={stakePerWorkerEth}
                onChange={(e) => setStakePerWorkerEth(Number(e.target.value))}
                step={0.01}
                min={0}
                className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
              />
            </div>
          </div>

          {/* Simulate Cheater */}
          <div className="flex items-center gap-3 py-2">
            <button
              type="button"
              onClick={() => setSimulateCheater(!simulateCheater)}
              className={cn(
                'relative w-9 h-5 rounded-full transition-colors',
                simulateCheater ? 'bg-white' : 'bg-helix-border',
              )}
            >
              <motion.div
                className={cn(
                  'absolute top-0.5 w-4 h-4 rounded-full',
                  simulateCheater ? 'bg-black' : 'bg-helix-muted',
                )}
                animate={{ left: simulateCheater ? 18 : 2 }}
                transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              />
            </button>
            <div>
              <span className="text-sm text-helix-text">Simulate Cheater</span>
              <p className="text-2xs text-helix-muted">Demo: inject a malicious worker</p>
            </div>
          </div>

          {/* Store on 0G */}
          <div className="flex items-center gap-3 py-2">
            <button
              type="button"
              onClick={() => setStoreOn0G(!storeOn0G)}
              className={cn(
                'relative w-9 h-5 rounded-full transition-colors',
                storeOn0G ? 'bg-white' : 'bg-helix-border',
              )}
            >
              <motion.div
                className={cn(
                  'absolute top-0.5 w-4 h-4 rounded-full',
                  storeOn0G ? 'bg-black' : 'bg-helix-muted',
                )}
                animate={{ left: storeOn0G ? 18 : 2 }}
                transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              />
            </button>
            <div>
              <span className="text-sm text-helix-text">Store on 0G</span>
              <p className="text-2xs text-helix-muted">Save trained model to 0G decentralized storage</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Start Button */}
      <div className="lg:col-span-2">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={isStarting || workersOnline < 2 || !!versionError}
          className={cn(
            'w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium text-sm transition-all',
            (isStarting || workersOnline < 2 || !!versionError)
              ? 'bg-helix-border text-helix-muted cursor-not-allowed'
              : 'bg-white text-black hover:bg-white/90',
          )}
        >
          {isStarting ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Starting Training...
            </>
          ) : workersOnline < 2 ? (
            <>
              <AlertTriangle size={16} />
              Waiting for Workers ({workersOnline}/2)
            </>
          ) : (
            <>
              <Play size={16} />
              Start Training ({workersOnline} workers)
            </>
          )}
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Phase Indicator Component
// ============================================================================

interface PhaseIndicatorProps {
  currentPhase: number;
  description: string;
}

function PhaseIndicator({ currentPhase, description }: PhaseIndicatorProps) {
  return (
    <Card variant="glass">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-white">Phase Progress</h3>
        <Badge variant="pulse">
          Phase {currentPhase} / {TOTAL_PHASES}
        </Badge>
      </div>
      <div className="flex gap-1 mb-3">
        {Array.from({ length: TOTAL_PHASES }, (_, i) => {
          const phase = i + 1;
          const isComplete = phase < currentPhase;
          const isCurrent = phase === currentPhase;
          return (
            <div
              key={phase}
              className={cn(
                'flex-1 h-1.5 rounded-full transition-all duration-300',
                isComplete ? 'bg-white' : isCurrent ? 'bg-white/60' : 'bg-helix-border',
              )}
            />
          );
        })}
      </div>
      <p className="text-sm text-helix-text2">
        {description || PHASE_DESCRIPTIONS[currentPhase] || 'Processing...'}
      </p>
    </Card>
  );
}

// ============================================================================
// Stat Tiles
// ============================================================================

interface StatTileProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  highlight?: boolean;
}

function StatTile({ label, value, icon, highlight }: StatTileProps) {
  return (
    <div
      className={cn(
        'bg-helix-surface border border-helix-border rounded-lg p-4',
        highlight && 'border-white/20',
      )}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-helix-dim">{icon}</span>
        <span className="label-text">{label}</span>
      </div>
      <p className="text-xl font-light tracking-tight text-helix-text font-mono">{value}</p>
    </div>
  );
}

// ============================================================================
// Loss Curve Chart
// ============================================================================

interface LossCurveProps {
  data: { step: number; loss: number }[];
}

function LossCurve({ data }: LossCurveProps) {
  // Downsample for performance if needed
  const chartData = useMemo(() => {
    if (data.length <= 200) return data;
    const step = Math.ceil(data.length / 200);
    return data.filter((_, i) => i % step === 0 || i === data.length - 1);
  }, [data]);

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-[300px] text-helix-muted text-sm">
        Waiting for training data...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <defs>
          <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,255,255,0.08)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0)" />
          </linearGradient>
        </defs>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="#1e1e22"
          vertical={false}
        />
        <XAxis
          dataKey="step"
          stroke="#3e3e44"
          tick={{ fill: '#63636e', fontSize: 11, fontFamily: 'var(--font-geist-mono)' }}
          tickLine={false}
          axisLine={{ stroke: '#1e1e22' }}
        />
        <YAxis
          stroke="#3e3e44"
          tick={{ fill: '#63636e', fontSize: 11, fontFamily: 'var(--font-geist-mono)' }}
          tickLine={false}
          axisLine={false}
          width={52}
          tickFormatter={(v: number) => v.toFixed(2)}
        />
        <RechartsTooltip
          content={<ChartTooltip />}
          cursor={{ stroke: '#2a2a2e', strokeWidth: 1 }}
        />
        <Line
          type="monotone"
          dataKey="loss"
          name="Loss"
          stroke="#ffffff"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3, fill: '#ffffff', stroke: '#111113', strokeWidth: 2 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ============================================================================
// Cheater Alert
// ============================================================================

interface CheaterAlertProps {
  cheater: { party_index: number; step: number };
}

function CheaterAlert({ cheater }: CheaterAlertProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-red-500/10 border border-red-500/30 rounded-lg p-4"
    >
      <div className="flex items-center gap-3">
        <AlertTriangle size={20} className="text-red-400 shrink-0" />
        <div>
          <p className="text-sm font-medium text-red-300">Cheater Detected</p>
          <p className="text-2xs text-red-300/70 mt-0.5">
            Worker {cheater.party_index} submitted invalid MAC at step {cheater.step}.
            Stake slashed, training continues with remaining workers.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

// ============================================================================
// Final Results
// ============================================================================

interface FinalResultsProps {
  session: {
    session_id: string;
    accuracy: number | null;
    elapsed_secs: number;
    started_at: number;
    checkpoints_submitted: number;
    mac_checks_passed: number;
    zk_proofs_generated: number;
    cheater_detected: { party_index: number; step: number } | null;
    total_steps: number;
    current_step: number;
    status: string;
  };
  version: string;
  onDownloadModel?: (sessionId: string) => Promise<void>;
  onStoreOnZeroG?: (sessionId: string, version?: string) => Promise<void>;
  isStoringOnZeroG?: boolean;
  zeroGResult?: ZeroGStorageResult | null;
  showStoreOn0G?: boolean;
}

function FinalResults({ session, version, onDownloadModel, onStoreOnZeroG, isStoringOnZeroG, zeroGResult, showStoreOn0G }: FinalResultsProps) {
  const isSuccess = session.status === 'complete';
  const [copied, setCopied] = useState(false);

  const copyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      <Card variant="glass" className="overflow-visible">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            {isSuccess ? (
              <CheckCircle size={24} className="text-green-400" />
            ) : (
              <XCircle size={24} className="text-red-400" />
            )}
            <h3 className="text-lg font-semibold text-white">
              {isSuccess ? 'Training Complete' : 'Training Failed'}
            </h3>
            <Badge variant="default" className="font-mono">v{version}</Badge>
          </div>
          {isSuccess && onDownloadModel && (
            <button
              type="button"
              onClick={() => onDownloadModel(session.session_id)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
            >
              <Download size={14} />
              Download Model
            </button>
          )}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {session.accuracy !== null && (
            <div>
              <p className="label-text mb-1">Final Accuracy</p>
              <p className="text-2xl font-mono font-light text-white">
                {(session.accuracy * 100).toFixed(1)}%
              </p>
            </div>
          )}
          <div>
            <p className="label-text mb-1">Time</p>
            <p className="text-2xl font-mono font-light text-white">
              {session.elapsed_secs > 0
                ? `${session.elapsed_secs.toFixed(1)}s`
                : session.started_at > 0
                  ? `${((Date.now() / 1000) - session.started_at).toFixed(1)}s`
                  : '--'}
            </p>
          </div>
          <div>
            <p className="label-text mb-1">Steps Completed</p>
            <p className="text-2xl font-mono font-light text-white">
              {session.current_step} / {session.total_steps}
            </p>
          </div>
          <div>
            <p className="label-text mb-1">Checkpoints</p>
            <p className="text-2xl font-mono font-light text-white">
              {session.checkpoints_submitted}
            </p>
          </div>
        </div>
      </Card>

      {/* 0G Storage */}
      {isSuccess && showStoreOn0G && (
        <Card variant="default">
          <div className="flex items-center gap-2 mb-3">
            <HardDrive size={14} className="text-helix-text2" />
            <h4 className="text-sm font-medium text-white">0G Decentralized Storage</h4>
          </div>

          {zeroGResult ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <CheckCircle size={14} className="text-green-400 shrink-0" />
                <span className="text-sm text-green-300">Model stored on 0G Storage</span>
              </div>

              <div className="space-y-2">
                <div>
                  <p className="label-text mb-1">Root Hash</p>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-xs font-mono text-helix-text bg-helix-bg px-3 py-2 rounded-md border border-helix-border truncate">
                      {zeroGResult.rootHash}
                    </code>
                    <button
                      type="button"
                      onClick={() => copyHash(zeroGResult.rootHash)}
                      className="shrink-0 p-2 rounded-md bg-helix-bg border border-helix-border text-helix-muted hover:text-white transition-colors"
                    >
                      {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
                    </button>
                  </div>
                </div>

                <a
                  href={zeroGResult.explorerUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-xs text-helix-text2 hover:text-white transition-colors"
                >
                  View on 0G Explorer
                  <ExternalLink size={12} />
                </a>
              </div>
            </div>
          ) : (
            <div>
              {isStoringOnZeroG ? (
                <div className="flex items-center gap-3 py-2">
                  <Loader2 size={16} className="animate-spin text-helix-text2" />
                  <span className="text-sm text-helix-text2">Uploading model to 0G Storage...</span>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => onStoreOnZeroG?.(session.session_id, version)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-helix-surface border border-helix-border text-sm text-helix-text hover:border-helix-border2 hover:text-white transition-colors"
                >
                  <HardDrive size={14} />
                  Store Model on 0G
                </button>
              )}
            </div>
          )}
        </Card>
      )}
    </motion.div>
  );
}

// ============================================================================
// Live Progress Component
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

  return (
    <div className="space-y-5">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Wifi size={14} className="text-green-400" />
          ) : (
            <WifiOff size={14} className="text-helix-muted" />
          )}
          <span className="text-2xs text-helix-muted font-mono uppercase tracking-wider">
            {isConnected ? 'Connected' : 'Reconnecting...'}
          </span>
        </div>
        <Badge
          variant={session.status === 'running' ? 'pulse' : 'default'}
          className={cn(
            session.status === 'complete' && 'text-green-400',
            session.status === 'failed' && 'text-red-400',
          )}
        >
          {session.status}
        </Badge>
      </div>

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3"
        >
          <p className="text-sm text-red-300">{error}</p>
        </motion.div>
      )}

      {/* Phase Indicator */}
      <PhaseIndicator
        currentPhase={session.phase}
        description={session.phase_description}
      />

      {/* Step Progress */}
      <Card variant="default">
        <div className="flex items-center justify-between mb-2">
          <span className="label-text">Training Progress</span>
          <span className="text-sm font-mono text-helix-text">
            {session.current_step} / {session.total_steps}
          </span>
        </div>
        <ProgressBar value={stepProgress} size="md" />
        <p className="text-2xs text-helix-dim mt-2 text-right font-mono">
          {stepProgress.toFixed(1)}%
        </p>
      </Card>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatTile
          label="Current Loss"
          value={session.current_loss > 0 ? session.current_loss.toFixed(4) : '--'}
          icon={<Activity size={14} />}
        />
        <StatTile
          label="MAC Checks"
          value={session.mac_checks_passed}
          icon={<Shield size={14} />}
        />
        <StatTile
          label="Checkpoints"
          value={session.checkpoints_submitted}
          icon={<Zap size={14} />}
        />
        <StatTile
          label="Elapsed"
          value={
            session.elapsed_secs > 0
              ? `${session.elapsed_secs.toFixed(0)}s`
              : `${Math.floor((Date.now() / 1000) - session.started_at)}s`
          }
          icon={<Clock size={14} />}
        />
      </div>

      {/* ZK Proofs indicator */}
      {(session.zk_proofs_generated > 0 || session.zk_activated_by_risk) && (
        <Card variant="default">
          <div className="flex items-center gap-3">
            <Shield size={16} className="text-helix-text2" />
            <div>
              <p className="text-sm text-helix-text">
                ZK Proofs: {session.zk_proofs_generated} generated
              </p>
              {session.zk_activated_by_risk && (
                <p className="text-2xs text-helix-muted mt-0.5">
                  Activated by risk detection
                </p>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* Cheater Alert */}
      {session.cheater_detected && (
        <CheaterAlert cheater={session.cheater_detected} />
      )}

      {/* Loss Curve */}
      <Card variant="default">
        <h3 className="text-sm font-medium text-white mb-3">Loss Curve</h3>
        <LossCurve data={losses} />
      </Card>

      {/* Final Results */}
      {isTerminal && (
        <FinalResults
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
  );
}

// ============================================================================
// Training History Component
// ============================================================================

interface TrainingHistoryListProps {
  history: TrainingHistoryEntry[];
  onClearHistory: () => void;
}

function TrainingHistoryList({ history, onClearHistory }: TrainingHistoryListProps) {
  if (history.length === 0) return null;

  return (
    <Card variant="default">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <History size={14} className="text-helix-text2" />
          <h3 className="text-sm font-medium text-white">Training History</h3>
          <Badge variant="default">{history.length}</Badge>
        </div>
        <button
          type="button"
          onClick={onClearHistory}
          className="text-2xs text-helix-muted hover:text-red-400 transition-colors"
        >
          Clear
        </button>
      </div>

      <div className="space-y-2">
        {history.slice().reverse().map((entry) => (
          <div
            key={entry.sessionId}
            className="flex items-center gap-3 px-3 py-2.5 bg-helix-bg border border-helix-border rounded-lg"
          >
            {entry.status === 'complete' ? (
              <CheckCircle size={14} className="text-green-400 shrink-0" />
            ) : (
              <XCircle size={14} className="text-red-400 shrink-0" />
            )}

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <Badge variant="default" className="font-mono text-2xs">v{entry.version}</Badge>
                <span className="text-xs font-mono text-helix-muted truncate">
                  {entry.sessionId.slice(0, 8)}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-4 shrink-0">
              {entry.accuracy !== null && (
                <span className="text-xs font-mono text-helix-text">
                  {(entry.accuracy * 100).toFixed(1)}%
                </span>
              )}
              <span className="text-xs font-mono text-helix-muted">
                {entry.steps}/{entry.totalSteps}
              </span>
              {entry.storedOn0G ? (
                <span className="flex items-center gap-1 text-2xs text-green-400">
                  <HardDrive size={10} />
                  0G
                </span>
              ) : (
                <span className="text-2xs text-helix-dim">Local only</span>
              )}
              <span className="text-2xs text-helix-dim">
                {new Date(entry.date).toLocaleDateString()}
              </span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function TrainPage() {
  const {
    startTraining,
    uploadData,
    uploadWeights,
    downloadModel,
    storeOnZeroG,
    session,
    losses,
    isConnected,
    isStarting,
    error,
    uploadedData,
    uploadedWeights,
    workersOnline,
    zeroGResult,
    isStoringOnZeroG,
  } = useMpcTraining();

  const hasSession = session !== null;
  const [wantsStoreOn0G, setWantsStoreOn0G] = useState(false);
  const [currentVersion, setCurrentVersion] = useState('1.0.0');
  const [history, setHistory] = useState<TrainingHistoryEntry[]>([]);
  const historyRecordedRef = useRef(false);

  // Load history from localStorage on mount
  useEffect(() => {
    const h = getTrainingHistory();
    setHistory(h);
    setCurrentVersion(getNextVersion(h));
  }, []);

  // Record to history when session reaches terminal state
  useEffect(() => {
    if (!session) {
      historyRecordedRef.current = false;
      return;
    }
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

  // Update history entry when 0G storage completes
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <h1 className="page-title">Train</h1>
        {hasSession && (
          <div className="flex items-center gap-3">
            <Badge variant="default" className="font-mono">v{currentVersion}</Badge>
            <span className="text-2xs font-mono text-helix-muted">
              Session: {session.session_id.slice(0, 8)}...
            </span>
          </div>
        )}
      </div>

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
        />
      ) : (
        <LiveProgress
          session={session}
          losses={losses}
          isConnected={isConnected}
          error={error}
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
