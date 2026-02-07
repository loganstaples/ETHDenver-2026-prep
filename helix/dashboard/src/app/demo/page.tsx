'use client';

/**
 * HELIX Demo Page
 * Guided tour and demonstration of the HELIX training dashboard.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Play,
    Pause,
    RefreshCw,
    ChevronRight,
    ChevronLeft,
    X,
    Zap,
    Shield,
    Activity,
    Server,
    AlertTriangle,
    CheckCircle,
    Globe,
    TrendingDown,
    ExternalLink,
    Sparkles,
    Eye,
} from 'lucide-react';
import LiveLossCurve from '@/components/training/LiveLossCurve';
import ProofStream from '@/components/proof/ProofStream';
import LiveTopology from '@/components/network/LiveTopology';
import ErrorBoundsViz from '@/components/training/ErrorBoundsViz';
import { useTrainingStatus, type TrainingAlert } from '@/hooks/useTrainingStatus';
import { useProofStream } from '@/hooks/useProofStream';
import { useWorkerHealth } from '@/hooks/useWorkerHealth';

// ============================================================================
// Types
// ============================================================================

interface TourStep {
    id: string;
    title: string;
    description: string;
    target: 'training' | 'proofs' | 'network' | 'overview' | 'adversarial';
    highlight?: string;
    action?: () => void;
}

// ============================================================================
// Tour Steps Configuration
// ============================================================================

const TOUR_STEPS: TourStep[] = [
    {
        id: 'welcome',
        title: 'Welcome to HELIX',
        description: 'HELIX enables trustless AI training on decentralized compute. This demo shows real-time training with zero-knowledge proof verification. Watch as gradients are computed, proofs are generated, and workers are coordinated.',
        target: 'overview',
    },
    {
        id: 'training',
        title: 'Live Training Progress',
        description: 'Monitor training metrics in real-time. The loss curve shows model improvement, while accuracy tracks performance. Every computation is bounded by error guarantees tracked through the approximate VM.',
        target: 'training',
        highlight: 'training-section',
    },
    {
        id: 'loss',
        title: 'Loss & Convergence',
        description: 'Watch the loss decrease as training progresses. The convergence rate shows how quickly the model is learning. Error bounds ensure numerical approximations stay within acceptable limits.',
        target: 'training',
    },
    {
        id: 'proofs',
        title: 'ZK Proof Generation',
        description: 'Every training step generates a zero-knowledge proof. Watch proofs move through stages: witness generation, circuit setup, proving, and on-chain verification.',
        target: 'proofs',
        highlight: 'proofs-section',
    },
    {
        id: 'verification',
        title: 'On-Chain Verification',
        description: 'Proofs are verified on-chain using Halo2 circuits. Invalid proofs trigger automatic slashing. This ensures workers cannot submit fake computations.',
        target: 'proofs',
    },
    {
        id: 'network',
        title: 'Network Topology',
        description: 'See the decentralized network of compute workers, aggregators, and verifiers. Each node shows real-time health metrics. The MPC protocol ensures no single node sees the full model.',
        target: 'network',
        highlight: 'network-section',
    },
    {
        id: 'workers',
        title: 'Worker Health',
        description: 'Monitor CPU, GPU, memory, and network metrics for each worker. Unhealthy nodes are automatically detected. Click any node to see detailed performance data.',
        target: 'network',
    },
    {
        id: 'adversarial',
        title: 'Security Guarantees',
        description: 'HELIX detects and punishes malicious behavior. Invalid proofs, byzantine gradients, and timeouts all result in stake slashing. Economic incentives keep the network honest.',
        target: 'adversarial',
        highlight: 'adversarial-section',
    },
    {
        id: 'conclusion',
        title: 'Trustless AI Training',
        description: 'You\'ve seen how HELIX enables verifiable private AI training. Every computation proven. Model weights hidden. Workers economically aligned. The future of decentralized AI.',
        target: 'overview',
    },
];

// ============================================================================
// Sub-Components
// ============================================================================

function TourOverlay({
    step,
    currentIndex,
    totalSteps,
    onNext,
    onPrev,
    onSkip,
    onClose,
}: {
    step: TourStep;
    currentIndex: number;
    totalSteps: number;
    onNext: () => void;
    onPrev: () => void;
    onSkip: () => void;
    onClose: () => void;
}) {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 pointer-events-none"
        >
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/50 pointer-events-auto" onClick={onClose} />

            {/* Tour card */}
            <motion.div
                initial={{ y: 50, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 50, opacity: 0 }}
                className="absolute bottom-8 left-1/2 -translate-x-1/2 w-full max-w-xl bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl pointer-events-auto"
            >
                <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                                <Sparkles className="w-5 h-5 text-emerald-400" />
                            </div>
                            <div>
                                <h3 className="font-semibold text-white text-lg">{step.title}</h3>
                                <p className="text-sm text-neutral-400">Step {currentIndex + 1} of {totalSteps}</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-1 text-neutral-400 hover:text-white transition-colors"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    <p className="text-neutral-300 mb-6 leading-relaxed">
                        {step.description}
                    </p>

                    {/* Progress bar */}
                    <div className="w-full h-1 bg-neutral-800 rounded-full mb-6 overflow-hidden">
                        <motion.div
                            className="h-full bg-emerald-500 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${((currentIndex + 1) / totalSteps) * 100}%` }}
                            transition={{ duration: 0.3 }}
                        />
                    </div>

                    {/* Navigation */}
                    <div className="flex items-center justify-between">
                        <button
                            onClick={onSkip}
                            className="text-sm text-neutral-400 hover:text-white transition-colors"
                        >
                            Skip tour
                        </button>

                        <div className="flex items-center gap-3">
                            <button
                                onClick={onPrev}
                                disabled={currentIndex === 0}
                                className="flex items-center gap-1 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium text-white transition-colors"
                            >
                                <ChevronLeft className="w-4 h-4" />
                                Back
                            </button>
                            <button
                                onClick={onNext}
                                className="flex items-center gap-1 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm font-medium text-white transition-colors"
                            >
                                {currentIndex === totalSteps - 1 ? 'Finish' : 'Next'}
                                {currentIndex < totalSteps - 1 && <ChevronRight className="w-4 h-4" />}
                            </button>
                        </div>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
}

function DemoControls({
    isPlaying,
    onTogglePlay,
    onReset,
    onStartTour,
    speed,
    onSpeedChange,
}: {
    isPlaying: boolean;
    onTogglePlay: () => void;
    onReset: () => void;
    onStartTour: () => void;
    speed: number;
    onSpeedChange: (speed: number) => void;
}) {
    return (
        <div className="flex items-center gap-4 bg-neutral-800/50 rounded-lg p-2">
            <button
                onClick={onTogglePlay}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isPlaying
                        ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30'
                        : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30'
                }`}
            >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isPlaying ? 'Pause' : 'Play'}
            </button>

            <button
                onClick={onReset}
                className="flex items-center gap-2 px-4 py-2 bg-neutral-700/50 hover:bg-neutral-600/50 rounded-lg text-sm font-medium text-neutral-300 transition-colors"
            >
                <RefreshCw className="w-4 h-4" />
                Reset
            </button>

            <div className="h-6 w-px bg-neutral-700" />

            <div className="flex items-center gap-2">
                <span className="text-sm text-neutral-400">Speed:</span>
                <select
                    value={speed}
                    onChange={(e) => onSpeedChange(Number(e.target.value))}
                    className="bg-neutral-700 text-white text-sm rounded px-2 py-1 border-none outline-none"
                >
                    <option value={0.5}>0.5x</option>
                    <option value={1}>1x</option>
                    <option value={2}>2x</option>
                    <option value={4}>4x</option>
                </select>
            </div>

            <div className="h-6 w-px bg-neutral-700" />

            <button
                onClick={onStartTour}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30 rounded-lg text-sm font-medium transition-colors"
            >
                <Eye className="w-4 h-4" />
                Guided Tour
            </button>
        </div>
    );
}

function StatsOverview({
    training,
    proofs,
    workers,
}: {
    training: ReturnType<typeof useTrainingStatus>;
    proofs: ReturnType<typeof useProofStream>;
    workers: ReturnType<typeof useWorkerHealth>;
}) {
    return (
        <div className="grid grid-cols-4 gap-4">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-neutral-800/50 rounded-xl border border-neutral-700 p-4"
            >
                <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                        <Activity className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div>
                        <div className="text-sm text-neutral-400">Training Progress</div>
                        <div className="text-xl font-bold text-white">
                            {training.status?.progress.toFixed(1)}%
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-2 text-sm">
                    <TrendingDown className="w-4 h-4 text-emerald-400" />
                    <span className="text-neutral-400">Loss:</span>
                    <span className="text-white font-medium">{training.metrics?.loss.toFixed(4)}</span>
                </div>
            </motion.div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-neutral-800/50 rounded-xl border border-neutral-700 p-4"
            >
                <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-indigo-500/20 flex items-center justify-center">
                        <Shield className="w-5 h-5 text-indigo-400" />
                    </div>
                    <div>
                        <div className="text-sm text-neutral-400">Proofs Verified</div>
                        <div className="text-xl font-bold text-white">{proofs.stats.verified}</div>
                    </div>
                </div>
                <div className="flex items-center gap-2 text-sm">
                    <Zap className="w-4 h-4 text-indigo-400" />
                    <span className="text-neutral-400">Generating:</span>
                    <span className="text-white font-medium">{proofs.stats.generating}</span>
                </div>
            </motion.div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-neutral-800/50 rounded-xl border border-neutral-700 p-4"
            >
                <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-cyan-500/20 flex items-center justify-center">
                        <Globe className="w-5 h-5 text-cyan-400" />
                    </div>
                    <div>
                        <div className="text-sm text-neutral-400">Network Health</div>
                        <div className="text-xl font-bold text-white">
                            {workers.networkSummary.healthScore}%
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-2 text-sm">
                    <Server className="w-4 h-4 text-cyan-400" />
                    <span className="text-neutral-400">Active:</span>
                    <span className="text-white font-medium">
                        {workers.networkSummary.healthyWorkers}/{workers.networkSummary.totalWorkers}
                    </span>
                </div>
            </motion.div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-neutral-800/50 rounded-xl border border-neutral-700 p-4"
            >
                <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center">
                        <AlertTriangle className="w-5 h-5 text-amber-400" />
                    </div>
                    <div>
                        <div className="text-sm text-neutral-400">Security Alerts</div>
                        <div className="text-xl font-bold text-white">
                            {training.unacknowledgedAlerts.length}
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    <span className="text-neutral-400">Success Rate:</span>
                    <span className="text-white font-medium">
                        {(proofs.stats.successRate * 100).toFixed(1)}%
                    </span>
                </div>
            </motion.div>
        </div>
    );
}

function AdversarialSection({ training }: { training: ReturnType<typeof useTrainingStatus> }) {
    const [simulatingAttack, setSimulatingAttack] = useState(false);
    const [attackEvents, setAttackEvents] = useState<TrainingAlert[]>([]);

    const simulateAttack = useCallback(() => {
        setSimulatingAttack(true);
        // Add a simulated attack event
        const attackEvent: TrainingAlert = {
            id: `attack-${Date.now()}`,
            type: 'error',
            title: 'Invalid Proof Detected',
            message: 'Worker 0x742d...1231 submitted malicious gradient - SLASHING',
            timestamp: Date.now(),
            acknowledged: false,
            data: { prover: '0x742d35Cc6634C0532925a3b844Bc9e7595f01231', slashAmount: '0.5 HELIX' },
        };
        setAttackEvents(prev => [attackEvent, ...prev].slice(0, 10));
        setTimeout(() => setSimulatingAttack(false), 3000);
    }, []);

    // Combine alerts with simulated attack events
    const allEvents = useMemo(() => {
        return [...attackEvents, ...training.alerts].slice(0, 10);
    }, [attackEvents, training.alerts]);

    return (
        <div id="adversarial-section" className="bg-neutral-800/50 rounded-xl border border-neutral-700 p-4">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-red-500/20 flex items-center justify-center">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-white">Security Events</h3>
                        <p className="text-sm text-neutral-400">Adversarial detection & slashing</p>
                    </div>
                </div>

                <button
                    onClick={simulateAttack}
                    disabled={simulatingAttack}
                    className="flex items-center gap-2 px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 disabled:opacity-50 rounded-lg text-sm font-medium text-red-400 transition-colors"
                >
                    {simulatingAttack ? (
                        <>
                            <div className="w-4 h-4 border-2 border-red-400 border-t-transparent rounded-full animate-spin" />
                            Detecting...
                        </>
                    ) : (
                        <>
                            <Zap className="w-4 h-4" />
                            Simulate Attack
                        </>
                    )}
                </button>
            </div>

            <div className="space-y-2 max-h-48 overflow-y-auto">
                <AnimatePresence mode="popLayout">
                    {simulatingAttack && (
                        <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                            className="flex items-center gap-3 bg-red-500/10 border border-red-500/30 rounded-lg p-3"
                        >
                            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                            <div className="flex-1">
                                <div className="text-sm font-medium text-red-400">Invalid Proof Detected</div>
                                <div className="text-xs text-neutral-400">Worker 0x742d...1231 submitted malicious gradient</div>
                            </div>
                            <div className="text-xs text-red-400 font-mono">SLASHING...</div>
                        </motion.div>
                    )}

                    {allEvents.slice(0, 5).map((event) => {
                        const isError = event.type === 'error';
                        const isWarning = event.type === 'warning';
                        return (
                            <motion.div
                                key={event.id}
                                layout
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className={`flex items-center gap-3 rounded-lg p-3 ${
                                    isError
                                        ? 'bg-red-500/10 border border-red-500/20'
                                        : isWarning
                                            ? 'bg-amber-500/10 border border-amber-500/20'
                                            : 'bg-emerald-500/10 border border-emerald-500/20'
                                }`}
                            >
                                <div className={`w-2 h-2 rounded-full ${
                                    isError
                                        ? 'bg-red-500'
                                        : isWarning
                                            ? 'bg-amber-500'
                                            : 'bg-emerald-500'
                                }`} />
                                <div className="flex-1 min-w-0">
                                    <div className={`text-sm font-medium truncate ${
                                        isError
                                            ? 'text-red-400'
                                            : isWarning
                                                ? 'text-amber-400'
                                                : 'text-emerald-400'
                                    }`}>
                                        {event.title}: {event.message}
                                    </div>
                                    <div className="text-xs text-neutral-500">
                                        {new Date(event.timestamp).toLocaleTimeString()}
                                    </div>
                                </div>
                                {event.acknowledged && (
                                    <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                                )}
                            </motion.div>
                        );
                    })}
                </AnimatePresence>

                {allEvents.length === 0 && !simulatingAttack && (
                    <div className="text-center py-4 text-neutral-500 text-sm">
                        No security events detected
                    </div>
                )}
            </div>
        </div>
    );
}

// ============================================================================
// Main Component
// ============================================================================

export default function DemoPage() {
    const [isPlaying, setIsPlaying] = useState(true);
    const [speed, setSpeed] = useState(1);
    const [showTour, setShowTour] = useState(false);
    const [tourStep, setTourStep] = useState(0);
    const [hasSeenTour, setHasSeenTour] = useState(false);

    const training = useTrainingStatus({
        modelId: BigInt(1),
        enablePolling: isPlaying,
        pollingInterval: 2000 / speed,
    });

    const proofs = useProofStream({
        modelId: BigInt(1),
        enableSimulation: isPlaying,
        simulationInterval: 1000 / speed,
    });

    const workers = useWorkerHealth({
        modelId: BigInt(1),
        enablePolling: isPlaying,
        pollingInterval: 3000 / speed,
    });

    // Auto-start tour for first-time visitors
    useEffect(() => {
        const toured = localStorage.getItem('helix-demo-toured');
        if (!toured && !hasSeenTour) {
            const timer = setTimeout(() => {
                setShowTour(true);
                setHasSeenTour(true);
            }, 2000);
            return () => clearTimeout(timer);
        }
    }, [hasSeenTour]);

    const handleNextStep = useCallback(() => {
        if (tourStep < TOUR_STEPS.length - 1) {
            setTourStep(tourStep + 1);
        } else {
            setShowTour(false);
            localStorage.setItem('helix-demo-toured', 'true');
        }
    }, [tourStep]);

    const handlePrevStep = useCallback(() => {
        if (tourStep > 0) {
            setTourStep(tourStep - 1);
        }
    }, [tourStep]);

    const handleSkipTour = useCallback(() => {
        setShowTour(false);
        localStorage.setItem('helix-demo-toured', 'true');
    }, []);

    const handleStartTour = useCallback(() => {
        setTourStep(0);
        setShowTour(true);
    }, []);

    const handleReset = useCallback(() => {
        training.refresh();
        proofs.refresh();
        workers.refresh();
    }, [training, proofs, workers]);

    const currentStep = TOUR_STEPS[tourStep];

    return (
        <div className="min-h-screen bg-neutral-950 text-white">
            {/* Header */}
            <header className="sticky top-0 z-40 bg-neutral-950/80 backdrop-blur-xl border-b border-neutral-800">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
                                    <Zap className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h1 className="font-bold text-xl bg-gradient-to-r from-emerald-400 to-cyan-400 text-transparent bg-clip-text">
                                        HELIX Demo
                                    </h1>
                                    <p className="text-xs text-neutral-400">Trustless AI Training</p>
                                </div>
                            </div>

                            <div className="h-8 w-px bg-neutral-800" />

                            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/20 rounded-full">
                                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                                <span className="text-sm font-medium text-emerald-400">Live Demo</span>
                            </div>
                        </div>

                        <DemoControls
                            isPlaying={isPlaying}
                            onTogglePlay={() => setIsPlaying(!isPlaying)}
                            onReset={handleReset}
                            onStartTour={handleStartTour}
                            speed={speed}
                            onSpeedChange={setSpeed}
                        />
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
                {/* Stats Overview */}
                <StatsOverview training={training} proofs={proofs} workers={workers} />

                {/* Training Section */}
                <section id="training-section" className={`${
                    showTour && (currentStep?.target === 'training') ? 'ring-2 ring-emerald-500 ring-offset-4 ring-offset-neutral-950 rounded-xl' : ''
                }`}>
                    <div className="grid grid-cols-3 gap-6">
                        <div className="col-span-2">
                            <LiveLossCurve
                                modelId={BigInt(1)}
                                height={350}
                                showAccuracy={true}
                                showErrorBound={true}
                            />
                        </div>
                        <div>
                            <ErrorBoundsViz
                                modelId={BigInt(1)}
                                showDetails={false}
                                showTimeline={true}
                                compact={false}
                            />
                        </div>
                    </div>
                </section>

                {/* Proofs and Network Grid */}
                <div className="grid grid-cols-2 gap-6">
                    {/* Proofs */}
                    <section id="proofs-section" className={`${
                        showTour && (currentStep?.target === 'proofs') ? 'ring-2 ring-indigo-500 ring-offset-4 ring-offset-neutral-950 rounded-xl' : ''
                    }`}>
                        <ProofStream
                            modelId={BigInt(1)}
                            maxDisplay={6}
                            showStats={true}
                            compact={true}
                        />
                    </section>

                    {/* Network */}
                    <section id="network-section" className={`${
                        showTour && (currentStep?.target === 'network') ? 'ring-2 ring-cyan-500 ring-offset-4 ring-offset-neutral-950 rounded-xl' : ''
                    }`}>
                        <LiveTopology
                            modelId={BigInt(1)}
                            height={500}
                            showDetails={true}
                        />
                    </section>
                </div>

                {/* Adversarial Section */}
                <section id="adversarial-section" className={`${
                    showTour && (currentStep?.target === 'adversarial') ? 'ring-2 ring-red-500 ring-offset-4 ring-offset-neutral-950 rounded-xl' : ''
                }`}>
                    <AdversarialSection training={training} />
                </section>

                {/* Footer CTA */}
                <div className="bg-gradient-to-r from-emerald-500/10 via-cyan-500/10 to-indigo-500/10 rounded-xl border border-neutral-800 p-8 text-center">
                    <h2 className="text-2xl font-bold mb-3">
                        <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-indigo-400 text-transparent bg-clip-text">
                            Train AI on Untrusted Hardware
                        </span>
                    </h2>
                    <p className="text-neutral-400 max-w-2xl mx-auto mb-6">
                        Model stays private. Every computation verified. Workers economically aligned.
                        HELIX enables trustless AI training at 30x overhead vs 10,000x for exact proofs.
                    </p>
                    <div className="flex items-center justify-center gap-4">
                        <a
                            href="https://github.com/helix-protocol/helix"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-6 py-3 bg-white/10 hover:bg-white/20 rounded-lg font-medium transition-colors"
                        >
                            View on GitHub
                            <ExternalLink className="w-4 h-4" />
                        </a>
                        <button
                            onClick={handleStartTour}
                            className="flex items-center gap-2 px-6 py-3 bg-emerald-600 hover:bg-emerald-500 rounded-lg font-medium transition-colors"
                        >
                            <Eye className="w-4 h-4" />
                            Take the Tour
                        </button>
                    </div>
                </div>
            </main>

            {/* Tour Overlay */}
            <AnimatePresence>
                {showTour && currentStep && (
                    <TourOverlay
                        step={currentStep}
                        currentIndex={tourStep}
                        totalSteps={TOUR_STEPS.length}
                        onNext={handleNextStep}
                        onPrev={handlePrevStep}
                        onSkip={handleSkipTour}
                        onClose={() => setShowTour(false)}
                    />
                )}
            </AnimatePresence>
        </div>
    );
}
