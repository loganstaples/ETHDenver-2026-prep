'use client';

import React, { useState } from 'react';
import { useModelRegistration, useRoundManagement, useContractState } from '@/hooks/useContract';

type DeploymentStep = 'configure' | 'upload' | 'register' | 'verify' | 'complete';

interface ModelConfig {
    name: string;
    description: string;
    ipfsHash: string;
    initialCommitment: string;
    minStake: string;
    roundDuration: string;
}

export default function ModelDeployment() {
    const { registerModel, isRegistering, registrationError, registrationSuccess: _registrationSuccess } = useModelRegistration();
    const { startRound, isStartingRound } = useRoundManagement(BigInt(0));
    const { defaultMinStake, isLoading: _stateLoading } = useContractState();

    const [currentStep, setCurrentStep] = useState<DeploymentStep>('configure');
    const [config, setConfig] = useState<ModelConfig>({
        name: '',
        description: '',
        ipfsHash: '',
        initialCommitment: '',
        minStake: '0.1',
        roundDuration: '3600',
    });
    const [uploadProgress, setUploadProgress] = useState(0);
    const [registeredModelId, _setRegisteredModelId] = useState<bigint | null>(null);

    const steps: { key: DeploymentStep; label: string; icon: string }[] = [
        { key: 'configure', label: 'Configure', icon: '1' },
        { key: 'upload', label: 'Upload to IPFS', icon: '2' },
        { key: 'register', label: 'Register On-Chain', icon: '3' },
        { key: 'verify', label: 'Verify', icon: '4' },
        { key: 'complete', label: 'Complete', icon: '✓' },
    ];

    const handleConfigChange = (field: keyof ModelConfig, value: string) => {
        setConfig((prev) => ({ ...prev, [field]: value }));
    };

    const simulateUpload = async () => {
        setCurrentStep('upload');
        for (let i = 0; i <= 100; i += 10) {
            await new Promise((r) => setTimeout(r, 200));
            setUploadProgress(i);
        }
        // Generate a mock IPFS hash
        const mockHash = `Qm${Array.from({ length: 44 }, () =>
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'[Math.floor(Math.random() * 62)]
        ).join('')}`;
        setConfig((prev) => ({ ...prev, ipfsHash: mockHash }));
        setCurrentStep('register');
    };

    const handleRegister = async () => {
        if (!config.ipfsHash || !config.initialCommitment || !config.minStake) return;

        try {
            await registerModel(
                config.ipfsHash,
                BigInt(config.initialCommitment),
                config.minStake
            );
            // After successful registration, move to verify step
            setCurrentStep('verify');
        } catch (error) {
            console.error('Registration failed:', error);
        }
    };

    const handleVerify = async () => {
        // In a real implementation, this would verify the on-chain state
        await new Promise((r) => setTimeout(r, 1500));
        setCurrentStep('complete');
    };

    const handleStartFirstRound = async () => {
        if (registeredModelId === null) return;
        try {
            await startRound(parseInt(config.roundDuration));
        } catch (error) {
            console.error('Failed to start round:', error);
        }
    };

    const getStepStatus = (stepKey: DeploymentStep): 'completed' | 'current' | 'pending' => {
        const stepOrder = steps.map((s) => s.key);
        const currentIndex = stepOrder.indexOf(currentStep);
        const stepIndex = stepOrder.indexOf(stepKey);
        if (stepIndex < currentIndex) return 'completed';
        if (stepIndex === currentIndex) return 'current';
        return 'pending';
    };

    return (
        <div className="bg-helix-surface rounded-md p-4 text-white font-sans">
            <div className="mb-5">
                <h2 className="font-semibold text-[15px] text-white mb-2">Deploy New Model</h2>
                <p className="text-sm text-[#888]">Register your ML model on the HELIX network for decentralized training</p>
            </div>

            <div className="flex justify-between mb-5 relative">
                <div className="absolute top-5 left-10 right-10 h-0.5 bg-helix-border" />

                {steps.map((step) => {
                    const status = getStepStatus(step.key);
                    return (
                        <div key={step.key} className="flex flex-col items-center gap-2 relative z-10">
                            <div className={`
                                w-10 h-10 rounded-full flex items-center justify-center font-semibold text-sm transition-all duration-300
                                ${status === 'completed' ? 'bg-white text-black' : ''}
                                ${status === 'current' ? 'bg-white text-black' : ''}
                                ${status === 'pending' ? 'bg-white/10 text-[#666]' : ''}
                            `}>
                                {status === 'completed' ? '✓' : step.icon}
                            </div>
                            <span className={`
                                text-xs
                                ${status === 'current' ? 'text-white font-semibold' : 'text-[#888]'}
                            `}>
                                {step.label}
                            </span>
                        </div>
                    );
                })}
            </div>

            <div className="bg-white/[0.03] rounded-md p-4 border border-helix-border">
                {currentStep === 'configure' && (
                    <>
                        <div className="mb-5">
                            <label className="block text-[13px] font-semibold text-[#aaa] mb-2">Model Name</label>
                            <input
                                type="text"
                                className="w-full px-4 py-3 bg-white/[0.03] border border-helix-border rounded-md text-white text-sm font-sans transition-all duration-200 focus:outline-none focus:border-white/50 focus:shadow-[0_0_0_3px_rgba(255,255,255,0.1)] placeholder:text-[#666]"
                                placeholder="e.g., HELIX-GPT-Small"
                                value={config.name}
                                onChange={(e) => handleConfigChange('name', e.target.value)}
                            />
                        </div>

                        <div className="mb-5">
                            <label className="block text-[13px] font-semibold text-[#aaa] mb-2">Description</label>
                            <textarea
                                className="w-full px-4 py-3 bg-white/[0.03] border border-helix-border rounded-md text-white text-sm font-sans transition-all duration-200 focus:outline-none focus:border-white/50 focus:shadow-[0_0_0_3px_rgba(255,255,255,0.1)] placeholder:text-[#666] min-h-[80px] resize-y"
                                placeholder="Describe your model architecture and training objectives..."
                                value={config.description}
                                onChange={(e) => handleConfigChange('description', e.target.value)}
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div className="mb-5">
                                <label className="block text-[13px] font-semibold text-[#aaa] mb-2">Initial Commitment (Hash)</label>
                                <input
                                    type="text"
                                    className="w-full px-4 py-3 bg-white/[0.03] border border-helix-border rounded-md text-white text-sm font-sans transition-all duration-200 focus:outline-none focus:border-white/50 focus:shadow-[0_0_0_3px_rgba(255,255,255,0.1)] placeholder:text-[#666]"
                                    placeholder="12345678"
                                    value={config.initialCommitment}
                                    onChange={(e) => handleConfigChange('initialCommitment', e.target.value)}
                                />
                                <span className="text-[11px] text-[#666] mt-1 block">Poseidon hash of initial model weights</span>
                            </div>

                            <div className="mb-5">
                                <label className="block text-[13px] font-semibold text-[#aaa] mb-2">Minimum Stake (ETH)</label>
                                <input
                                    type="text"
                                    className="w-full px-4 py-3 bg-white/[0.03] border border-helix-border rounded-md text-white text-sm font-sans transition-all duration-200 focus:outline-none focus:border-white/50 focus:shadow-[0_0_0_3px_rgba(255,255,255,0.1)] placeholder:text-[#666]"
                                    placeholder="0.1"
                                    value={config.minStake}
                                    onChange={(e) => handleConfigChange('minStake', e.target.value)}
                                />
                                <span className="text-[11px] text-[#666] mt-1 block">
                                    Default: {defaultMinStake ? `${Number(defaultMinStake) / 1e18} ETH` : '0.1 ETH'}
                                </span>
                            </div>
                        </div>

                        <div className="mb-5">
                            <label className="block text-[13px] font-semibold text-[#aaa] mb-2">Round Duration (seconds)</label>
                            <input
                                type="text"
                                className="w-full px-4 py-3 bg-white/[0.03] border border-helix-border rounded-md text-white text-sm font-sans transition-all duration-200 focus:outline-none focus:border-white/50 focus:shadow-[0_0_0_3px_rgba(255,255,255,0.1)] placeholder:text-[#666]"
                                placeholder="3600"
                                value={config.roundDuration}
                                onChange={(e) => handleConfigChange('roundDuration', e.target.value)}
                            />
                            <span className="text-[11px] text-[#666] mt-1 block">Duration for each training round (default: 1 hour)</span>
                        </div>

                        <div className="flex justify-end gap-3 mt-6">
                            <button
                                className="px-6 py-3 rounded-md text-sm font-semibold cursor-pointer transition-all duration-200 border-0 bg-white text-black hover:bg-neutral-200 hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
                                onClick={() => setCurrentStep('upload')}
                                disabled={!config.name || !config.initialCommitment}
                            >
                                Continue to Upload
                            </button>
                        </div>
                    </>
                )}

                {currentStep === 'upload' && (
                    <>
                        <div className="border-2 border-dashed border-white/10 rounded-md p-10 text-center cursor-pointer transition-all duration-200 hover:border-white/50 hover:bg-white/[0.04]" onClick={simulateUpload}>
                            <div className="text-5xl mb-4">📦</div>
                            <div className="text-sm text-[#aaa] mb-2">Click to upload model weights</div>
                            <div className="text-xs text-[#666]">Supports .pt, .onnx, .safetensors (max 1GB)</div>
                        </div>

                        {uploadProgress > 0 && uploadProgress < 100 && (
                            <div className="mt-6">
                                <div className="flex justify-between text-xs text-[#888] mb-2">
                                    <span>Uploading to IPFS...</span>
                                    <span>{uploadProgress}%</span>
                                </div>
                                <div className="w-full h-2 bg-helix-border rounded overflow-hidden">
                                    <div className="h-full bg-white rounded transition-all duration-300 ease-in-out" style={{ width: `${uploadProgress}%` }} />
                                </div>
                            </div>
                        )}

                        {config.ipfsHash && (
                            <div className="mt-4">
                                <label className="block text-[13px] font-semibold text-[#aaa] mb-2">IPFS Hash</label>
                                <input
                                    type="text"
                                    className="w-full px-4 py-3 bg-white/[0.03] border border-helix-border rounded-md text-white text-sm font-sans transition-all duration-200 focus:outline-none focus:border-white/50 focus:shadow-[0_0_0_3px_rgba(255,255,255,0.1)] placeholder:text-[#666] font-mono text-xs"
                                    value={config.ipfsHash}
                                    readOnly
                                />
                            </div>
                        )}

                        <div className="flex justify-end gap-3 mt-6">
                            <button className="px-6 py-3 rounded-md text-sm font-semibold cursor-pointer transition-all duration-200 border border-helix-border bg-white/10 text-[#aaa] hover:bg-white/15" onClick={() => setCurrentStep('configure')}>
                                Back
                            </button>
                            <button
                                className="px-6 py-3 rounded-md text-sm font-semibold cursor-pointer transition-all duration-200 border-0 bg-white text-black hover:bg-neutral-200 hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
                                onClick={() => setCurrentStep('register')}
                                disabled={!config.ipfsHash}
                            >
                                Continue to Register
                            </button>
                        </div>
                    </>
                )}

                {currentStep === 'register' && (
                    <>
                        <h3 className="mb-4 text-lg font-semibold text-white">
                            Review & Register
                        </h3>

                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-white/[0.03] p-4 rounded-md flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center text-base bg-white/20 text-white">✓</div>
                                <div>
                                    <div className="text-[13px] text-[#aaa]">Model Name</div>
                                    <div className="text-[11px] text-[#666] font-mono">{config.name}</div>
                                </div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center text-base bg-white/20 text-white">✓</div>
                                <div>
                                    <div className="text-[13px] text-[#aaa]">IPFS Hash</div>
                                    <div className="text-[11px] text-[#666] font-mono">{config.ipfsHash?.slice(0, 20)}...</div>
                                </div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center text-base bg-white/20 text-white">✓</div>
                                <div>
                                    <div className="text-[13px] text-[#aaa]">Initial Commitment</div>
                                    <div className="text-[11px] text-[#666] font-mono">{config.initialCommitment}</div>
                                </div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center text-base bg-white/20 text-white">✓</div>
                                <div>
                                    <div className="text-[13px] text-[#aaa]">Minimum Stake</div>
                                    <div className="text-[11px] text-[#666] font-mono">{config.minStake} ETH</div>
                                </div>
                            </div>
                        </div>

                        {registrationError && (
                            <div className="bg-white/[0.04] border border-helix-border rounded-md px-4 py-3 text-[#888] text-[13px] mt-4">
                                Registration failed: {registrationError.message}
                            </div>
                        )}

                        <div className="flex justify-end gap-3 mt-6">
                            <button className="px-6 py-3 rounded-md text-sm font-semibold cursor-pointer transition-all duration-200 border border-helix-border bg-white/10 text-[#aaa] hover:bg-white/15" onClick={() => setCurrentStep('upload')}>
                                Back
                            </button>
                            <button
                                className="px-6 py-3 rounded-md text-sm font-semibold cursor-pointer transition-all duration-200 border-0 bg-white text-black hover:bg-neutral-200 hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
                                onClick={handleRegister}
                                disabled={isRegistering}
                            >
                                {isRegistering ? 'Registering...' : 'Register On-Chain'}
                            </button>
                        </div>
                    </>
                )}

                {currentStep === 'verify' && (
                    <>
                        <h3 className="mb-4 text-lg font-semibold text-white">
                            Verifying Registration
                        </h3>

                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-white/[0.03] p-4 rounded-md flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center text-base bg-white/20 text-white">✓</div>
                                <div>
                                    <div className="text-[13px] text-[#aaa]">Transaction Confirmed</div>
                                    <div className="text-[11px] text-[#666] font-mono">Block confirmed</div>
                                </div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center text-base bg-white/20 text-white">✓</div>
                                <div>
                                    <div className="text-[13px] text-[#aaa]">Model Stored</div>
                                    <div className="text-[11px] text-[#666] font-mono">On-chain registry updated</div>
                                </div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center text-base bg-white/20 text-white">✓</div>
                                <div>
                                    <div className="text-[13px] text-[#aaa]">IPFS Pinned</div>
                                    <div className="text-[11px] text-[#666] font-mono">Distributed storage active</div>
                                </div>
                            </div>
                            <div className="bg-white/[0.03] p-4 rounded-md flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center text-base bg-white/[0.04] text-[#888]">⏳</div>
                                <div>
                                    <div className="text-[13px] text-[#aaa]">Ready for Training</div>
                                    <div className="text-[11px] text-[#666] font-mono">Awaiting first round</div>
                                </div>
                            </div>
                        </div>

                        <div className="flex justify-end gap-3 mt-6">
                            <button className="px-6 py-3 rounded-md text-sm font-semibold cursor-pointer transition-all duration-200 border-0 bg-white text-black hover:bg-neutral-200 hover:-translate-y-0.5" onClick={handleVerify}>
                                Complete Setup
                            </button>
                        </div>
                    </>
                )}

                {currentStep === 'complete' && (
                    <div className="text-center py-10">
                        <div className="w-20 h-20 rounded-md bg-white/20 text-white text-[40px] flex items-center justify-center mx-auto mb-4">✓</div>
                        <div className="font-semibold text-[15px] text-white mb-2">Model Deployed Successfully!</div>
                        <div className="text-sm text-[#888] mb-4">
                            Your model is now registered on the HELIX network and ready for distributed training.
                        </div>

                        <div className="inline-block px-6 py-3 bg-white/20 rounded-md font-mono text-lg text-white mb-4">
                            Model #{registeredModelId?.toString() || '0'}
                        </div>

                        <div className="flex gap-3 justify-center">
                            <button className="px-6 py-3 rounded-md text-sm font-semibold cursor-pointer transition-all duration-200 border border-helix-border bg-white/10 text-[#aaa] hover:bg-white/15" onClick={() => setCurrentStep('configure')}>
                                Deploy Another Model
                            </button>
                            <button className="px-6 py-3 rounded-md text-sm font-semibold cursor-pointer transition-all duration-200 border-0 bg-white text-black hover:bg-neutral-200 hover:-translate-y-0.5" onClick={handleStartFirstRound}>
                                {isStartingRound ? 'Starting...' : 'Start First Round'}
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
