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
    const { registerModel, isRegistering, registrationError, registrationSuccess } = useModelRegistration();
    const { startRound, isStartingRound } = useRoundManagement(BigInt(0));
    const { defaultMinStake, isLoading: stateLoading } = useContractState();

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
    const [registeredModelId, setRegisteredModelId] = useState<bigint | null>(null);

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
        <div className="model-deployment">
            <style jsx>{`
                .model-deployment {
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border-radius: 16px;
                    padding: 24px;
                    color: #fff;
                    font-family: 'Inter', -apple-system, sans-serif;
                }

                .header {
                    margin-bottom: 32px;
                }

                .title {
                    font-size: 24px;
                    font-weight: 700;
                    background: linear-gradient(90deg, #06b6d4, #6366f1);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 8px;
                }

                .subtitle {
                    font-size: 14px;
                    color: #9ca3af;
                }

                .steps-container {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 32px;
                    position: relative;
                }

                .steps-container::before {
                    content: '';
                    position: absolute;
                    top: 20px;
                    left: 40px;
                    right: 40px;
                    height: 2px;
                    background: rgba(255, 255, 255, 0.1);
                }

                .step {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 8px;
                    position: relative;
                    z-index: 1;
                }

                .step-icon {
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 600;
                    font-size: 14px;
                    transition: all 0.3s;
                }

                .step-icon.completed {
                    background: #22c55e;
                    color: #fff;
                }

                .step-icon.current {
                    background: #6366f1;
                    color: #fff;
                    box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
                }

                .step-icon.pending {
                    background: rgba(255, 255, 255, 0.1);
                    color: #6b7280;
                }

                .step-label {
                    font-size: 12px;
                    color: #9ca3af;
                }

                .step-label.current {
                    color: #6366f1;
                    font-weight: 600;
                }

                .content {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                }

                .form-group {
                    margin-bottom: 20px;
                }

                .form-label {
                    display: block;
                    font-size: 13px;
                    font-weight: 600;
                    color: #d1d5db;
                    margin-bottom: 8px;
                }

                .form-input {
                    width: 100%;
                    padding: 12px 16px;
                    background: rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    color: #fff;
                    font-size: 14px;
                    font-family: inherit;
                    transition: all 0.2s;
                }

                .form-input:focus {
                    outline: none;
                    border-color: rgba(99, 102, 241, 0.5);
                    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
                }

                .form-input::placeholder {
                    color: #6b7280;
                }

                .form-textarea {
                    min-height: 80px;
                    resize: vertical;
                }

                .form-hint {
                    font-size: 11px;
                    color: #6b7280;
                    margin-top: 4px;
                }

                .form-row {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 16px;
                }

                .btn {
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s;
                    border: none;
                }

                .btn-primary {
                    background: linear-gradient(135deg, #6366f1, #8b5cf6);
                    color: #fff;
                }

                .btn-primary:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
                }

                .btn-primary:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }

                .btn-secondary {
                    background: rgba(255, 255, 255, 0.1);
                    color: #d1d5db;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }

                .btn-secondary:hover {
                    background: rgba(255, 255, 255, 0.15);
                }

                .actions {
                    display: flex;
                    justify-content: flex-end;
                    gap: 12px;
                    margin-top: 24px;
                }

                .upload-zone {
                    border: 2px dashed rgba(255, 255, 255, 0.2);
                    border-radius: 12px;
                    padding: 40px;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .upload-zone:hover {
                    border-color: rgba(99, 102, 241, 0.5);
                    background: rgba(99, 102, 241, 0.05);
                }

                .upload-icon {
                    font-size: 48px;
                    margin-bottom: 16px;
                }

                .upload-text {
                    font-size: 14px;
                    color: #d1d5db;
                    margin-bottom: 8px;
                }

                .upload-hint {
                    font-size: 12px;
                    color: #6b7280;
                }

                .progress-container {
                    margin-top: 24px;
                }

                .progress-label {
                    display: flex;
                    justify-content: space-between;
                    font-size: 12px;
                    color: #9ca3af;
                    margin-bottom: 8px;
                }

                .progress-bar-bg {
                    width: 100%;
                    height: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                    overflow: hidden;
                }

                .progress-bar {
                    height: 100%;
                    background: linear-gradient(90deg, #6366f1, #a855f7);
                    border-radius: 4px;
                    transition: width 0.3s ease;
                }

                .verification-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 16px;
                }

                .verification-item {
                    background: rgba(0, 0, 0, 0.2);
                    padding: 16px;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }

                .verification-icon {
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                }

                .verification-icon.success {
                    background: rgba(34, 197, 94, 0.2);
                    color: #22c55e;
                }

                .verification-icon.pending {
                    background: rgba(245, 158, 11, 0.2);
                    color: #f59e0b;
                }

                .verification-label {
                    font-size: 13px;
                    color: #d1d5db;
                }

                .verification-value {
                    font-size: 11px;
                    color: #6b7280;
                    font-family: 'JetBrains Mono', monospace;
                }

                .success-container {
                    text-align: center;
                    padding: 40px;
                }

                .success-icon {
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    background: rgba(34, 197, 94, 0.2);
                    color: #22c55e;
                    font-size: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 24px;
                }

                .success-title {
                    font-size: 24px;
                    font-weight: 700;
                    color: #22c55e;
                    margin-bottom: 8px;
                }

                .success-subtitle {
                    font-size: 14px;
                    color: #9ca3af;
                    margin-bottom: 24px;
                }

                .model-id-badge {
                    display: inline-block;
                    padding: 12px 24px;
                    background: rgba(99, 102, 241, 0.2);
                    border-radius: 8px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 18px;
                    color: #818cf8;
                    margin-bottom: 24px;
                }

                .error-message {
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.3);
                    border-radius: 8px;
                    padding: 12px 16px;
                    color: #fca5a5;
                    font-size: 13px;
                    margin-top: 16px;
                }
            `}</style>

            <div className="header">
                <h2 className="title">Deploy New Model</h2>
                <p className="subtitle">Register your ML model on the HELIX network for decentralized training</p>
            </div>

            <div className="steps-container">
                {steps.map((step) => {
                    const status = getStepStatus(step.key);
                    return (
                        <div key={step.key} className="step">
                            <div className={`step-icon ${status}`}>
                                {status === 'completed' ? '✓' : step.icon}
                            </div>
                            <span className={`step-label ${status === 'current' ? 'current' : ''}`}>
                                {step.label}
                            </span>
                        </div>
                    );
                })}
            </div>

            <div className="content">
                {currentStep === 'configure' && (
                    <>
                        <div className="form-group">
                            <label className="form-label">Model Name</label>
                            <input
                                type="text"
                                className="form-input"
                                placeholder="e.g., HELIX-GPT-Small"
                                value={config.name}
                                onChange={(e) => handleConfigChange('name', e.target.value)}
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Description</label>
                            <textarea
                                className="form-input form-textarea"
                                placeholder="Describe your model architecture and training objectives..."
                                value={config.description}
                                onChange={(e) => handleConfigChange('description', e.target.value)}
                            />
                        </div>

                        <div className="form-row">
                            <div className="form-group">
                                <label className="form-label">Initial Commitment (Hash)</label>
                                <input
                                    type="text"
                                    className="form-input"
                                    placeholder="12345678"
                                    value={config.initialCommitment}
                                    onChange={(e) => handleConfigChange('initialCommitment', e.target.value)}
                                />
                                <span className="form-hint">Poseidon hash of initial model weights</span>
                            </div>

                            <div className="form-group">
                                <label className="form-label">Minimum Stake (ETH)</label>
                                <input
                                    type="text"
                                    className="form-input"
                                    placeholder="0.1"
                                    value={config.minStake}
                                    onChange={(e) => handleConfigChange('minStake', e.target.value)}
                                />
                                <span className="form-hint">
                                    Default: {defaultMinStake ? `${Number(defaultMinStake) / 1e18} ETH` : '0.1 ETH'}
                                </span>
                            </div>
                        </div>

                        <div className="form-group">
                            <label className="form-label">Round Duration (seconds)</label>
                            <input
                                type="text"
                                className="form-input"
                                placeholder="3600"
                                value={config.roundDuration}
                                onChange={(e) => handleConfigChange('roundDuration', e.target.value)}
                            />
                            <span className="form-hint">Duration for each training round (default: 1 hour)</span>
                        </div>

                        <div className="actions">
                            <button
                                className="btn btn-primary"
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
                        <div className="upload-zone" onClick={simulateUpload}>
                            <div className="upload-icon">📦</div>
                            <div className="upload-text">Click to upload model weights</div>
                            <div className="upload-hint">Supports .pt, .onnx, .safetensors (max 1GB)</div>
                        </div>

                        {uploadProgress > 0 && uploadProgress < 100 && (
                            <div className="progress-container">
                                <div className="progress-label">
                                    <span>Uploading to IPFS...</span>
                                    <span>{uploadProgress}%</span>
                                </div>
                                <div className="progress-bar-bg">
                                    <div className="progress-bar" style={{ width: `${uploadProgress}%` }} />
                                </div>
                            </div>
                        )}

                        {config.ipfsHash && (
                            <div style={{ marginTop: '16px' }}>
                                <label className="form-label">IPFS Hash</label>
                                <input
                                    type="text"
                                    className="form-input"
                                    value={config.ipfsHash}
                                    readOnly
                                    style={{ fontFamily: 'monospace', fontSize: '12px' }}
                                />
                            </div>
                        )}

                        <div className="actions">
                            <button className="btn btn-secondary" onClick={() => setCurrentStep('configure')}>
                                Back
                            </button>
                            <button
                                className="btn btn-primary"
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
                        <h3 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: '600' }}>
                            Review & Register
                        </h3>

                        <div className="verification-grid">
                            <div className="verification-item">
                                <div className="verification-icon success">✓</div>
                                <div>
                                    <div className="verification-label">Model Name</div>
                                    <div className="verification-value">{config.name}</div>
                                </div>
                            </div>
                            <div className="verification-item">
                                <div className="verification-icon success">✓</div>
                                <div>
                                    <div className="verification-label">IPFS Hash</div>
                                    <div className="verification-value">{config.ipfsHash?.slice(0, 20)}...</div>
                                </div>
                            </div>
                            <div className="verification-item">
                                <div className="verification-icon success">✓</div>
                                <div>
                                    <div className="verification-label">Initial Commitment</div>
                                    <div className="verification-value">{config.initialCommitment}</div>
                                </div>
                            </div>
                            <div className="verification-item">
                                <div className="verification-icon success">✓</div>
                                <div>
                                    <div className="verification-label">Minimum Stake</div>
                                    <div className="verification-value">{config.minStake} ETH</div>
                                </div>
                            </div>
                        </div>

                        {registrationError && (
                            <div className="error-message">
                                Registration failed: {registrationError.message}
                            </div>
                        )}

                        <div className="actions">
                            <button className="btn btn-secondary" onClick={() => setCurrentStep('upload')}>
                                Back
                            </button>
                            <button
                                className="btn btn-primary"
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
                        <h3 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: '600' }}>
                            Verifying Registration
                        </h3>

                        <div className="verification-grid">
                            <div className="verification-item">
                                <div className="verification-icon success">✓</div>
                                <div>
                                    <div className="verification-label">Transaction Confirmed</div>
                                    <div className="verification-value">Block confirmed</div>
                                </div>
                            </div>
                            <div className="verification-item">
                                <div className="verification-icon success">✓</div>
                                <div>
                                    <div className="verification-label">Model Stored</div>
                                    <div className="verification-value">On-chain registry updated</div>
                                </div>
                            </div>
                            <div className="verification-item">
                                <div className="verification-icon success">✓</div>
                                <div>
                                    <div className="verification-label">IPFS Pinned</div>
                                    <div className="verification-value">Distributed storage active</div>
                                </div>
                            </div>
                            <div className="verification-item">
                                <div className="verification-icon pending">⏳</div>
                                <div>
                                    <div className="verification-label">Ready for Training</div>
                                    <div className="verification-value">Awaiting first round</div>
                                </div>
                            </div>
                        </div>

                        <div className="actions">
                            <button className="btn btn-primary" onClick={handleVerify}>
                                Complete Setup
                            </button>
                        </div>
                    </>
                )}

                {currentStep === 'complete' && (
                    <div className="success-container">
                        <div className="success-icon">✓</div>
                        <div className="success-title">Model Deployed Successfully!</div>
                        <div className="success-subtitle">
                            Your model is now registered on the HELIX network and ready for distributed training.
                        </div>

                        <div className="model-id-badge">
                            Model #{registeredModelId?.toString() || '0'}
                        </div>

                        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                            <button className="btn btn-secondary" onClick={() => setCurrentStep('configure')}>
                                Deploy Another Model
                            </button>
                            <button className="btn btn-primary" onClick={handleStartFirstRound}>
                                {isStartingRound ? 'Starting...' : 'Start First Round'}
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
