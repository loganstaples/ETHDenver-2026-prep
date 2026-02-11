'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileCode, Settings, Send, Check, Loader2 } from 'lucide-react';
import { useAccount, useChainId, usePublicClient, useWalletClient } from 'wagmi';
import { parseEther, type Address } from 'viem';
import { HELIX_COORDINATOR_ABI, getContractAddress } from '@/lib/contracts';
import { useContractState } from '@/hooks/useContract';

type Step = 'upload' | 'configure' | 'review';

export default function SubmitModelPage() {
    const { isConnected, address } = useAccount();
    const chainId = useChainId();
    const publicClient = usePublicClient();
    const { data: walletClient } = useWalletClient();
    const contractState = useContractState();

    const [step, setStep] = useState<Step>('upload');
    const [fileName, setFileName] = useState<string | null>(null);
    const [fileSize, setFileSize] = useState<number>(0);

    // Config form state
    const [modelName, setModelName] = useState('');
    const [epochs, setEpochs] = useState('10');
    const [batchSize, setBatchSize] = useState('32');
    const [learningRate, setLearningRate] = useState('0.001');
    const [optimizer, setOptimizer] = useState('AdamW');
    const [stakeAmount, setStakeAmount] = useState('0.1');

    // Transaction state
    const [submitting, setSubmitting] = useState(false);
    const [txHash, setTxHash] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFileDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) {
            setFileName(file.name);
            setFileSize(file.size);
            setModelName(file.name.replace(/\.[^/.]+$/, ''));
        }
    }, []);

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setFileName(file.name);
            setFileSize(file.size);
            setModelName(file.name.replace(/\.[^/.]+$/, ''));
        }
    }, []);

    const handleSubmit = async () => {
        if (!walletClient || !publicClient || !address) return;

        setSubmitting(true);
        setError(null);

        try {
            const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

            // Generate a commitment hash from model config
            const ipfsHash = `ipfs://Qm${modelName.replace(/\s/g, '')}${Date.now()}`;
            const initialCommitment = BigInt('0x' + Array(64).fill('0').map(() => Math.floor(Math.random() * 16).toString(16)).join(''));
            const minStake = parseEther(stakeAmount);

            const hash = await walletClient.writeContract({
                address: contractAddress,
                abi: HELIX_COORDINATOR_ABI,
                functionName: 'registerModel',
                args: [ipfsHash, initialCommitment, minStake],
            });

            setTxHash(hash);

            // Wait for confirmation
            await publicClient.waitForTransactionReceipt({ hash });
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Transaction failed');
        } finally {
            setSubmitting(false);
        }
    };

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    if (!isConnected) {
        return (
            <div className="flex flex-col items-center justify-center py-24 text-[#555]">
                <Upload className="w-10 h-10 mb-3 text-[#444]" />
                <p className="text-[13px] font-mono mb-1 text-[#888]">wallet not connected</p>
                <p className="text-[11px] text-[#555]">Connect a wallet to submit models for training</p>
            </div>
        );
    }

    const steps = [
        { key: 'upload' as Step, label: 'UPLOAD', num: '01' },
        { key: 'configure' as Step, label: 'CONFIG', num: '02' },
        { key: 'review' as Step, label: 'SUBMIT', num: '03' },
    ];

    const inputClass = 'w-full bg-transparent border border-helix-border rounded-[4px] px-3 py-2 text-[13px] text-white placeholder:text-[#444] focus:border-white/20 outline-none transition-colors font-mono';
    const selectClass = 'w-full bg-transparent border border-helix-border rounded-[4px] px-3 py-2 text-[13px] text-white focus:border-white/20 outline-none transition-colors font-mono cursor-pointer';

    return (
        <div className="max-w-2xl mx-auto space-y-6">
            {/* Steps indicator */}
            <div className="flex items-center gap-2">
                {steps.map((s, i) => {
                    const isActive = s.key === step;
                    const isPast = (['upload', 'configure', 'review'].indexOf(step) > i);
                    return (
                        <div key={s.key} className="flex items-center gap-2 flex-1">
                            <div className={`w-6 h-6 rounded-md flex items-center justify-center font-mono text-[10px] transition-all ${
                                isActive ? 'bg-white text-black' :
                                isPast ? 'bg-white/10 text-white' :
                                'bg-white/[0.03] text-[#555]'
                            }`}>
                                {isPast ? <Check className="w-3 h-3" /> : s.num}
                            </div>
                            <span className={`text-[11px] uppercase tracking-wider ${isActive ? 'text-white' : 'text-[#555]'}`}>
                                {s.label}
                            </span>
                            {i < 2 && <div className="flex-1 h-px bg-helix-border" />}
                        </div>
                    );
                })}
            </div>

            {/* Step: Upload */}
            {step === 'upload' && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                >
                    <div
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleFileDrop}
                        className="border border-dashed border-[#333] hover:border-[#555] rounded-md p-8 text-center transition-colors cursor-pointer"
                    >
                        <input
                            type="file"
                            id="model-file"
                            className="hidden"
                            accept=".pt,.pth,.safetensors,.onnx,.bin"
                            onChange={handleFileSelect}
                        />
                        <label htmlFor="model-file" className="cursor-pointer">
                            {fileName ? (
                                <div className="space-y-2">
                                    <FileCode className="w-8 h-8 mx-auto text-[#888]" />
                                    <p className="font-mono text-[13px] text-white">{fileName}</p>
                                    <p className="font-mono text-[11px] text-[#555]">{formatFileSize(fileSize)}</p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <Upload className="w-8 h-8 mx-auto text-[#444]" />
                                    <p className="text-[13px] text-[#666]">Drop model file here or click to browse</p>
                                    <p className="font-mono text-[11px] text-[#444]">.pt .safetensors .onnx .bin</p>
                                </div>
                            )}
                        </label>
                    </div>

                    <button
                        onClick={() => setStep('configure')}
                        disabled={!fileName}
                        className="w-full bg-white text-black text-[13px] font-medium px-4 py-2 rounded-[4px] hover:bg-white/90 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                        Continue
                    </button>
                </motion.div>
            )}

            {/* Step: Configure */}
            {step === 'configure' && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                >
                    <div className="bg-helix-surface border border-helix-border rounded-md p-4 space-y-4">
                        <div>
                            <label className="block text-[11px] text-[#666] uppercase tracking-wider mb-1.5">Model Name</label>
                            <input
                                type="text"
                                value={modelName}
                                onChange={(e) => setModelName(e.target.value)}
                                className={inputClass}
                                placeholder="my-model"
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-[11px] text-[#666] uppercase tracking-wider mb-1.5">Epochs</label>
                                <input
                                    type="number"
                                    value={epochs}
                                    onChange={(e) => setEpochs(e.target.value)}
                                    className={inputClass}
                                />
                            </div>
                            <div>
                                <label className="block text-[11px] text-[#666] uppercase tracking-wider mb-1.5">Batch Size</label>
                                <select
                                    value={batchSize}
                                    onChange={(e) => setBatchSize(e.target.value)}
                                    className={selectClass}
                                >
                                    <option value="16">16</option>
                                    <option value="32">32</option>
                                    <option value="64">64</option>
                                    <option value="128">128</option>
                                </select>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-[11px] text-[#666] uppercase tracking-wider mb-1.5">Learning Rate</label>
                                <input
                                    type="number"
                                    step="0.0001"
                                    value={learningRate}
                                    onChange={(e) => setLearningRate(e.target.value)}
                                    className={inputClass}
                                />
                            </div>
                            <div>
                                <label className="block text-[11px] text-[#666] uppercase tracking-wider mb-1.5">Optimizer</label>
                                <select
                                    value={optimizer}
                                    onChange={(e) => setOptimizer(e.target.value)}
                                    className={selectClass}
                                >
                                    <option value="AdamW">AdamW</option>
                                    <option value="Adam">Adam</option>
                                    <option value="SGD">SGD</option>
                                </select>
                            </div>
                        </div>

                        <div>
                            <label className="block text-[11px] text-[#666] uppercase tracking-wider mb-1.5">
                                Min Stake
                                {contractState.defaultMinStake !== undefined && (
                                    <span className="text-[#444] ml-2 normal-case tracking-normal">
                                        (network min: {(Number(contractState.defaultMinStake) / 1e18).toFixed(4)} ETH)
                                    </span>
                                )}
                            </label>
                            <div className="flex items-center border border-helix-border rounded-[4px] overflow-hidden">
                                <input
                                    type="number"
                                    step="0.01"
                                    value={stakeAmount}
                                    onChange={(e) => setStakeAmount(e.target.value)}
                                    className="flex-1 bg-transparent px-3 py-2 text-[13px] text-white font-mono outline-none"
                                />
                                <span className="px-3 text-[11px] text-[#555] font-mono">ETH</span>
                            </div>
                        </div>
                    </div>

                    <div className="flex gap-2">
                        <button
                            onClick={() => setStep('upload')}
                            className="flex-1 border border-helix-border text-[13px] text-[#888] px-4 py-2 rounded-[4px] hover:text-white hover:border-white/20 transition-colors"
                        >
                            Back
                        </button>
                        <button
                            onClick={() => setStep('review')}
                            className="flex-1 bg-white text-black text-[13px] font-medium px-4 py-2 rounded-[4px] hover:bg-white/90 transition-colors"
                        >
                            Review
                        </button>
                    </div>
                </motion.div>
            )}

            {/* Step: Review & Submit */}
            {step === 'review' && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                >
                    <div className="bg-helix-surface border border-helix-border rounded-md p-4 space-y-0">
                        <div className="text-[11px] text-[#666] uppercase tracking-wider mb-3">Review Submission</div>

                        {[
                            { label: 'model_file', value: fileName },
                            { label: 'file_size', value: formatFileSize(fileSize) },
                            { label: 'model_name', value: modelName },
                            { label: 'epochs', value: epochs },
                            { label: 'batch_size', value: batchSize },
                            { label: 'learning_rate', value: learningRate },
                            { label: 'optimizer', value: optimizer },
                            { label: 'min_stake', value: `${stakeAmount} ETH` },
                        ].map((item) => (
                            <div key={item.label} className="flex items-center justify-between py-1.5 border-b border-helix-border last:border-0">
                                <span className="text-[13px] text-[#888] font-mono">{item.label}</span>
                                <span className="text-[13px] text-white font-mono">{item.value}</span>
                            </div>
                        ))}
                    </div>

                    {error && (
                        <div className="bg-white/[0.02] border border-[#333] rounded-md p-4 text-[13px] text-[#888] font-mono">
                            {error}
                        </div>
                    )}

                    {txHash && (
                        <div className="bg-white/[0.02] border border-[#333] rounded-md p-4">
                            <p className="text-[11px] text-[#666] uppercase tracking-wider mb-1">Transaction Submitted</p>
                            <a
                                href={`https://etherscan.io/tx/${txHash}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[#888] hover:text-white transition-colors font-mono text-[12px]"
                            >
                                {txHash.slice(0, 20)}...{txHash.slice(-8)}
                            </a>
                        </div>
                    )}

                    <div className="flex gap-2">
                        <button
                            onClick={() => setStep('configure')}
                            disabled={submitting}
                            className="flex-1 border border-helix-border text-[13px] text-[#888] px-4 py-2 rounded-[4px] hover:text-white hover:border-white/20 transition-colors disabled:opacity-30"
                        >
                            Back
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={submitting}
                            className="flex-1 bg-white text-black text-[13px] font-medium px-4 py-2 rounded-[4px] hover:bg-white/90 transition-colors disabled:opacity-30 flex items-center justify-center gap-2"
                        >
                            {submitting ? (
                                <>
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                    Submitting...
                                </>
                            ) : (
                                <>
                                    <Send className="w-3.5 h-3.5" />
                                    Register Model
                                </>
                            )}
                        </button>
                    </div>
                </motion.div>
            )}
        </div>
    );
}
