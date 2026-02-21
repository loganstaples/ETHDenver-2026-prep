'use client';

import { useState, useCallback, useEffect, type FormEvent } from 'react';
import { Loader2, CheckCircle2 } from 'lucide-react';
import { Modal } from '@/components/ui/Modal';
import { useModelRegistration } from '@/hooks/useContract';

interface RegisterModelDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const inputClassName =
  'w-full bg-helix-surface border border-helix-border rounded-lg px-3 py-2 text-sm text-white font-mono focus:border-helix-border2 focus:outline-none placeholder:text-helix-dim transition-colors';

export function RegisterModelDialog({ isOpen, onClose }: RegisterModelDialogProps) {
  const [ipfsHash, setIpfsHash] = useState('');
  const [initialCommitment, setInitialCommitment] = useState('');
  const [minStake, setMinStake] = useState('');

  const {
    registerModel,
    isRegistering,
    registrationSuccess,
    registrationError,
    modelId,
  } = useModelRegistration();

  const resetForm = useCallback(() => {
    setIpfsHash('');
    setInitialCommitment('');
    setMinStake('');
  }, []);

  // Reset form when dialog closes
  useEffect(() => {
    if (!isOpen) {
      resetForm();
    }
  }, [isOpen, resetForm]);

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!ipfsHash || !initialCommitment || !minStake) return;

      try {
        const commitmentBigInt = BigInt(initialCommitment);
        await registerModel(ipfsHash, commitmentBigInt, minStake);
      } catch {
        // Error is captured by the hook
      }
    },
    [ipfsHash, initialCommitment, minStake, registerModel],
  );

  const canSubmit = ipfsHash.length > 0 && initialCommitment.length > 0 && minStake.length > 0 && !isRegistering;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Register Model">
      {registrationSuccess ? (
        <div className="flex flex-col items-center py-6 space-y-3">
          <CheckCircle2 size={32} className="text-white" strokeWidth={1.5} />
          <p className="text-sm font-medium text-white">Model registered successfully</p>
          {modelId !== null && (
            <p className="text-2xs font-mono text-helix-muted">
              Model ID: {modelId.toString()}
            </p>
          )}
          <button
            type="button"
            onClick={onClose}
            className="mt-2 bg-white text-black font-medium text-sm px-4 py-2 rounded-lg hover:bg-white/90 transition-colors"
          >
            Done
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label
              htmlFor="ipfs-hash"
              className="block text-sm text-helix-muted"
            >
              IPFS Hash
            </label>
            <input
              id="ipfs-hash"
              type="text"
              value={ipfsHash}
              onChange={(e) => setIpfsHash(e.target.value)}
              placeholder="QmXoypiz..."
              className={inputClassName}
              disabled={isRegistering}
            />
          </div>

          <div className="space-y-1.5">
            <label
              htmlFor="commitment"
              className="block text-sm text-helix-muted"
            >
              Initial Commitment
            </label>
            <input
              id="commitment"
              type="text"
              value={initialCommitment}
              onChange={(e) => setInitialCommitment(e.target.value)}
              placeholder="0"
              className={inputClassName}
              disabled={isRegistering}
            />
          </div>

          <div className="space-y-1.5">
            <label
              htmlFor="min-stake"
              className="block text-sm text-helix-muted"
            >
              Minimum Stake (ADI)
            </label>
            <input
              id="min-stake"
              type="text"
              value={minStake}
              onChange={(e) => setMinStake(e.target.value)}
              placeholder="0.001"
              className={inputClassName}
              disabled={isRegistering}
            />
          </div>

          {registrationError && (
            <p className="text-2xs text-red-400 font-mono">
              {registrationError.message}
            </p>
          )}

          <button
            type="submit"
            disabled={!canSubmit}
            className="w-full bg-white text-black font-medium text-sm px-4 py-2 rounded-lg hover:bg-white/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isRegistering ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Registering...
              </>
            ) : (
              'Register Model'
            )}
          </button>
        </form>
      )}
    </Modal>
  );
}
