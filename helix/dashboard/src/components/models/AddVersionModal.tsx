'use client';

import { useState, useRef } from 'react';
import { CheckCircle, XCircle, Loader2, Plus, Upload, Shield } from 'lucide-react';
import { useAccount, useSignMessage } from 'wagmi';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/lib/utils';
import { useModelRegistry } from '@/hooks/useModelRegistry';
import { deriveModelKey, encryptWeights } from '@/lib/model-encryption';

const SEMVER_REGEX = /^\d+\.\d+\.\d+$/;

type VersionPhase = 'idle' | 'validating' | 'encrypting' | 'uploading' | 'registering' | 'done' | 'error';

interface AddVersionModalProps {
  tokenId: number;
  onClose: () => void;
  onAdded?: () => void;
}

export function AddVersionModal({ tokenId, onClose, onAdded }: AddVersionModalProps) {
  const [version, setVersion] = useState('1.0.0');
  const [accuracy, setAccuracy] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [versionError, setVersionError] = useState<string | null>(null);
  const [phase, setPhase] = useState<VersionPhase>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [resultHash, setResultHash] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const { isConnected } = useAccount();
  const { signMessageAsync } = useSignMessage();
  const { addVersion, isWritePending } = useModelRegistry();

  const reset = () => {
    setVersion('1.0.0');
    setAccuracy('');
    setFile(null);
    setVersionError(null);
    setPhase('idle');
    setErrorMsg('');
    setResultHash('');
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleVersionChange = (v: string) => {
    setVersion(v);
    if (v.trim() && !SEMVER_REGEX.test(v.trim())) {
      setVersionError('Must be semver (e.g. 1.0.0)');
    } else {
      setVersionError(null);
    }
  };

  const handleSubmit = async () => {
    const v = version.trim() || '1.0.0';
    if (!SEMVER_REGEX.test(v)) return;

    const acc = accuracy ? parseFloat(accuracy) : 0;
    const sessionId = `v${v}-${Date.now()}`;

    try {
      if (file) {
        setPhase('validating');
        const text = await file.text();
        const data = JSON.parse(text);
        const weights = data.weights || data;
        if (!weights.w1 || !weights.b1 || !weights.w2 || !weights.b2) {
          throw new Error('Invalid model format. Expected { w1, b1, w2, b2 } weight arrays.');
        }

        let rootHash = '';

        if (isConnected) {
          setPhase('encrypting');
          const key = await deriveModelKey(
            (message: string) => signMessageAsync({ message }),
            tokenId,
          );
          const encrypted = await encryptWeights(key, JSON.stringify(weights));
          const encryptedBase64 = btoa(String.fromCharCode(...encrypted));

          setPhase('uploading');
          const res = await fetch('/api/store-on-0g', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              encrypted: true,
              encryptedPayload: encryptedBase64,
              accuracy: acc > 0 ? acc / 100 : null,
              version: v,
            }),
          });

          if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            throw new Error(errBody.error || `Upload failed: HTTP ${res.status}`);
          }

          const result = await res.json();
          rootHash = result.root_hash;
          setResultHash(rootHash);
        } else {
          setPhase('uploading');
          const res = await fetch('/api/store-on-0g', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              weights,
              accuracy: acc > 0 ? acc / 100 : null,
              version: v,
            }),
          });

          if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            throw new Error(errBody.error || `Upload failed: HTTP ${res.status}`);
          }

          const result = await res.json();
          rootHash = result.root_hash;
          setResultHash(rootHash);
        }

        setPhase('registering');
        addVersion({
          tokenId,
          semver: v,
          rootHash,
          accuracy: acc,
          sessionId,
          weightsStored: true,
        });

        setPhase('done');
        onAdded?.();
      } else {
        setPhase('registering');
        addVersion({
          tokenId,
          semver: v,
          rootHash: '',
          accuracy: acc,
          sessionId,
          weightsStored: false,
        });

        setPhase('done');
        onAdded?.();
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to add version');
      setPhase('error');
    }
  };

  const canSubmit = !versionError && version.trim() && !isWritePending;

  return (
    <Modal isOpen onClose={handleClose} title="Add Version">
      <div className="space-y-4">
        {phase === 'idle' && (
          <>
            {/* Version */}
            <div>
              <label className="label-text block mb-1.5">Version (semver)</label>
              <input
                type="text"
                value={version}
                onChange={(e) => handleVersionChange(e.target.value)}
                placeholder="1.0.0"
                className={cn(
                  'w-full px-3 py-2 bg-helix-bg border rounded-md text-sm text-helix-text font-mono focus:outline-none transition-colors',
                  versionError ? 'border-red-500/50' : 'border-helix-border focus:border-helix-border2',
                )}
              />
              {versionError && <p className="text-2xs text-red-400 mt-1">{versionError}</p>}
            </div>

            {/* Weights File */}
            <div>
              <label className="label-text block mb-1.5">Weights File (optional, JSON)</label>
              <div
                onClick={() => fileRef.current?.click()}
                className={cn(
                  'flex flex-col items-center justify-center gap-2 p-6 rounded-xl border-2 border-dashed cursor-pointer transition-colors',
                  file
                    ? 'border-green-500/30 bg-green-500/5'
                    : 'border-helix-border hover:border-helix-border2 bg-helix-bg',
                )}
              >
                {file ? (
                  <>
                    <CheckCircle size={20} className="text-green-400" />
                    <span className="text-sm text-green-300">{file.name}</span>
                    <span className="text-2xs text-helix-muted">
                      {(file.size / 1024).toFixed(0)} KB
                    </span>
                  </>
                ) : (
                  <>
                    <Upload size={20} className="text-helix-muted" />
                    <span className="text-sm text-helix-text2">Click to select weights file</span>
                    <span className="text-2xs text-helix-muted">
                      JSON with w1, b1, w2, b2 weight arrays
                    </span>
                  </>
                )}
                <input
                  ref={fileRef}
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </div>
              {file && isConnected && (
                <div className="flex items-center gap-2 mt-2 text-2xs text-helix-muted">
                  <Shield size={12} className="text-green-400 shrink-0" />
                  Weights will be encrypted with your wallet key before upload
                </div>
              )}
            </div>

            {/* Accuracy */}
            <div>
              <label className="label-text block mb-1.5">Accuracy % (optional)</label>
              <input
                type="number"
                value={accuracy}
                onChange={(e) => setAccuracy(e.target.value)}
                placeholder="e.g. 95.5"
                min={0}
                max={100}
                step={0.1}
                className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text font-mono focus:outline-none focus:border-helix-border2 transition-colors"
              />
            </div>

            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canSubmit}
              className={cn(
                'w-full flex items-center justify-center gap-2 py-3 rounded-xl font-medium text-sm transition-all',
                !canSubmit
                  ? 'bg-helix-border text-helix-muted cursor-not-allowed'
                  : 'bg-white text-black hover:bg-white/90',
              )}
            >
              <Plus size={16} />
              {file ? 'Upload & Register Version' : 'Register Version'}
            </button>
          </>
        )}

        {/* Progress states */}
        {(phase === 'validating' || phase === 'encrypting' || phase === 'uploading' || phase === 'registering') && (
          <div className="flex flex-col items-center py-8 gap-4">
            <Loader2 size={24} className="animate-spin text-white" />
            <p className="text-sm text-helix-text2">
              {phase === 'validating' && 'Validating model weights...'}
              {phase === 'encrypting' && 'Encrypting weights (sign in wallet)...'}
              {phase === 'uploading' && 'Uploading to 0G decentralized storage...'}
              {phase === 'registering' && 'Registering on-chain (confirm in wallet)...'}
            </p>
            <div className="flex items-center gap-2">
              {(['validating', 'encrypting', 'uploading', 'registering'] as const).map((s, i) => (
                <div
                  key={s}
                  className={cn(
                    'w-2 h-2 rounded-full transition-colors',
                    phase === s ? 'bg-white' :
                    (['validating', 'encrypting', 'uploading', 'registering'].indexOf(phase) > i)
                      ? 'bg-green-400' : 'bg-helix-border',
                  )}
                />
              ))}
            </div>
          </div>
        )}

        {phase === 'done' && (
          <div className="flex flex-col items-center py-6 gap-3">
            <CheckCircle size={32} className="text-green-400" />
            <p className="text-sm font-medium text-white">Version Registered</p>
            {resultHash && (
              <div className="w-full">
                <p className="label-text mb-1">0G Root Hash</p>
                <code className="block text-xs font-mono text-helix-text bg-helix-bg px-3 py-2 rounded-md border border-helix-border truncate">
                  {resultHash}
                </code>
              </div>
            )}
            <button
              type="button"
              onClick={handleClose}
              className="mt-2 px-6 py-2 rounded-xl bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
            >
              Done
            </button>
          </div>
        )}

        {phase === 'error' && (
          <div className="flex flex-col items-center py-6 gap-3">
            <XCircle size={32} className="text-red-400" />
            <p className="text-sm text-red-300">{errorMsg}</p>
            <button
              type="button"
              onClick={() => setPhase('idle')}
              className="mt-2 px-6 py-2 rounded-xl bg-helix-surface border border-helix-border text-sm text-helix-text hover:text-white transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </Modal>
  );
}
