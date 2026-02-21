'use client';

import { useState, useEffect } from 'react';
import { Loader2, Plus } from 'lucide-react';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/lib/utils';
import { useModelRegistry } from '@/hooks/useModelRegistry';

const SLUG_REGEX = /^[a-z0-9]+(-[a-z0-9]+)*$/;

interface CreateModelModalProps {
  onClose: () => void;
  onCreated?: () => void;
}

export function CreateModelModal({ onClose, onCreated }: CreateModelModalProps) {
  const [slug, setSlug] = useState('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [slugError, setSlugError] = useState<string | null>(null);

  const { createModel, isWritePending, isConfirming, isSuccess } = useModelRegistry();

  useEffect(() => {
    if (isSuccess) {
      onCreated?.();
      handleClose();
    }
  }, [isSuccess]);

  const reset = () => {
    setSlug('');
    setName('');
    setDescription('');
    setSlugError(null);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleSlugChange = (v: string) => {
    const lower = v.toLowerCase().replace(/\s+/g, '-');
    setSlug(lower);
    if (lower && !SLUG_REGEX.test(lower)) {
      setSlugError('Must be kebab-case (e.g. my-mnist-model)');
    } else {
      setSlugError(null);
    }
  };

  const canSubmit = slug.trim() && name.trim() && !slugError && !isWritePending;

  const handleSubmit = () => {
    if (!canSubmit) return;
    createModel({ slug: slug.trim(), name: name.trim(), description: description.trim() });
    // Keep modal open — it will show the pending state via isWritePending
  };

  return (
    <Modal isOpen onClose={handleClose} title="Create Model">
      <div className="space-y-4">
        {(isWritePending || isConfirming) ? (
          <div className="flex flex-col items-center py-8 gap-4">
            <Loader2 size={24} className="animate-spin text-white" />
            <p className="text-sm text-helix-text2">
              {isConfirming ? 'Confirming transaction...' : 'Creating model (confirm in wallet)...'}
            </p>
          </div>
        ) : (
          <>
            {/* Slug */}
            <div>
              <label className="label-text block mb-1.5">Slug</label>
              <input
                type="text"
                value={slug}
                onChange={(e) => handleSlugChange(e.target.value)}
                placeholder="my-mnist-model"
                className={cn(
                  'w-full px-3 py-2 bg-helix-bg border rounded-md text-sm text-helix-text font-mono focus:outline-none transition-colors',
                  slugError ? 'border-red-500/50' : 'border-helix-border focus:border-helix-border2',
                )}
              />
              {slugError && <p className="text-2xs text-red-400 mt-1">{slugError}</p>}
              <p className="text-2xs text-helix-dim mt-1">Unique identifier, kebab-case</p>
            </div>

            {/* Name */}
            <div>
              <label className="label-text block mb-1.5">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="MNIST Classifier"
                className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text focus:outline-none focus:border-helix-border2 transition-colors"
              />
            </div>

            {/* Description */}
            <div>
              <label className="label-text block mb-1.5">Description (optional)</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="784 to 128 to 10 neural network for handwritten digit classification"
                rows={3}
                className="w-full px-3 py-2 bg-helix-bg border border-helix-border rounded-md text-sm text-helix-text focus:outline-none focus:border-helix-border2 transition-colors resize-none"
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
              Create Model
            </button>
          </>
        )}
      </div>
    </Modal>
  );
}
