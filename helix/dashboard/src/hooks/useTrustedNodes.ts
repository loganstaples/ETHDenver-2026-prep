'use client';

import { useState, useCallback, useEffect } from 'react';

const STORAGE_KEY = 'helix-trusted-nodes';
const MAX_TRUSTED_NODES = 100;

export interface UseTrustedNodesReturn {
  trustedNodes: string[];
  addNode: (address: string) => { ok: boolean; error?: string };
  removeNode: (address: string) => void;
  clearAll: () => void;
  isTrusted: (address: string) => boolean;
}

function isValidEthAddress(address: string): boolean {
  return /^0x[0-9a-fA-F]{40}$/.test(address);
}

function loadTrustedNodes(): string[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isValidEthAddress) : [];
  } catch {
    return [];
  }
}

function saveTrustedNodes(nodes: string[]) {
  if (typeof window === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(nodes));
}

/**
 * Read trusted node addresses from localStorage.
 * Standalone function for use outside React components (e.g. in callbacks).
 */
export function getTrustedNodes(): string[] {
  return loadTrustedNodes();
}

/**
 * Check whether at least one trusted node is among the currently active workers.
 * Returns `{ ok: true }` if the check passes (or if no trusted nodes are configured).
 * Returns `{ ok: false, error }` if trusted nodes are configured but none are active.
 * Returns `{ ok: true, warning }` if the workers API is unreachable.
 */
export async function checkTrustedWorkersActive(
  apiBase: string,
  trustedNodes: string[],
): Promise<{ ok: boolean; error?: string; warning?: string }> {
  if (trustedNodes.length === 0) return { ok: true };
  try {
    const res = await fetch(`${apiBase}/api/workers`);
    if (!res.ok) {
      return { ok: true, warning: 'Could not verify trusted workers (API unavailable). Proceeding without verification.' };
    }
    const data = await res.json();
    const workers: { address?: string; status?: string }[] =
      Array.isArray(data) ? data : (data.workers ?? []);
    const activeAddresses = workers
      .filter(w => w.status !== 'offline' && w.address)
      .map(w => (w.address as string).toLowerCase());
    const trustedActive = trustedNodes.filter(tn =>
      activeAddresses.includes(tn.toLowerCase())
    );
    if (trustedActive.length === 0) {
      return {
        ok: false,
        error: 'No trusted nodes are currently active. Add trusted nodes in Settings, or wait for at least one to come online.',
      };
    }
    return { ok: true };
  } catch {
    return { ok: true, warning: 'Could not verify trusted workers (network error). Proceeding without verification.' };
  }
}

export function useTrustedNodes(): UseTrustedNodesReturn {
  const [trustedNodes, setTrustedNodes] = useState<string[]>([]);

  // Load from localStorage on mount
  useEffect(() => {
    setTrustedNodes(loadTrustedNodes());
  }, []);

  const addNode = useCallback((address: string): { ok: boolean; error?: string } => {
    const trimmed = address.trim();
    if (!isValidEthAddress(trimmed)) {
      return { ok: false, error: 'Invalid Ethereum address (must be 0x + 40 hex chars)' };
    }
    const normalized = trimmed.toLowerCase();
    const current = loadTrustedNodes();
    if (current.length >= MAX_TRUSTED_NODES) {
      return { ok: false, error: `Maximum of ${MAX_TRUSTED_NODES} trusted nodes allowed` };
    }
    if (current.some(n => n.toLowerCase() === normalized)) {
      return { ok: false, error: 'Address already in trusted list' };
    }
    // Store normalized (lowercase) for consistent comparison
    const updated = [...current, normalized];
    saveTrustedNodes(updated);
    setTrustedNodes(updated);
    return { ok: true };
  }, []);

  const removeNode = useCallback((address: string) => {
    const normalized = address.toLowerCase();
    const current = loadTrustedNodes();
    const updated = current.filter(n => n.toLowerCase() !== normalized);
    saveTrustedNodes(updated);
    setTrustedNodes(updated);
  }, []);

  const clearAll = useCallback(() => {
    saveTrustedNodes([]);
    setTrustedNodes([]);
  }, []);

  const isTrusted = useCallback((address: string): boolean => {
    const normalized = address.toLowerCase();
    return trustedNodes.some(n => n.toLowerCase() === normalized);
  }, [trustedNodes]);

  return { trustedNodes, addNode, removeNode, clearAll, isTrusted };
}
