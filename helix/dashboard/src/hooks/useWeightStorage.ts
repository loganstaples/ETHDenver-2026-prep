'use client';

import { useCallback } from 'react';

const DB_NAME = 'helix-weights';
const DB_VERSION = 1;
const STORE_NAME = 'weights';

export interface StoredWeights {
  jobId: number;
  step: number;
  commitment: string;
  weights: ArrayBuffer;
  timestamp: number;
  status: 'stopped' | 'paused' | 'completed';
  modelName: string;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'jobId' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export function useWeightStorage() {
  const saveWeights = useCallback(async (data: StoredWeights) => {
    const db = await openDB();
    return new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(data);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }, []);

  const loadWeights = useCallback(async (jobId: number): Promise<StoredWeights | null> => {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const request = tx.objectStore(STORE_NAME).get(jobId);
      request.onsuccess = () => resolve(request.result ?? null);
      request.onerror = () => reject(request.error);
    });
  }, []);

  const deleteWeights = useCallback(async (jobId: number) => {
    const db = await openDB();
    return new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(jobId);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }, []);

  const listWeights = useCallback(async (): Promise<StoredWeights[]> => {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const request = tx.objectStore(STORE_NAME).getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }, []);

  const downloadWeights = useCallback(async (jobId: number) => {
    const data = await loadWeights(jobId);
    if (!data) return;
    const blob = new Blob([data.weights], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${data.modelName}-step${data.step}-weights.bin`;
    a.click();
    URL.revokeObjectURL(url);
  }, [loadWeights]);

  return { saveWeights, loadWeights, deleteWeights, listWeights, downloadWeights };
}
