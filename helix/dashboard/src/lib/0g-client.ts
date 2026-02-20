/**
 * Client-side helpers for interacting with 0G Storage via the Next.js API routes.
 * These wrap the server-side routes that use @0glabs/0g-ts-sdk.
 */

const API_BASE = '';  // Same-origin API routes

export interface StoreOn0GParams {
  sessionId: string;
  weights?: Record<string, unknown>;
  encrypted?: boolean;
  encryptedPayload?: string;  // base64-encoded encrypted bytes
  accuracy?: number;
  version?: string;
}

export interface StoreOn0GResult {
  status: string;
  root_hash: string;
  tx_hash: string;
  explorer_url: string;
  retrieval_url: string;
  session_id: string;
}

export interface FetchFrom0GResult {
  data: string | Record<string, unknown>;
  encoding: 'json' | 'base64';
}

/** Upload model weights (encrypted or plaintext) to 0G Storage via API route */
export async function storeOn0G(params: StoreOn0GParams): Promise<StoreOn0GResult> {
  const res = await fetch(`${API_BASE}/api/store-on-0g`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Store on 0G failed: HTTP ${res.status}`);
  }
  return res.json();
}

/** Fetch model weights from 0G Storage by root hash */
export async function fetchFrom0G(rootHash: string): Promise<FetchFrom0GResult> {
  const res = await fetch(`${API_BASE}/api/fetch-from-0g`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rootHash }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Fetch from 0G failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Full flow: fetch encrypted weights from 0G, decrypt with owner's key.
 * Returns the plaintext weights as a parsed JSON object.
 */
export async function fetchAndDecryptWeights(
  rootHash: string,
  decryptFn: (combined: Uint8Array) => Promise<string>,
): Promise<Record<string, unknown>> {
  const result = await fetchFrom0G(rootHash);

  if (result.encoding === 'json') {
    // Plaintext JSON — return directly
    return result.data as Record<string, unknown>;
  }

  // Base64-encoded encrypted binary
  const binaryStr = atob(result.data as string);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }

  const plaintext = await decryptFn(bytes);
  return JSON.parse(plaintext);
}

/**
 * Full flow: encrypt weights and upload to 0G.
 * Returns the 0G storage result with root_hash.
 */
export async function encryptAndStoreWeights(
  sessionId: string,
  weightsJson: string,
  encryptFn: (data: string) => Promise<Uint8Array>,
  version?: string,
  accuracy?: number,
): Promise<StoreOn0GResult> {
  const encrypted = await encryptFn(weightsJson);

  // Convert to base64 for transport
  let binary = '';
  for (let i = 0; i < encrypted.length; i++) {
    binary += String.fromCharCode(encrypted[i]);
  }
  const encryptedPayload = btoa(binary);

  return storeOn0G({
    sessionId,
    encrypted: true,
    encryptedPayload,
    version,
    accuracy,
  });
}
