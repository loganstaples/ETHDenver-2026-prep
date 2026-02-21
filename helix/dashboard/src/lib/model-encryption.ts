/**
 * Model weight encryption for 0G Storage privacy.
 *
 * Flow:
 * 1. deriveModelKey(signMessage, tokenId) → CryptoKey
 *    - Asks wallet to sign "helix-model-key-{tokenId}"
 *    - SHA-256 of signature → 32-byte AES-GCM key
 * 2. encryptWeights(key, weightsJson) → Uint8Array (combined [12-byte IV][ciphertext])
 * 3. decryptWeights(key, combined) → weightsJson string
 *
 * NOTE: crypto.subtle requires a secure context (HTTPS or localhost).
 * When accessed via plain HTTP on a LAN IP, we fall back to a simple
 * XOR-based obfuscation so the demo flow doesn't break.
 */

const SIGN_PREFIX = 'helix-model-key-';

/** True when the Web Crypto subtle API is fully available (secure context). */
function hasSubtle(): boolean {
  try {
    return typeof globalThis?.crypto?.subtle?.digest === 'function';
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Simple fallback for non-secure contexts (HTTP over LAN).
// This is NOT real encryption — it's just enough to keep the mint flow
// working during a demo.  The "key" is derived from the signature string.
// ---------------------------------------------------------------------------

/** Derive a 32-byte key buffer from a signature string (simple hash). */
function simpleDeriveKey(signature: string): Uint8Array {
  const encoder = new TextEncoder();
  const sigBytes = encoder.encode(signature);
  const key = new Uint8Array(32);
  for (let i = 0; i < sigBytes.length; i++) {
    key[i % 32] ^= sigBytes[i];
  }
  return key;
}

function simpleEncrypt(keyBytes: Uint8Array, data: Uint8Array): Uint8Array {
  const iv = new Uint8Array(12);
  if (typeof globalThis.crypto !== 'undefined') {
    globalThis.crypto.getRandomValues(iv);
  } else {
    for (let i = 0; i < 12; i++) iv[i] = Math.floor(Math.random() * 256);
  }
  const out = new Uint8Array(12 + data.length);
  out.set(iv, 0);
  for (let i = 0; i < data.length; i++) {
    out[12 + i] = data[i] ^ keyBytes[i % 32] ^ iv[i % 12];
  }
  return out;
}

function simpleDecrypt(keyBytes: Uint8Array, combined: Uint8Array): Uint8Array {
  const iv = combined.slice(0, 12);
  const cipher = combined.slice(12);
  const out = new Uint8Array(cipher.length);
  for (let i = 0; i < cipher.length; i++) {
    out[i] = cipher[i] ^ keyBytes[i % 32] ^ iv[i % 12];
  }
  return out;
}

// We store the fallback key bytes on the CryptoKey-like wrapper so
// encrypt/decrypt can retrieve them.
const FALLBACK_KEY_SLOT = Symbol('fallbackKey');

type MaybeKey = CryptoKey | { [FALLBACK_KEY_SLOT]: Uint8Array };

function isFallbackKey(k: MaybeKey): k is { [FALLBACK_KEY_SLOT]: Uint8Array } {
  return FALLBACK_KEY_SLOT in k;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function deriveModelKey(
  signMessage: (message: string) => Promise<string>,
  tokenId: number | bigint,
): Promise<CryptoKey> {
  const message = `${SIGN_PREFIX}${tokenId}`;
  const signature = await signMessage(message);

  if (hasSubtle()) {
    try {
      const encoder = new TextEncoder();
      const sigBytes = encoder.encode(signature);
      const hashBuffer = await crypto.subtle.digest('SHA-256', sigBytes);
      return crypto.subtle.importKey(
        'raw',
        hashBuffer,
        { name: 'AES-GCM' },
        false,
        ['encrypt', 'decrypt'],
      );
    } catch (e) {
      console.warn('[model-encryption] crypto.subtle failed at runtime, using fallback:', e);
    }
  }

  // Fallback for non-secure context (HTTP LAN demo)
  console.warn('[model-encryption] crypto.subtle unavailable (non-HTTPS); using fallback obfuscation');
  const keyBytes = simpleDeriveKey(signature);
  return { [FALLBACK_KEY_SLOT]: keyBytes } as unknown as CryptoKey;
}

export async function encryptWeights(
  key: CryptoKey,
  weightsJson: string,
): Promise<Uint8Array> {
  const encoder = new TextEncoder();
  const data = encoder.encode(weightsJson);

  if (!isFallbackKey(key as MaybeKey) && hasSubtle()) {
    try {
      const iv = crypto.getRandomValues(new Uint8Array(12));
      const ciphertext = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv },
        key,
        data,
      );
      const combined = new Uint8Array(12 + ciphertext.byteLength);
      combined.set(iv, 0);
      combined.set(new Uint8Array(ciphertext), 12);
      return combined;
    } catch (e) {
      console.warn('[model-encryption] crypto.subtle.encrypt failed, using fallback:', e);
    }
  }

  // Fallback
  const fb = key as unknown as { [FALLBACK_KEY_SLOT]: Uint8Array };
  const keyBytes = fb[FALLBACK_KEY_SLOT] ?? simpleDeriveKey('');
  return simpleEncrypt(keyBytes, data);
}

export async function decryptWeights(
  key: CryptoKey,
  combined: Uint8Array,
): Promise<string> {
  const isFallback = isFallbackKey(key as MaybeKey);

  if (!isFallback && hasSubtle()) {
    // Real AES-GCM key — decrypt or fail loudly (no silent XOR fallback).
    const iv = combined.slice(0, 12);
    const ciphertext = combined.slice(12);
    const plaintext = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv },
      key,
      ciphertext,
    );
    return new TextDecoder().decode(plaintext);
  }

  // Fallback path — only used when data was encrypted with the XOR method
  // (non-secure context / HTTP LAN demo).
  const fb = key as unknown as { [FALLBACK_KEY_SLOT]: Uint8Array };
  const keyBytes = fb[FALLBACK_KEY_SLOT] ?? simpleDeriveKey('');
  const plainBytes = simpleDecrypt(keyBytes, combined);
  const result = new TextDecoder().decode(plainBytes);

  // Validate the XOR-decrypted output looks like JSON — if not, the key
  // is wrong or the data was AES-GCM encrypted in a different context.
  const trimmed = result.trimStart();
  if (trimmed.length === 0 || (trimmed[0] !== '{' && trimmed[0] !== '[')) {
    throw new Error(
      'Decryption produced invalid output. The weights may have been encrypted ' +
      'with a different key or encryption method.',
    );
  }

  return result;
}

/** Convert a Uint8Array to hex string for display */
export function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/** Convert hex string back to Uint8Array */
export function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return bytes;
}
