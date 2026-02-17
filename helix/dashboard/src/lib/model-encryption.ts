/**
 * Model weight encryption for 0G Storage privacy.
 *
 * Flow:
 * 1. deriveModelKey(signMessage, tokenId) → CryptoKey
 *    - Asks wallet to sign "helix-model-key-{tokenId}"
 *    - SHA-256 of signature → 32-byte AES-GCM key
 * 2. encryptWeights(key, weightsJson) → Uint8Array (combined [12-byte IV][ciphertext])
 * 3. decryptWeights(key, combined) → weightsJson string
 */

const SIGN_PREFIX = 'helix-model-key-';

export async function deriveModelKey(
  signMessage: (message: string) => Promise<string>,
  tokenId: number | bigint,
): Promise<CryptoKey> {
  const message = `${SIGN_PREFIX}${tokenId}`;
  const signature = await signMessage(message);

  // SHA-256 the signature to get 32 bytes
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
}

export async function encryptWeights(
  key: CryptoKey,
  weightsJson: string,
): Promise<Uint8Array> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encoder = new TextEncoder();
  const data = encoder.encode(weightsJson);

  const ciphertext = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    data,
  );

  // Combine: [12-byte IV][ciphertext]
  const combined = new Uint8Array(12 + ciphertext.byteLength);
  combined.set(iv, 0);
  combined.set(new Uint8Array(ciphertext), 12);
  return combined;
}

export async function decryptWeights(
  key: CryptoKey,
  combined: Uint8Array,
): Promise<string> {
  const iv = combined.slice(0, 12);
  const ciphertext = combined.slice(12);

  const plaintext = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv },
    key,
    ciphertext,
  );

  const decoder = new TextDecoder();
  return decoder.decode(plaintext);
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
