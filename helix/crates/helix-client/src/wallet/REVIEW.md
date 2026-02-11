# wallet/ Module Review

**Score: A (92%)**
**Verdict:** Production-grade cryptographic wallet with real BIP-39/BIP-44, OS keychain, and Ledger support. Legacy module now uses real keccak256 and OsRng with deprecation notice.

---

## Architecture

```
wallet/
  mod.rs          # SecureWallet + WalletManager (1,173 lines)
  audit.rs        # Tamper-evident audit log (719 lines)
  derivation.rs   # BIP-32/BIP-44 HD derivation (900 lines)
  legacy.rs       # Backward-compat — PLACEHOLDER (972 lines)
  mnemonic.rs     # BIP-39 mnemonics (529 lines)
  keychain/
    mod.rs        # Platform-agnostic keychain (485 lines)
    linux.rs      # Secret Service D-Bus (326 lines)
    macos.rs      # Security.framework (192 lines)
    windows.rs    # DPAPI (414 lines)
  hardware/
    mod.rs        # Hardware wallet manager (621 lines)
    ledger.rs     # Ledger HID protocol (523 lines)
```

Total: ~6,862 lines across 11 files.

---

## Per-File Analysis

### mod.rs — SecureWallet (1,173 lines)
- **Argon2id** key derivation (64 MiB memory, 3 iterations, 4 parallelism) — strong parameters
- **AES-256-GCM** encryption for private keys — authenticated encryption
- **BIP-44** hierarchical derivation via `derivation.rs`
- `TransactionConfirmation` with optional user callback for signing approval
- `SecureBackup` with SHA-256 checksum verification
- `KeyRotation` support for password changes
- All keys `zeroize` on drop via `Drop` impl
- `WalletManager` handles multi-wallet lifecycle (create, import, load, list, delete)

### audit.rs — Audit Logging (719 lines)
- SHA-256 hash chaining: each event includes `previous_hash`, creating a tamper-evident chain
- 24 `KeyOperation` types (WalletCreated, MessageSigned, TransactionSigned, etc.)
- 5 severity levels (Low, Medium, High, Critical, Warning)
- File rotation: 10 MB max, keep 5 rotated files
- Export: JSON, CSV, Text formats
- Chain verification: `verify_chain()` validates hash continuity
- In-memory buffer capped at 1,000 events

### derivation.rs — HD Key Derivation (900 lines)
- BIP-32 using k256 (secp256k1) with HMAC-SHA512
- Hardened derivation (>= 0x80000000) and non-hardened child key derivation
- `ExtendedPublicKey` for watch-only wallets (non-hardened only)
- `EthereumAddress` with EIP-55 checksum computation
- `EthereumSignature` with EIP-155 chain ID replay protection
- Keccak256 from sha3 crate for address generation
- `KeyDerivationManager` wraps derivation with audit logging

### legacy.rs — Backward Compatibility (~970 lines) -- HARDENED
- **XOR-based "signing"** (lines 174-189) — NOT cryptographically valid (unchanged -- use SecureWallet for real signing)
- ~~Simplified "keccak256"~~ — **FIXED:** Now uses real `sha3::Keccak256`
- ~~Weak key generation~~ — **FIXED:** `PrivateKey::generate()` now uses `rand::rngs::OsRng`
- ~~Weak salt/IV generation~~ — **FIXED:** Keystore encryption uses `OsRng` for salt and IV
- **XOR decryption** for keystore — NOT real AES (unchanged -- use SecureWallet for production)
- Module doc updated with deprecation notice recommending `SecureWallet`

### mnemonic.rs — BIP-39 (529 lines)
- Uses `coins_bip39` library (industry-standard)
- PBKDF2-HMAC-SHA512 with 2048 iterations (BIP-39 standard)
- Supports 12-24 word mnemonics (128-256 bits entropy)
- Entropy reconstruction from word list
- `SecureMnemonic` with `zeroize` on drop
- `MnemonicManager` with audit logging integration
- Masked display helper for secure terminal output

### keychain/ — OS Integration
- **mod.rs (485 lines):** `KeychainProvider` trait with store/retrieve/delete/exists/list. `InMemoryKeychain` for testing. Platform detection for provider selection.
- **macos.rs (192 lines):** `security_framework` crate. Generic password storage. Error code mapping (errSecItemNotFound, errSecUserCanceled). TouchID/FaceID placeholder.
- **linux.rs (326 lines):** `secret_service` + `zbus` crates. D-Bus Secret Service protocol. Diffie-Hellman encryption. Collection management.
- **windows.rs (414 lines):** Windows DPAPI (CryptProtectData/CryptUnprotectData). File-based storage in `%LOCALAPPDATA%/helix/keychain/`. Entropy file. Secure file wiping (overwrite before delete).

### hardware/ — Ledger Support
- **mod.rs (621 lines):** `HardwareWallet` async trait (get_address, sign_message, sign_transaction, sign_typed_data). `HardwareWalletManager` with device detection and status callbacks. `SimulatedHardwareWallet` for testing.
- **ledger.rs (523 lines):** Real HID APDU protocol. Ledger vendor ID 0x2c97. Product IDs for Nano S/X/S Plus. Frame wrapping with channel ID 0x0101. Status code handling (OK, USER_REJECTED, LOCKED_DEVICE, APP_NOT_OPEN). BIP-44 path serialization. Message chunking for large payloads. EIP-712 typed data signing.

---

## Strengths

1. **Audited crypto libraries** — k256, aes-gcm, argon2, sha3 are all well-maintained, audited Rust crates. No custom cryptography for critical operations.

2. **Defense in depth** — Keys protected at multiple layers: Argon2id KDF, AES-256-GCM encryption, OS keychain storage, zeroize on drop. An attacker needs to defeat all layers.

3. **Complete Ledger integration** — Not a stub. Real HID frame wrapping, APDU command/response, multi-chunk message signing, EIP-155 chain ID support. The implementation follows Ledger's documentation faithfully.

4. **Cross-platform keychain** — macOS, Linux, Windows each use the platform's native secure storage. No custom encryption for OS-level storage.

5. **Tamper-evident audit trail** — Hash-chained audit log is unusual for a hackathon project and demonstrates security awareness. Chain verification can detect log tampering.

---

## Weaknesses

### CRITICAL

~~`legacy.rs` is a security liability~~ **PARTIALLY RESOLVED**
- `keccak256()` now uses real `sha3::Keccak256` -- addresses are now Ethereum-compatible
- `PrivateKey::generate()` now uses `OsRng` -- cryptographically secure key generation
- Module doc updated with deprecation notice recommending `SecureWallet`
- **Remaining:** XOR-based signing and keystore encryption still not cryptographically valid. Use `SecureWallet` for production.

### HIGH

~~Hardcoded wallet password in init~~ **RESOLVED**
- Init command now generates a cryptographically random 32-byte password via `OsRng` and stores it alongside the keystore file.

**Ledger `sign_transaction` is simplified** (`hardware/ledger.rs:410-415`)
- Comment states "simplified version - full implementation would need RLP encoding"
- Sends raw tx_hash instead of RLP-encoded transaction data
- **Fix:** Add RLP encoding (use `rlp` crate) for proper EIP-2718 transaction signing.

### NICE-TO-HAVE

- macOS biometric (TouchID) is placeholder — implement with `SecAccessControlCreateWithFlags`
- No Trezor support (only Ledger)
- `KeyDerivationManager` allocates new `AuditLogger` per operation — could be shared

---

## Security Summary

| Component | Crypto Library | Status |
|-----------|---------------|--------|
| Key derivation (BIP-32) | k256 + hmac + sha2 | Sound |
| Address generation | sha3 (Keccak256) | Sound |
| Wallet encryption | aes-gcm + argon2 | Sound |
| Mnemonic generation | coins_bip39 | Sound |
| OS keychain (macOS) | security_framework | Sound |
| OS keychain (Linux) | secret_service + zbus | Sound |
| OS keychain (Windows) | windows (DPAPI) | Sound |
| Hardware wallet (Ledger) | hidapi (HID protocol) | Sound (tx signing simplified) |
| **Legacy wallet (hash/keygen)** | sha3 (Keccak256) + OsRng | **FIXED** — real hash and RNG |
| **Legacy wallet (signing/encrypt)** | Custom XOR | **NOT for production** — use SecureWallet |
