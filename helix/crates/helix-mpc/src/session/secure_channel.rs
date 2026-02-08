//! Encrypted communication channel using AES-GCM with DH-derived session keys.
//!
//! Wraps any [`MPCTransport`] to add authenticated encryption. Each pair of
//! parties performs an X25519 Diffie-Hellman key exchange, then derives a
//! symmetric AES-256-GCM key for encrypting all subsequent messages.

use std::collections::HashMap;

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, SharedSecret};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

use super::transport::MPCTransport;

/// Per-peer encryption state.
struct PeerCipher {
    cipher: Aes256Gcm,
    /// Send nonce counter (incremented after each message).
    send_counter: u64,
    /// Receive nonce counter (incremented after each message).
    recv_counter: u64,
}

/// DH public key message exchanged during key agreement.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DHPublicKeyMsg {
    pub_key: [u8; 32],
}

/// An encrypted wrapper around an [`MPCTransport`].
///
/// After construction via [`SecureChannel::establish`], all `send`/`recv` calls
/// are automatically encrypted with AES-256-GCM using per-peer session keys
/// derived from X25519 Diffie-Hellman.
pub struct SecureChannel<T: MPCTransport> {
    inner: T,
    /// Per-peer cipher state.
    peer_ciphers: HashMap<String, PeerCipher>,
}

impl<T: MPCTransport> SecureChannel<T> {
    /// Establishes encrypted channels with all peers.
    ///
    /// 1. Generates an ephemeral X25519 key pair (using StaticSecret to allow
    ///    multiple DH operations with different peers)
    /// 2. Exchanges public keys with all peers via the underlying transport
    /// 3. Derives per-peer AES-256-GCM keys from the shared secrets
    pub async fn establish(inner: T) -> MPCResult<Self> {
        let peers = inner.peers();
        let mut peer_ciphers = HashMap::new();

        // Generate session key pair. We use StaticSecret (not EphemeralSecret)
        // because we need to perform DH with multiple peers. The key is still
        // ephemeral — it exists only for this session.
        let secret = x25519_dalek::StaticSecret::random_from_rng(rand::thread_rng());
        let our_pub = PublicKey::from(&secret);

        let our_id = inner.party_id().0.clone();
        let our_msg = bincode::serialize(&DHPublicKeyMsg {
            pub_key: our_pub.to_bytes(),
        })
        .map_err(|e| MPCError::CommunicationError(format!("DH serialize: {}", e)))?;

        // Exchange public keys and derive session keys with each peer.
        // To avoid deadlock, the party with the smaller ID sends first.
        for peer in &peers {
            let their_pub_bytes = if our_id < peer.0 {
                // We send first, then receive
                inner.send(peer, &our_msg).await?;
                let resp = inner.recv(peer).await?;
                let dh_msg: DHPublicKeyMsg = bincode::deserialize(&resp).map_err(|e| {
                    MPCError::CommunicationError(format!("DH deserialize: {}", e))
                })?;
                dh_msg.pub_key
            } else {
                // We receive first, then send
                let resp = inner.recv(peer).await?;
                let dh_msg: DHPublicKeyMsg = bincode::deserialize(&resp).map_err(|e| {
                    MPCError::CommunicationError(format!("DH deserialize: {}", e))
                })?;
                inner.send(peer, &our_msg).await?;
                dh_msg.pub_key
            };

            let their_pub = PublicKey::from(their_pub_bytes);
            let shared = secret.diffie_hellman(&their_pub);
            let aes_key = derive_aes_key(&shared, &our_id, &peer.0);
            let cipher = Aes256Gcm::new_from_slice(&aes_key).map_err(|e| {
                MPCError::CommunicationError(format!("AES key init: {}", e))
            })?;
            peer_ciphers.insert(
                peer.0.clone(),
                PeerCipher {
                    cipher,
                    send_counter: 0,
                    recv_counter: 0,
                },
            );
        }

        Ok(Self {
            inner,
            peer_ciphers,
        })
    }

    /// Sends an encrypted message to a peer.
    pub async fn send_encrypted(&mut self, party: &PartyId, plaintext: &[u8]) -> MPCResult<()> {
        let pc = self.peer_ciphers.get_mut(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!("no cipher for party {}", party))
        })?;

        let nonce_bytes = counter_to_nonce(pc.send_counter);
        let nonce = Nonce::from_slice(&nonce_bytes);
        pc.send_counter += 1;

        let ciphertext = pc
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| MPCError::CommunicationError(format!("encrypt failed: {}", e)))?;

        self.inner.send(party, &ciphertext).await
    }

    /// Receives and decrypts a message from a peer.
    pub async fn recv_encrypted(&mut self, party: &PartyId) -> MPCResult<Vec<u8>> {
        let ciphertext = self.inner.recv(party).await?;

        let pc = self.peer_ciphers.get_mut(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!("no cipher for party {}", party))
        })?;

        let nonce_bytes = counter_to_nonce(pc.recv_counter);
        let nonce = Nonce::from_slice(&nonce_bytes);
        pc.recv_counter += 1;

        pc.cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| MPCError::CommunicationError(format!("decrypt failed: {}", e)))
    }

    /// Broadcasts an encrypted message to all peers.
    pub async fn broadcast_encrypted(&mut self, plaintext: &[u8]) -> MPCResult<()> {
        let peers = self.inner.peers();
        for peer in peers {
            self.send_encrypted(&peer, plaintext).await?;
        }
        Ok(())
    }

    /// Returns the local party's ID.
    pub fn party_id(&self) -> &PartyId {
        self.inner.party_id()
    }

    /// Returns peer party IDs.
    pub fn peers(&self) -> Vec<PartyId> {
        self.inner.peers()
    }
}

/// Derives a 256-bit AES key from the DH shared secret and both party IDs.
///
/// Uses SHA-256 as a simple KDF: `key = SHA-256(shared_secret || min(id_a, id_b) || max(id_a, id_b))`.
/// Sorting the IDs ensures both parties derive the same key regardless of role.
fn derive_aes_key(shared: &SharedSecret, id_a: &str, id_b: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(shared.as_bytes());
    // Sort to ensure both parties compute the same key
    if id_a < id_b {
        hasher.update(id_a.as_bytes());
        hasher.update(id_b.as_bytes());
    } else {
        hasher.update(id_b.as_bytes());
        hasher.update(id_a.as_bytes());
    }
    hasher.finalize().into()
}

/// Converts a counter to a 96-bit (12-byte) AES-GCM nonce.
fn counter_to_nonce(counter: u64) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[4..12].copy_from_slice(&counter.to_be_bytes());
    nonce
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::transport::LocalTransport;

    #[tokio::test]
    async fn test_secure_channel_establish() {
        let parties: Vec<PartyId> = (0..2).map(PartyId::from_index).collect();
        let mut transports = LocalTransport::create_mesh(&parties);

        // Establish in parallel
        let t0 = transports.remove(0);
        let t1 = transports.remove(0);

        let (sc0, sc1) = tokio::join!(
            SecureChannel::establish(t0),
            SecureChannel::establish(t1),
        );

        let mut sc0 = sc0.unwrap();
        let mut sc1 = sc1.unwrap();

        // Send encrypted message
        sc0.send_encrypted(&parties[1], b"secret hello")
            .await
            .unwrap();

        let decrypted = sc1.recv_encrypted(&parties[0]).await.unwrap();
        assert_eq!(decrypted, b"secret hello");
    }

    #[tokio::test]
    async fn test_secure_channel_bidirectional() {
        let parties: Vec<PartyId> = (0..2).map(PartyId::from_index).collect();
        let mut transports = LocalTransport::create_mesh(&parties);

        let t0 = transports.remove(0);
        let t1 = transports.remove(0);

        let (sc0, sc1) = tokio::join!(
            SecureChannel::establish(t0),
            SecureChannel::establish(t1),
        );

        let mut sc0 = sc0.unwrap();
        let mut sc1 = sc1.unwrap();

        // Party 0 → 1
        sc0.send_encrypted(&parties[1], b"msg from 0")
            .await
            .unwrap();
        let m1 = sc1.recv_encrypted(&parties[0]).await.unwrap();
        assert_eq!(m1, b"msg from 0");

        // Party 1 → 0
        sc1.send_encrypted(&parties[0], b"msg from 1")
            .await
            .unwrap();
        let m0 = sc0.recv_encrypted(&parties[1]).await.unwrap();
        assert_eq!(m0, b"msg from 1");
    }

    #[tokio::test]
    async fn test_secure_channel_three_parties() {
        let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
        let mut transports = LocalTransport::create_mesh(&parties);

        let t0 = transports.remove(0);
        let t1 = transports.remove(0);
        let t2 = transports.remove(0);

        let (sc0, sc1, sc2) = tokio::join!(
            SecureChannel::establish(t0),
            SecureChannel::establish(t1),
            SecureChannel::establish(t2),
        );

        let mut sc0 = sc0.unwrap();
        let mut sc1 = sc1.unwrap();
        let mut sc2 = sc2.unwrap();

        // Party 0 broadcasts
        sc0.broadcast_encrypted(b"from party 0").await.unwrap();

        let m1 = sc1.recv_encrypted(&parties[0]).await.unwrap();
        let m2 = sc2.recv_encrypted(&parties[0]).await.unwrap();
        assert_eq!(m1, b"from party 0");
        assert_eq!(m2, b"from party 0");
    }

    #[test]
    fn test_counter_to_nonce() {
        let n0 = counter_to_nonce(0);
        assert_eq!(n0, [0; 12]);

        let n1 = counter_to_nonce(1);
        assert_eq!(n1[11], 1);
        assert_eq!(n1[0..4], [0; 4]);
    }

    #[test]
    fn test_derive_aes_key_symmetric() {
        // Key derivation should be symmetric regardless of party order
        let secret = x25519_dalek::StaticSecret::random_from_rng(rand::thread_rng());
        let pub_key = PublicKey::from(&secret);
        let shared = secret.diffie_hellman(&pub_key);

        let k1 = derive_aes_key(&shared, "party-0", "party-1");
        let k2 = derive_aes_key(&shared, "party-1", "party-0");
        assert_eq!(k1, k2);
    }
}
