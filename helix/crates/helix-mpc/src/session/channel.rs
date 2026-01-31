//! Communication channels for MPC message passing.
//!
//! Provides an abstraction over the communication layer between parties.
//! The `LocalChannel` implementation uses in-memory queues for local testing
//! and demos. A network-based implementation would use TCP/TLS connections.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::types::PartyId;

/// A message exchanged between MPC parties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Sender.
    pub from: PartyId,
    /// Recipient.
    pub to: PartyId,
    /// Message type tag.
    pub msg_type: MessageType,
    /// Payload (serialized data).
    pub payload: Vec<u8>,
    /// Sequence number for ordering.
    pub sequence: u64,
}

/// Types of messages exchanged during MPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Share of a value to be opened (Beaver d/e shares).
    OpenShare,
    /// Reconstructed (opened) value.
    OpenedValue,
    /// Re-sharing zero-share contribution.
    ReshareContribution,
    /// Beaver triple share distribution.
    BeaverTriple,
    /// Share commitment.
    Commitment,
    /// Commitment verification result.
    VerificationResult,
    /// Session control message (join, leave, ready, etc.).
    Control(String),
    /// Gradient share for aggregation.
    GradientShare,
    /// Model weight share.
    WeightShare,
}

/// Trait for MPC communication channels.
pub trait MPCChannel: Send + Sync {
    /// Sends a message to a specific party.
    fn send(&self, message: Message) -> Result<(), String>;

    /// Receives all pending messages for a party.
    fn receive(&self, party: &PartyId) -> Vec<Message>;

    /// Broadcasts a message to all other parties.
    fn broadcast(&self, from: &PartyId, msg_type: MessageType, payload: Vec<u8>);

    /// Returns the number of pending messages for a party.
    fn pending_count(&self, party: &PartyId) -> usize;
}

/// In-memory channel for local testing and demos.
///
/// All parties share the same process; messages are stored in a shared HashMap.
#[derive(Debug, Clone)]
pub struct LocalChannel {
    /// Mailboxes: party_id → list of messages.
    mailboxes: Arc<Mutex<HashMap<String, Vec<Message>>>>,
    /// All registered parties.
    parties: Vec<PartyId>,
    /// Sequence counter.
    sequence: Arc<Mutex<u64>>,
}

impl LocalChannel {
    /// Creates a new local channel for the given parties.
    pub fn new(parties: &[PartyId]) -> Self {
        let mut mailboxes = HashMap::new();
        for p in parties {
            mailboxes.insert(p.0.clone(), Vec::new());
        }

        Self {
            mailboxes: Arc::new(Mutex::new(mailboxes)),
            parties: parties.to_vec(),
            sequence: Arc::new(Mutex::new(0)),
        }
    }

    fn next_sequence(&self) -> u64 {
        let mut seq = self.sequence.lock().unwrap();
        let val = *seq;
        *seq += 1;
        val
    }
}

impl MPCChannel for LocalChannel {
    fn send(&self, message: Message) -> Result<(), String> {
        let mut mailboxes = self.mailboxes.lock().unwrap();
        mailboxes
            .entry(message.to.0.clone())
            .or_default()
            .push(message);
        Ok(())
    }

    fn receive(&self, party: &PartyId) -> Vec<Message> {
        let mut mailboxes = self.mailboxes.lock().unwrap();
        mailboxes
            .get_mut(&party.0)
            .map(|msgs| std::mem::take(msgs))
            .unwrap_or_default()
    }

    fn broadcast(&self, from: &PartyId, msg_type: MessageType, payload: Vec<u8>) {
        let seq = self.next_sequence();
        let mut mailboxes = self.mailboxes.lock().unwrap();

        for party in &self.parties {
            if party != from {
                let msg = Message {
                    from: from.clone(),
                    to: party.clone(),
                    msg_type: msg_type.clone(),
                    payload: payload.clone(),
                    sequence: seq,
                };
                mailboxes
                    .entry(party.0.clone())
                    .or_default()
                    .push(msg);
            }
        }
    }

    fn pending_count(&self, party: &PartyId) -> usize {
        let mailboxes = self.mailboxes.lock().unwrap();
        mailboxes.get(&party.0).map_or(0, |msgs| msgs.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_local_channel_send_receive() {
        let parties = test_parties(3);
        let channel = LocalChannel::new(&parties);

        let msg = Message {
            from: parties[0].clone(),
            to: parties[1].clone(),
            msg_type: MessageType::OpenShare,
            payload: vec![1, 2, 3],
            sequence: 0,
        };

        channel.send(msg).unwrap();
        assert_eq!(channel.pending_count(&parties[1]), 1);
        assert_eq!(channel.pending_count(&parties[0]), 0);

        let received = channel.receive(&parties[1]);
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].payload, vec![1, 2, 3]);
        assert_eq!(channel.pending_count(&parties[1]), 0);
    }

    #[test]
    fn test_broadcast() {
        let parties = test_parties(3);
        let channel = LocalChannel::new(&parties);

        channel.broadcast(
            &parties[0],
            MessageType::Control("ready".into()),
            vec![],
        );

        // Party 0 should NOT receive its own broadcast.
        assert_eq!(channel.pending_count(&parties[0]), 0);
        // Parties 1 and 2 should each get one message.
        assert_eq!(channel.pending_count(&parties[1]), 1);
        assert_eq!(channel.pending_count(&parties[2]), 1);
    }
}
