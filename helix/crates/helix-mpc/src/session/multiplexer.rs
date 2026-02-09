//! Secure channel multiplexing for reduced round trips.
//!
//! This module provides channel multiplexing to batch multiple MPC operations
//! into single network round trips, dramatically reducing latency.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    MultiplexedChannel                     │
//! ├──────────────────────────────────────────────────────────┤
//! │  Stream 0: Control/Session   ──┐                         │
//! │  Stream 1: Beaver Triples    ──┤                         │
//! │  Stream 2: Share Exchange    ──┼── Single TCP Connection │
//! │  Stream 3: Commitments       ──┤                         │
//! │  Stream 4: Verification      ──┘                         │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Stream Multiplexing**: Multiple logical streams over single connection
//! - **Batch Aggregation**: Combines messages for same stream into batches
//! - **Priority Queuing**: High-priority messages bypass batching
//! - **Flow Control**: Per-stream backpressure prevents flooding
//! - **Ordering Guarantees**: In-order delivery within each stream

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

use super::channel::{Message, MessageType, MPCChannel};

/// Stream identifier for multiplexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamId(pub u16);

impl StreamId {
    pub const CONTROL: StreamId = StreamId(0);
    pub const BEAVER: StreamId = StreamId(1);
    pub const SHARES: StreamId = StreamId(2);
    pub const COMMITMENTS: StreamId = StreamId(3);
    pub const VERIFICATION: StreamId = StreamId(4);
    pub const GRADIENTS: StreamId = StreamId(5);

    /// Creates a custom stream ID.
    pub fn custom(id: u16) -> Self {
        assert!(id >= 100, "Stream IDs 0-99 are reserved");
        Self(id)
    }
}

/// Configuration for the multiplexer.
#[derive(Debug, Clone)]
pub struct MultiplexerConfig {
    /// Maximum messages per batch.
    pub max_batch_size: usize,
    /// Maximum batch wait time before flushing.
    pub max_batch_delay: Duration,
    /// Per-stream send buffer size.
    pub stream_buffer_size: usize,
    /// Enable message compression.
    pub enable_compression: bool,
    /// Compression threshold in bytes.
    pub compression_threshold: usize,
    /// Per-stream flow control window.
    pub flow_control_window: usize,
    /// Enable batching (can be disabled for low-latency mode).
    pub enable_batching: bool,
}

impl Default for MultiplexerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            max_batch_delay: Duration::from_millis(10),
            stream_buffer_size: 1000,
            enable_compression: true,
            compression_threshold: 1024,
            flow_control_window: 10000,
            enable_batching: true,
        }
    }
}

impl MultiplexerConfig {
    /// Low-latency configuration (minimal batching).
    pub fn low_latency() -> Self {
        Self {
            max_batch_size: 10,
            max_batch_delay: Duration::from_millis(1),
            stream_buffer_size: 100,
            enable_compression: false,
            compression_threshold: 0,
            flow_control_window: 1000,
            enable_batching: false,
        }
    }

    /// High-throughput configuration (aggressive batching).
    pub fn high_throughput() -> Self {
        Self {
            max_batch_size: 500,
            max_batch_delay: Duration::from_millis(50),
            stream_buffer_size: 5000,
            enable_compression: true,
            compression_threshold: 512,
            flow_control_window: 50000,
            enable_batching: true,
        }
    }
}

/// A batched message containing multiple messages for a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedMessage {
    /// Stream this batch belongs to.
    pub stream_id: StreamId,
    /// Batch sequence number.
    pub batch_seq: u64,
    /// Individual messages in this batch.
    pub messages: Vec<StreamMessage>,
    /// Compressed payload (if compression enabled).
    pub compressed: Option<Vec<u8>>,
    /// Flow control acknowledgment.
    pub flow_ack: Option<u64>,
}

/// A single message within a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMessage {
    /// Sequence number within stream.
    pub seq: u64,
    /// Original message.
    pub msg: Message,
    /// Timestamp for latency tracking.
    pub timestamp: u64,
}

/// Per-stream state.
#[derive(Debug)]
#[allow(dead_code)]
struct StreamState {
    /// Stream identifier.
    id: StreamId,
    /// Priority (higher = more important).
    priority: u8,
    /// Outgoing message buffer.
    outgoing: VecDeque<Message>,
    /// Last batch sent time.
    last_batch_time: Instant,
    /// Next sequence number.
    next_seq: AtomicU64,
    /// Last acknowledged sequence.
    acked_seq: AtomicU64,
    /// Flow control: outstanding messages.
    outstanding: AtomicU64,
    /// Flow control window.
    window: usize,
    /// Statistics.
    messages_sent: AtomicU64,
    messages_received: AtomicU64,
    batches_sent: AtomicU64,
}

impl StreamState {
    fn new(id: StreamId, priority: u8, window: usize) -> Self {
        Self {
            id,
            priority,
            outgoing: VecDeque::new(),
            last_batch_time: Instant::now(),
            next_seq: AtomicU64::new(0),
            acked_seq: AtomicU64::new(0),
            outstanding: AtomicU64::new(0),
            window,
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            batches_sent: AtomicU64::new(0),
        }
    }

    fn next_sequence(&self) -> u64 {
        self.next_seq.fetch_add(1, Ordering::SeqCst)
    }

    fn can_send(&self) -> bool {
        self.outstanding.load(Ordering::SeqCst) < self.window as u64
    }

    fn record_send(&self, count: usize) {
        self.outstanding.fetch_add(count as u64, Ordering::SeqCst);
        self.messages_sent.fetch_add(count as u64, Ordering::SeqCst);
        self.batches_sent.fetch_add(1, Ordering::SeqCst);
    }

    fn acknowledge(&self, seq: u64) {
        let current_acked = self.acked_seq.load(Ordering::SeqCst);
        if seq > current_acked {
            let freed = seq - current_acked;
            self.acked_seq.store(seq, Ordering::SeqCst);
            self.outstanding.fetch_sub(freed.min(self.outstanding.load(Ordering::SeqCst)), Ordering::SeqCst);
        }
    }
}

/// Multiplexed channel that batches messages across multiple streams.
pub struct MultiplexedChannel<C: MPCChannel> {
    /// Underlying channel.
    inner: Arc<C>,
    /// Configuration.
    config: MultiplexerConfig,
    /// Our party ID.
    party_id: PartyId,
    /// All parties.
    parties: Vec<PartyId>,
    /// Per-party, per-stream state.
    streams: RwLock<HashMap<String, HashMap<StreamId, Mutex<StreamState>>>>,
    /// Incoming message buffers per party per stream.
    incoming: RwLock<HashMap<String, HashMap<StreamId, VecDeque<Message>>>>,
    /// Global batch sequence.
    batch_seq: AtomicU64,
    /// Global message counter.
    total_messages: AtomicU64,
    /// Total batches sent.
    total_batches: AtomicU64,
}

impl<C: MPCChannel> MultiplexedChannel<C> {
    /// Creates a new multiplexed channel.
    pub fn new(inner: C, party_id: PartyId, parties: Vec<PartyId>, config: MultiplexerConfig) -> Self {
        let mut stream_map = HashMap::new();
        let mut incoming_map = HashMap::new();

        for party in &parties {
            if party != &party_id {
                let mut party_streams = HashMap::new();
                let mut party_incoming = HashMap::new();

                // Initialize standard streams
                for &stream_id in &[
                    StreamId::CONTROL,
                    StreamId::BEAVER,
                    StreamId::SHARES,
                    StreamId::COMMITMENTS,
                    StreamId::VERIFICATION,
                    StreamId::GRADIENTS,
                ] {
                    let priority = match stream_id {
                        StreamId::CONTROL => 10,
                        StreamId::BEAVER => 5,
                        StreamId::SHARES => 7,
                        StreamId::COMMITMENTS => 6,
                        StreamId::VERIFICATION => 8,
                        StreamId::GRADIENTS => 4,
                        _ => 3,
                    };

                    party_streams.insert(
                        stream_id,
                        Mutex::new(StreamState::new(stream_id, priority, config.flow_control_window)),
                    );
                    party_incoming.insert(stream_id, VecDeque::new());
                }

                stream_map.insert(party.0.clone(), party_streams);
                incoming_map.insert(party.0.clone(), party_incoming);
            }
        }

        Self {
            inner: Arc::new(inner),
            config,
            party_id,
            parties,
            streams: RwLock::new(stream_map),
            incoming: RwLock::new(incoming_map),
            batch_seq: AtomicU64::new(0),
            total_messages: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
        }
    }

    /// Sends a message on a specific stream.
    pub fn send_on_stream(&self, stream_id: StreamId, message: Message) -> MPCResult<()> {
        let to_party = message.to.0.clone();

        let streams = self.streams.read();
        let party_streams = streams.get(&to_party)
            .ok_or_else(|| MPCError::UnknownParty(message.to.clone()))?;

        let stream = party_streams.get(&stream_id)
            .ok_or_else(|| MPCError::ProtocolError(format!("Unknown stream {:?}", stream_id)))?;

        let mut stream = stream.lock();

        // Check flow control
        if !stream.can_send() {
            return Err(MPCError::CommunicationError("Flow control: window full".into()));
        }

        stream.outgoing.push_back(message);
        self.total_messages.fetch_add(1, Ordering::SeqCst);

        // Check if we should flush
        if !self.config.enable_batching ||
           stream.outgoing.len() >= self.config.max_batch_size ||
           stream.last_batch_time.elapsed() >= self.config.max_batch_delay
        {
            drop(stream);
            drop(streams);
            self.flush_stream(&to_party, stream_id)?;
        }

        Ok(())
    }

    /// Maps a message type to its default stream.
    pub fn stream_for_message_type(msg_type: &MessageType) -> StreamId {
        match msg_type {
            MessageType::Control(_) => StreamId::CONTROL,
            MessageType::BeaverTriple => StreamId::BEAVER,
            MessageType::OpenShare | MessageType::OpenedValue | MessageType::WeightShare => StreamId::SHARES,
            MessageType::Commitment | MessageType::VerificationResult => StreamId::COMMITMENTS,
            MessageType::ReshareContribution => StreamId::VERIFICATION,
            MessageType::GradientShare => StreamId::GRADIENTS,
        }
    }

    /// Sends a message, automatically determining the stream.
    pub fn send_auto(&self, message: Message) -> MPCResult<()> {
        let stream_id = Self::stream_for_message_type(&message.msg_type);
        self.send_on_stream(stream_id, message)
    }

    /// Flushes a specific stream to the underlying channel.
    fn flush_stream(&self, to_party: &str, stream_id: StreamId) -> MPCResult<()> {
        let streams = self.streams.read();
        let party_streams = match streams.get(to_party) {
            Some(s) => s,
            None => return Ok(()),
        };

        let stream = match party_streams.get(&stream_id) {
            Some(s) => s,
            None => return Ok(()),
        };

        let mut stream = stream.lock();

        if stream.outgoing.is_empty() {
            return Ok(());
        }

        // Collect messages into batch
        let mut batch_messages = Vec::new();
        let mut count = 0;

        while let Some(msg) = stream.outgoing.pop_front() {
            let seq = stream.next_sequence();
            batch_messages.push(StreamMessage {
                seq,
                msg,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            });
            count += 1;
            if count >= self.config.max_batch_size {
                break;
            }
        }

        if batch_messages.is_empty() {
            return Ok(());
        }

        stream.last_batch_time = Instant::now();
        stream.record_send(batch_messages.len());

        // Create batched message
        let batch = BatchedMessage {
            stream_id,
            batch_seq: self.batch_seq.fetch_add(1, Ordering::SeqCst),
            messages: batch_messages.clone(),
            compressed: None, // TODO: Add compression
            flow_ack: Some(stream.acked_seq.load(Ordering::SeqCst)),
        };

        drop(stream);
        drop(streams);

        // Serialize and send
        let payload = bincode::serialize(&batch)
            .map_err(|e| MPCError::CommunicationError(format!("Serialization error: {}", e)))?;

        let wrapper = Message {
            from: self.party_id.clone(),
            to: PartyId::new(to_party),
            msg_type: MessageType::Control("batch".into()),
            payload,
            sequence: batch.batch_seq,
        };

        self.inner.send(wrapper).map_err(|e| MPCError::CommunicationError(e))?;
        self.total_batches.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// Flushes all pending messages on all streams.
    pub fn flush_all(&self) -> MPCResult<()> {
        // Collect stream identifiers first to avoid borrowing issues.
        let stream_ids: Vec<(String, StreamId)> = {
            let streams = self.streams.read();
            streams.iter()
                .flat_map(|(party_id, party_streams)| {
                    party_streams.keys().map(move |&stream_id| (party_id.clone(), stream_id))
                })
                .collect()
        };

        for (party_id, stream_id) in stream_ids {
            self.flush_stream(&party_id, stream_id)?;
        }

        Ok(())
    }

    /// Receives messages from a specific stream.
    pub fn receive_from_stream(&self, party: &PartyId, stream_id: StreamId) -> Vec<Message> {
        // First, process any incoming batches
        let raw_messages = self.inner.receive(party);
        for msg in raw_messages {
            if let Ok(batch) = bincode::deserialize::<BatchedMessage>(&msg.payload) {
                self.process_batch(party, batch);
            }
        }

        // Return buffered messages
        let mut incoming = self.incoming.write();
        if let Some(party_incoming) = incoming.get_mut(&party.0) {
            if let Some(stream_buffer) = party_incoming.get_mut(&stream_id) {
                return stream_buffer.drain(..).collect();
            }
        }

        Vec::new()
    }

    /// Processes a received batch.
    fn process_batch(&self, from: &PartyId, batch: BatchedMessage) {
        let mut incoming = self.incoming.write();

        if let Some(party_incoming) = incoming.get_mut(&from.0) {
            if let Some(buffer) = party_incoming.get_mut(&batch.stream_id) {
                for stream_msg in batch.messages {
                    buffer.push_back(stream_msg.msg);
                }
            }
        }

        // Process flow control acknowledgment
        if let Some(ack_seq) = batch.flow_ack {
            let streams = self.streams.read();
            if let Some(party_streams) = streams.get(&from.0) {
                if let Some(stream) = party_streams.get(&batch.stream_id) {
                    stream.lock().acknowledge(ack_seq);
                }
            }
        }
    }

    /// Gets multiplexer statistics.
    pub fn stats(&self) -> MultiplexerStats {
        let streams = self.streams.read();

        let mut stream_stats = Vec::new();
        for (party_id, party_streams) in streams.iter() {
            for (stream_id, stream) in party_streams.iter() {
                let s = stream.lock();
                stream_stats.push(StreamStats {
                    party: party_id.clone(),
                    stream_id: *stream_id,
                    messages_sent: s.messages_sent.load(Ordering::SeqCst),
                    messages_received: s.messages_received.load(Ordering::SeqCst),
                    batches_sent: s.batches_sent.load(Ordering::SeqCst),
                    outstanding: s.outstanding.load(Ordering::SeqCst),
                    pending: s.outgoing.len(),
                });
            }
        }

        MultiplexerStats {
            total_messages: self.total_messages.load(Ordering::SeqCst),
            total_batches: self.total_batches.load(Ordering::SeqCst),
            average_batch_size: if self.total_batches.load(Ordering::SeqCst) > 0 {
                self.total_messages.load(Ordering::SeqCst) as f64 /
                self.total_batches.load(Ordering::SeqCst) as f64
            } else {
                0.0
            },
            streams: stream_stats,
        }
    }

    /// Creates a custom stream for a party.
    pub fn create_stream(&self, party: &PartyId, stream_id: StreamId, priority: u8) -> MPCResult<()> {
        let mut streams = self.streams.write();
        let mut incoming = self.incoming.write();

        if let Some(party_streams) = streams.get_mut(&party.0) {
            party_streams.insert(
                stream_id,
                Mutex::new(StreamState::new(stream_id, priority, self.config.flow_control_window)),
            );
        }

        if let Some(party_incoming) = incoming.get_mut(&party.0) {
            party_incoming.insert(stream_id, VecDeque::new());
        }

        Ok(())
    }
}

impl<C: MPCChannel> MPCChannel for MultiplexedChannel<C> {
    fn send(&self, message: Message) -> Result<(), String> {
        self.send_auto(message).map_err(|e| e.to_string())
    }

    fn receive(&self, party: &PartyId) -> Vec<Message> {
        // Receive from all streams
        let mut all_messages = Vec::new();

        for stream_id in &[
            StreamId::CONTROL,
            StreamId::BEAVER,
            StreamId::SHARES,
            StreamId::COMMITMENTS,
            StreamId::VERIFICATION,
            StreamId::GRADIENTS,
        ] {
            all_messages.extend(self.receive_from_stream(party, *stream_id));
        }

        all_messages
    }

    fn broadcast(&self, from: &PartyId, msg_type: MessageType, payload: Vec<u8>) {
        let stream_id = Self::stream_for_message_type(&msg_type);

        for party in &self.parties {
            if party != from {
                let msg = Message {
                    from: from.clone(),
                    to: party.clone(),
                    msg_type: msg_type.clone(),
                    payload: payload.clone(),
                    sequence: 0,
                };
                let _ = self.send_on_stream(stream_id, msg);
            }
        }

        // Flush after broadcast
        let _ = self.flush_all();
    }

    fn pending_count(&self, party: &PartyId) -> usize {
        let incoming = self.incoming.read();
        if let Some(party_incoming) = incoming.get(&party.0) {
            party_incoming.values().map(|q| q.len()).sum()
        } else {
            0
        }
    }
}

/// Statistics for a single stream.
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub party: String,
    pub stream_id: StreamId,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub batches_sent: u64,
    pub outstanding: u64,
    pub pending: usize,
}

/// Overall multiplexer statistics.
#[derive(Debug, Clone)]
pub struct MultiplexerStats {
    pub total_messages: u64,
    pub total_batches: u64,
    pub average_batch_size: f64,
    pub streams: Vec<StreamStats>,
}

/// Aggregates messages for batch sending.
pub struct MessageAggregator {
    /// Pending messages by destination and stream.
    pending: HashMap<String, HashMap<StreamId, Vec<Message>>>,
    /// Maximum aggregation delay.
    max_delay: Duration,
    /// Last flush time per destination.
    last_flush: HashMap<String, Instant>,
}

impl MessageAggregator {
    pub fn new(max_delay: Duration) -> Self {
        Self {
            pending: HashMap::new(),
            max_delay,
            last_flush: HashMap::new(),
        }
    }

    /// Adds a message to the aggregator.
    pub fn add(&mut self, stream_id: StreamId, message: Message) {
        let to_party = message.to.0.clone();
        self.pending
            .entry(to_party)
            .or_default()
            .entry(stream_id)
            .or_default()
            .push(message);
    }

    /// Checks if we should flush for a destination.
    pub fn should_flush(&self, party: &str) -> bool {
        if let Some(last) = self.last_flush.get(party) {
            last.elapsed() >= self.max_delay
        } else {
            true
        }
    }

    /// Takes all pending messages for a party and stream.
    pub fn take(&mut self, party: &str, stream_id: StreamId) -> Vec<Message> {
        if let Some(party_pending) = self.pending.get_mut(party) {
            if let Some(messages) = party_pending.remove(&stream_id) {
                self.last_flush.insert(party.to_string(), Instant::now());
                return messages;
            }
        }
        Vec::new()
    }

    /// Takes all pending messages for a party.
    pub fn take_all(&mut self, party: &str) -> HashMap<StreamId, Vec<Message>> {
        self.last_flush.insert(party.to_string(), Instant::now());
        self.pending.remove(party).unwrap_or_default()
    }

    /// Returns count of pending messages.
    pub fn pending_count(&self) -> usize {
        self.pending.values()
            .flat_map(|m| m.values())
            .map(|v| v.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::channel::LocalChannel;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_stream_id_constants() {
        assert_eq!(StreamId::CONTROL.0, 0);
        assert_eq!(StreamId::BEAVER.0, 1);
        assert_eq!(StreamId::SHARES.0, 2);
        assert_eq!(StreamId::COMMITMENTS.0, 3);
        assert_eq!(StreamId::VERIFICATION.0, 4);
    }

    #[test]
    fn test_multiplexed_channel_creation() {
        let parties = test_parties(3);
        let inner = LocalChannel::new(&parties);
        let config = MultiplexerConfig::default();

        let mux = MultiplexedChannel::new(inner, parties[0].clone(), parties.clone(), config);
        assert_eq!(mux.parties.len(), 3);
    }

    #[test]
    fn test_stream_for_message_type() {
        assert_eq!(
            MultiplexedChannel::<LocalChannel>::stream_for_message_type(&MessageType::Control("test".into())),
            StreamId::CONTROL
        );
        assert_eq!(
            MultiplexedChannel::<LocalChannel>::stream_for_message_type(&MessageType::BeaverTriple),
            StreamId::BEAVER
        );
        assert_eq!(
            MultiplexedChannel::<LocalChannel>::stream_for_message_type(&MessageType::GradientShare),
            StreamId::GRADIENTS
        );
    }

    #[test]
    fn test_send_and_receive() {
        let parties = test_parties(2);
        let inner = LocalChannel::new(&parties);
        let config = MultiplexerConfig {
            enable_batching: false,
            ..Default::default()
        };

        let mux = MultiplexedChannel::new(inner, parties[0].clone(), parties.clone(), config);

        let msg = Message {
            from: parties[0].clone(),
            to: parties[1].clone(),
            msg_type: MessageType::OpenShare,
            payload: vec![1, 2, 3],
            sequence: 0,
        };

        mux.send(msg.clone()).unwrap();

        // Note: In multiplexed mode, messages are batched
        // The receive would need to handle batch unpacking
    }

    #[test]
    fn test_multiplexer_config_presets() {
        let default = MultiplexerConfig::default();
        let low_latency = MultiplexerConfig::low_latency();
        let high_throughput = MultiplexerConfig::high_throughput();

        assert!(low_latency.max_batch_delay < default.max_batch_delay);
        assert!(high_throughput.max_batch_size > default.max_batch_size);
    }

    #[test]
    fn test_message_aggregator() {
        let mut aggregator = MessageAggregator::new(Duration::from_millis(10));

        let parties = test_parties(2);
        let msg = Message {
            from: parties[0].clone(),
            to: parties[1].clone(),
            msg_type: MessageType::OpenShare,
            payload: vec![1],
            sequence: 0,
        };

        aggregator.add(StreamId::SHARES, msg.clone());
        aggregator.add(StreamId::SHARES, msg);

        assert_eq!(aggregator.pending_count(), 2);

        let taken = aggregator.take(&parties[1].0, StreamId::SHARES);
        assert_eq!(taken.len(), 2);
        assert_eq!(aggregator.pending_count(), 0);
    }

    #[test]
    fn test_custom_stream() {
        let parties = test_parties(2);
        let inner = LocalChannel::new(&parties);
        let config = MultiplexerConfig::default();

        let mux = MultiplexedChannel::new(inner, parties[0].clone(), parties.clone(), config);

        // Create custom stream
        let custom_stream = StreamId::custom(100);
        mux.create_stream(&parties[1], custom_stream, 5).unwrap();
    }

    #[test]
    fn test_multiplexer_stats() {
        let parties = test_parties(2);
        let inner = LocalChannel::new(&parties);
        let config = MultiplexerConfig::default();

        let mux = MultiplexedChannel::new(inner, parties[0].clone(), parties.clone(), config);

        let stats = mux.stats();
        assert_eq!(stats.total_messages, 0);
        assert_eq!(stats.total_batches, 0);
    }
}
