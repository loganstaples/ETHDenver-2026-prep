//! Beaver triple generation and management for secure multiplication.
//!
//! A Beaver triple is a tuple (a, b, c) of shared values where c = a * b.
//! These are pre-generated and consumed during the online phase to enable
//! secure multiplication of secret-shared values without revealing them.
//!
//! # The Beaver Multiplication Protocol
//!
//! To compute [x * y] from shares [x] and [y]:
//! 1. Consume a pre-generated triple ([a], [b], [c]) where c = a*b
//! 2. Each party locally computes [d] = [x] - [a] and [e] = [y] - [b]
//! 3. Parties reconstruct d and e (these are random-looking, reveal nothing)
//! 4. Each party computes [x*y] = [c] + d*[b] + e*[a] + d*e
//!
//! This works because:
//!   x*y = (a+d)*(b+e) = ab + ae + db + de = c + ae + db + de
//!
//! # Triple Generation Methods
//!
//! - `TrustedDealer`: Centralized generation (for testing/demos)
//! - `DistributedTripleGen`: Simulated distributed generation
//! - `OTTripleGenerator`: OT-based distributed generation (no trusted party)

pub mod async_pool;
pub mod dealer;
pub mod distributed;
pub mod ot;
pub mod pipeline;
pub mod pool;
pub mod triple;

pub use dealer::TrustedDealer;
pub use distributed::{
    DistributedTripleGen, NetworkDistributedDealer,
    DistributedTriplePool, DistributedPoolConfig,
    generate_and_fill_pool,
};
pub use ot::{OTTripleGenerator, OTSender, OTReceiver, OTReceiverKeys, OTExtension, CorrelatedOT};
pub use pool::BeaverPool;
pub use triple::{BeaverTriple, MatrixBeaverTriple, VectorBeaverTriple};
pub use pipeline::{
    BeaverPipeline, DemandPredictor, GenerationRequest, PipelineBuilder,
    PipelineConfig, PipelineStats, Priority, TripleSource,
};
pub use async_pool::{AsyncPoolConfig, AsyncTriplePool};
