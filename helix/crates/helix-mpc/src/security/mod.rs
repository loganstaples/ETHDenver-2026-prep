//! Security module: share commitments, verification, and audit trails.

pub mod commitment;
pub mod verification;
pub mod audit;

pub use commitment::ShareCommitment;
pub use verification::ShareVerifier;
pub use audit::AuditLog;
