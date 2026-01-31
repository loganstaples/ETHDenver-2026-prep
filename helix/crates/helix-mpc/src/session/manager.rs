//! MPC session lifecycle management.

use std::collections::HashMap;
use std::sync::Arc;

use crate::beaver::dealer::TrustedDealer;
use crate::beaver::pool::BeaverPool;
use crate::error::{MPCError, MPCResult};
use crate::protocols::reshare::Resharing;
use crate::sharing::model::ModelShare;
use crate::sharing::AdditiveSharing;
use crate::types::{MPCConfig, MPCPhase, PartyId, PartyRole};
use super::channel::{LocalChannel, MPCChannel};

/// An MPC session coordinating multi-party computation.
#[derive(Debug)]
pub struct MPCSession {
    /// Session configuration.
    pub config: MPCConfig,
    /// Current phase.
    pub phase: MPCPhase,
    /// Registered participants.
    pub participants: HashMap<PartyId, ParticipantInfo>,
    /// Beaver triple pools (one per party).
    pub pools: Vec<BeaverPool>,
    /// Model shares (one per party, set during input sharing).
    pub model_shares: Vec<Option<ModelShare>>,
    /// Current training step (for re-sharing decisions).
    pub current_step: u64,
    /// Session ID.
    pub session_id: String,
    /// Communication channel.
    channel: Arc<LocalChannel>,
}

/// Information about a registered participant.
#[derive(Debug, Clone)]
pub struct ParticipantInfo {
    pub party_id: PartyId,
    pub index: usize,
    pub role: PartyRole,
    pub stake: u64,
    pub is_active: bool,
}

impl MPCSession {
    /// Creates a new MPC session.
    pub fn new(config: MPCConfig, session_id: impl Into<String>) -> MPCResult<Self> {
        config.validate().map_err(MPCError::InvalidConfig)?;

        let n = config.num_parties;
        let parties: Vec<PartyId> = (0..n).map(PartyId::from_index).collect();
        let channel = Arc::new(LocalChannel::new(&parties));

        let pools: Vec<BeaverPool> = (0..n)
            .map(|i| BeaverPool::new(i, n, config.beaver_batch_size))
            .collect();

        Ok(Self {
            config,
            phase: MPCPhase::Preprocessing,
            participants: HashMap::new(),
            pools,
            model_shares: vec![None; n],
            current_step: 0,
            session_id: session_id.into(),
            channel,
        })
    }

    /// Registers a participant in the session.
    pub fn register_participant(
        &mut self,
        party_id: PartyId,
        index: usize,
        role: PartyRole,
        stake: u64,
    ) -> MPCResult<()> {
        if index >= self.config.num_parties {
            return Err(MPCError::IndexOutOfRange {
                index,
                num_parties: self.config.num_parties,
            });
        }

        if self.participants.contains_key(&party_id) {
            return Err(MPCError::DuplicateParty(party_id));
        }

        self.participants.insert(
            party_id.clone(),
            ParticipantInfo {
                party_id,
                index,
                role,
                stake,
                is_active: true,
            },
        );

        Ok(())
    }

    /// Returns the number of active participants.
    pub fn active_count(&self) -> usize {
        self.participants.values().filter(|p| p.is_active).count()
    }

    /// Runs the preprocessing phase: generates Beaver triples.
    pub fn preprocess(
        &mut self,
        hidden_dim: usize,
        num_layers: usize,
    ) -> MPCResult<()> {
        if self.phase != MPCPhase::Preprocessing {
            return Err(MPCError::SessionError(format!(
                "Cannot preprocess in phase {:?}",
                self.phase,
            )));
        }

        let mut dealer = TrustedDealer::with_seed(0xBE11C);

        for pool in &mut self.pools {
            pool.fill_for_training_step(&mut dealer, hidden_dim, num_layers);
        }

        self.phase = MPCPhase::InputSharing;
        Ok(())
    }

    /// Distributes model weight shares to participants.
    pub fn distribute_model_shares(
        &mut self,
        shares: Vec<ModelShare>,
    ) -> MPCResult<()> {
        if self.phase != MPCPhase::InputSharing {
            return Err(MPCError::SessionError(format!(
                "Cannot distribute shares in phase {:?}",
                self.phase,
            )));
        }

        if shares.len() != self.config.num_parties {
            return Err(MPCError::ShareCountMismatch {
                expected: self.config.num_parties,
                got: shares.len(),
            });
        }

        for (i, share) in shares.into_iter().enumerate() {
            self.model_shares[i] = Some(share);
        }

        self.phase = MPCPhase::Computation;
        Ok(())
    }

    /// Gets a mutable reference to a party's Beaver pool.
    pub fn pool_mut(&mut self, party_index: usize) -> MPCResult<&mut BeaverPool> {
        self.pools
            .get_mut(party_index)
            .ok_or(MPCError::IndexOutOfRange {
                index: party_index,
                num_parties: self.config.num_parties,
            })
    }

    /// Gets all pools as mutable (for protocols that need all parties' pools).
    pub fn all_pools_mut(&mut self) -> &mut [BeaverPool] {
        &mut self.pools
    }

    /// Gets a party's model share.
    pub fn model_share(&self, party_index: usize) -> MPCResult<&ModelShare> {
        self.model_shares
            .get(party_index)
            .and_then(|s| s.as_ref())
            .ok_or(MPCError::SessionError(format!(
                "No model share for party {}",
                party_index,
            )))
    }

    /// Gets a mutable reference to a party's model share.
    pub fn model_share_mut(&mut self, party_index: usize) -> MPCResult<&mut ModelShare> {
        self.model_shares
            .get_mut(party_index)
            .and_then(|s| s.as_mut())
            .ok_or(MPCError::SessionError(format!(
                "No model share for party {}",
                party_index,
            )))
    }

    /// Advances the training step counter and checks if re-sharing is needed.
    pub fn advance_step(&mut self) -> MPCResult<bool> {
        self.current_step += 1;

        let should_reshare =
            Resharing::should_reshare(self.current_step, self.config.reshare_interval);

        if should_reshare {
            self.reshare_all()?;
        }

        Ok(should_reshare)
    }

    /// Re-shares all model weights across parties.
    fn reshare_all(&mut self) -> MPCResult<()> {
        let shares: Vec<ModelShare> = self
            .model_shares
            .iter()
            .filter_map(|s| s.clone())
            .collect();

        if shares.len() != self.config.num_parties {
            return Err(MPCError::InsufficientParties {
                required: self.config.num_parties,
                available: shares.len(),
            });
        }

        let new_shares = Resharing::reshare_model(&shares, self.current_step);

        for (i, share) in new_shares.into_iter().enumerate() {
            self.model_shares[i] = Some(share);
        }

        Ok(())
    }

    /// Replenishes Beaver triples for the next training step.
    pub fn replenish_triples(
        &mut self,
        hidden_dim: usize,
        num_layers: usize,
    ) -> MPCResult<()> {
        let mut dealer = TrustedDealer::with_seed(
            (0xBE11C_u64).wrapping_add(self.current_step),
        );

        for pool in &mut self.pools {
            pool.fill_for_training_step(&mut dealer, hidden_dim, num_layers);
        }

        Ok(())
    }

    /// Transitions to output reconstruction phase.
    pub fn begin_reconstruction(&mut self) -> MPCResult<()> {
        if self.phase != MPCPhase::Computation {
            return Err(MPCError::SessionError(format!(
                "Cannot begin reconstruction in phase {:?}",
                self.phase,
            )));
        }
        self.phase = MPCPhase::OutputReconstruction;
        Ok(())
    }

    /// Marks the session as complete.
    pub fn complete(&mut self) {
        self.phase = MPCPhase::Complete;
    }

    /// Marks the session as failed.
    pub fn fail(&mut self) {
        self.phase = MPCPhase::Failed;
    }

    /// Returns session statistics.
    pub fn stats(&self) -> SessionStats {
        SessionStats {
            session_id: self.session_id.clone(),
            phase: self.phase,
            num_parties: self.config.num_parties,
            active_parties: self.active_count(),
            current_step: self.current_step,
            total_scalar_triples: self.pools.iter().map(|p| p.scalar_available()).sum(),
        }
    }
}

/// Session statistics.
#[derive(Debug, Clone)]
pub struct SessionStats {
    pub session_id: String,
    pub phase: MPCPhase,
    pub num_parties: usize,
    pub active_parties: usize,
    pub current_step: u64,
    pub total_scalar_triples: usize,
}

impl std::fmt::Display for SessionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Session '{}': phase={}, parties={}/{}, step={}, triples={}",
            self.session_id,
            self.phase,
            self.active_parties,
            self.num_parties,
            self.current_step,
            self.total_scalar_triples,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let config = MPCConfig::three_party();
        let session = MPCSession::new(config, "test-session").unwrap();

        assert_eq!(session.phase, MPCPhase::Preprocessing);
        assert_eq!(session.pools.len(), 3);
        assert_eq!(session.session_id, "test-session");
    }

    #[test]
    fn test_participant_registration() {
        let config = MPCConfig::three_party();
        let mut session = MPCSession::new(config, "test").unwrap();

        session
            .register_participant(PartyId::from_index(0), 0, PartyRole::Dealer, 1000)
            .unwrap();
        session
            .register_participant(PartyId::from_index(1), 1, PartyRole::Worker, 500)
            .unwrap();

        assert_eq!(session.active_count(), 2);

        // Duplicate registration should fail.
        let result = session.register_participant(
            PartyId::from_index(0),
            0,
            PartyRole::Worker,
            100,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_session_lifecycle() {
        let config = MPCConfig::three_party();
        let mut session = MPCSession::new(config, "test").unwrap();

        // Register parties.
        for i in 0..3 {
            session
                .register_participant(
                    PartyId::from_index(i),
                    i,
                    if i == 0 { PartyRole::Dealer } else { PartyRole::Worker },
                    1000,
                )
                .unwrap();
        }

        // Preprocess.
        session.preprocess(8, 1).unwrap();
        assert_eq!(session.phase, MPCPhase::InputSharing);

        // Distribute shares (empty shares for testing).
        let shares: Vec<ModelShare> = (0..3)
            .map(|i| ModelShare {
                party: PartyId::from_index(i),
                index: i,
                embeddings: None,
                layers: vec![],
                lm_head: None,
                extra_weights: HashMap::new(),
                model_name: "test".into(),
                num_layers: 0,
            })
            .collect();

        session.distribute_model_shares(shares).unwrap();
        assert_eq!(session.phase, MPCPhase::Computation);

        // Advance steps.
        session.advance_step().unwrap();
        assert_eq!(session.current_step, 1);

        let stats = session.stats();
        assert_eq!(stats.active_parties, 3);
    }

    #[test]
    fn test_phase_enforcement() {
        let config = MPCConfig::three_party();
        let mut session = MPCSession::new(config, "test").unwrap();

        // Can't distribute shares before preprocessing.
        let result = session.distribute_model_shares(vec![]);
        assert!(result.is_err());
    }
}
