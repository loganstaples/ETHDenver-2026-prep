//! Protocol invariants for runtime checking.
//!
//! Invariants are properties that must hold at specific points during protocol
//! execution. They can be checked at runtime (debug builds) or verified
//! statically.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::types::PartyId;

/// A protocol invariant that can be checked.
pub trait ProtocolInvariant: fmt::Debug + Send + Sync {
    /// Invariant identifier.
    fn id(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// Check the invariant against current state.
    fn check(&self, context: &InvariantContext) -> Result<(), InvariantViolation>;

    /// Whether this invariant should be checked at every step.
    fn is_continuous(&self) -> bool {
        false
    }
}

/// Context for invariant checking.
#[derive(Debug, Default)]
pub struct InvariantContext {
    /// Current protocol phase.
    pub phase: String,
    /// Current step/round number.
    pub step: u64,
    /// Party-specific state.
    pub party_state: HashMap<PartyId, PartyInvariantState>,
    /// Global state values.
    pub global_state: HashMap<String, f64>,
    /// Boolean flags.
    pub flags: HashMap<String, bool>,
}

/// Per-party state for invariant checking.
#[derive(Debug, Default, Clone)]
pub struct PartyInvariantState {
    /// Share values (for sum checking).
    pub shares: Vec<f64>,
    /// Commitment values.
    pub commitments: Vec<[u8; 32]>,
    /// MAC values.
    pub macs: Vec<f64>,
    /// Party is active.
    pub active: bool,
    /// Messages sent count.
    pub messages_sent: u64,
    /// Messages received count.
    pub messages_received: u64,
}

impl InvariantContext {
    /// Creates a new context.
    pub fn new(phase: impl Into<String>, step: u64) -> Self {
        Self {
            phase: phase.into(),
            step,
            ..Default::default()
        }
    }

    /// Adds party state.
    pub fn add_party(&mut self, party: PartyId, state: PartyInvariantState) {
        self.party_state.insert(party, state);
    }

    /// Sets a global value.
    pub fn set_global(&mut self, key: impl Into<String>, value: f64) {
        self.global_state.insert(key.into(), value);
    }

    /// Sets a flag.
    pub fn set_flag(&mut self, key: impl Into<String>, value: bool) {
        self.flags.insert(key.into(), value);
    }

    /// Gets total number of parties.
    pub fn num_parties(&self) -> usize {
        self.party_state.len()
    }

    /// Gets active parties.
    pub fn active_parties(&self) -> Vec<&PartyId> {
        self.party_state
            .iter()
            .filter(|(_, s)| s.active)
            .map(|(p, _)| p)
            .collect()
    }
}

/// An invariant violation.
#[derive(Debug, Clone)]
pub struct InvariantViolation {
    /// Invariant that was violated.
    pub invariant_id: String,
    /// Description of violation.
    pub description: String,
    /// Expected value (if applicable).
    pub expected: Option<String>,
    /// Actual value (if applicable).
    pub actual: Option<String>,
    /// Context at time of violation.
    pub context: ViolationContext,
}

/// Context captured when violation occurs.
#[derive(Debug, Clone, Default)]
pub struct ViolationContext {
    pub phase: String,
    pub step: u64,
    pub party: Option<PartyId>,
}

impl fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invariant '{}' violated: {}", self.invariant_id, self.description)?;
        if let Some(ref exp) = self.expected {
            write!(f, " (expected: {}, got: {})", exp, self.actual.as_deref().unwrap_or("?"))?;
        }
        Ok(())
    }
}

/// Share sum invariant: shares must sum to the secret.
#[derive(Debug)]
pub struct ShareSumInvariant {
    /// Invariant ID.
    pub id: String,
    /// Tolerance for numerical comparison.
    pub tolerance: f64,
}

impl ShareSumInvariant {
    /// Creates a new share sum invariant.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tolerance: 1e-10,
        }
    }

    /// Sets tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl ProtocolInvariant for ShareSumInvariant {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        "Sum of all shares equals the secret value"
    }

    fn check(&self, context: &InvariantContext) -> Result<(), InvariantViolation> {
        // Get expected secret from global state.
        let expected = context.global_state.get("expected_secret").copied();

        if expected.is_none() {
            return Ok(()); // No secret to check against.
        }

        let expected = expected.unwrap();

        // Sum all party shares.
        let mut share_sums: HashMap<usize, f64> = HashMap::new();

        for state in context.party_state.values() {
            for (i, &share) in state.shares.iter().enumerate() {
                *share_sums.entry(i).or_insert(0.0) += share;
            }
        }

        // Check each position.
        for (i, sum) in share_sums {
            if (sum - expected).abs() > self.tolerance {
                return Err(InvariantViolation {
                    invariant_id: self.id.clone(),
                    description: format!("Share sum at position {} incorrect", i),
                    expected: Some(format!("{:.6}", expected)),
                    actual: Some(format!("{:.6}", sum)),
                    context: ViolationContext {
                        phase: context.phase.clone(),
                        step: context.step,
                        party: None,
                    },
                });
            }
        }

        Ok(())
    }

    fn is_continuous(&self) -> bool {
        true
    }
}

/// MAC consistency invariant: MACs must verify.
#[derive(Debug)]
pub struct MACConsistencyInvariant {
    /// Invariant ID.
    pub id: String,
    /// Global MAC key (sum of party key shares).
    pub alpha: f64,
    /// Tolerance.
    pub tolerance: f64,
}

impl MACConsistencyInvariant {
    /// Creates a new MAC consistency invariant.
    pub fn new(id: impl Into<String>, alpha: f64) -> Self {
        Self {
            id: id.into(),
            alpha,
            tolerance: 1e-6,
        }
    }
}

impl ProtocolInvariant for MACConsistencyInvariant {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        "MAC values are consistent with key and shares"
    }

    fn check(&self, context: &InvariantContext) -> Result<(), InvariantViolation> {
        for (party, state) in &context.party_state {
            if state.shares.len() != state.macs.len() {
                return Err(InvariantViolation {
                    invariant_id: self.id.clone(),
                    description: "Share/MAC count mismatch".into(),
                    expected: Some(format!("{} MACs", state.shares.len())),
                    actual: Some(format!("{} MACs", state.macs.len())),
                    context: ViolationContext {
                        phase: context.phase.clone(),
                        step: context.step,
                        party: Some(party.clone()),
                    },
                });
            }

            for (i, (&share, &mac)) in state.shares.iter().zip(state.macs.iter()).enumerate() {
                let expected_mac = self.alpha * share;
                if (mac - expected_mac).abs() > self.tolerance {
                    return Err(InvariantViolation {
                        invariant_id: self.id.clone(),
                        description: format!("MAC verification failed at position {}", i),
                        expected: Some(format!("{:.6}", expected_mac)),
                        actual: Some(format!("{:.6}", mac)),
                        context: ViolationContext {
                            phase: context.phase.clone(),
                            step: context.step,
                            party: Some(party.clone()),
                        },
                    });
                }
            }
        }

        Ok(())
    }
}

/// Beaver triple invariant: a * b = c.
#[derive(Debug)]
pub struct BeaverTripleInvariant {
    /// Invariant ID.
    pub id: String,
    /// Tolerance.
    pub tolerance: f64,
}

impl BeaverTripleInvariant {
    /// Creates a new Beaver triple invariant.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tolerance: 1e-10,
        }
    }
}

impl ProtocolInvariant for BeaverTripleInvariant {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        "Beaver triples satisfy a * b = c"
    }

    fn check(&self, context: &InvariantContext) -> Result<(), InvariantViolation> {
        // Check triples stored in global state.
        let triple_count = context.global_state.get("triple_count").copied().unwrap_or(0.0) as usize;

        for i in 0..triple_count {
            let a = context.global_state.get(&format!("triple_{}_a", i));
            let b = context.global_state.get(&format!("triple_{}_b", i));
            let c = context.global_state.get(&format!("triple_{}_c", i));

            if let (Some(&a), Some(&b), Some(&c)) = (a, b, c) {
                let expected_c = a * b;
                if (c - expected_c).abs() > self.tolerance {
                    return Err(InvariantViolation {
                        invariant_id: self.id.clone(),
                        description: format!("Triple {} fails a*b=c", i),
                        expected: Some(format!("{:.6}", expected_c)),
                        actual: Some(format!("{:.6}", c)),
                        context: ViolationContext {
                            phase: context.phase.clone(),
                            step: context.step,
                            party: None,
                        },
                    });
                }
            }
        }

        Ok(())
    }
}

/// Message balance invariant: sent messages equal received messages.
#[derive(Debug)]
pub struct MessageBalanceInvariant {
    /// Invariant ID.
    pub id: String,
}

impl MessageBalanceInvariant {
    /// Creates a new message balance invariant.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl ProtocolInvariant for MessageBalanceInvariant {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        "Total sent messages equals total received messages"
    }

    fn check(&self, context: &InvariantContext) -> Result<(), InvariantViolation> {
        let total_sent: u64 = context.party_state.values().map(|s| s.messages_sent).sum();
        let total_received: u64 = context.party_state.values().map(|s| s.messages_received).sum();

        if total_sent != total_received {
            return Err(InvariantViolation {
                invariant_id: self.id.clone(),
                description: "Message count mismatch".into(),
                expected: Some(format!("{} received", total_sent)),
                actual: Some(format!("{} received", total_received)),
                context: ViolationContext {
                    phase: context.phase.clone(),
                    step: context.step,
                    party: None,
                },
            });
        }

        Ok(())
    }
}

/// Runtime invariant checker.
pub struct InvariantChecker {
    /// Registered invariants.
    invariants: Vec<Box<dyn ProtocolInvariant>>,
    /// Continuous invariants (checked every step).
    continuous: Vec<usize>,
    /// Violation history.
    violations: RwLock<Vec<InvariantViolation>>,
    /// Check count.
    checks_performed: AtomicU64,
}

impl InvariantChecker {
    /// Creates a new invariant checker.
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
            continuous: Vec::new(),
            violations: RwLock::new(Vec::new()),
            checks_performed: AtomicU64::new(0),
        }
    }

    /// Adds an invariant.
    pub fn add<I: ProtocolInvariant + 'static>(&mut self, invariant: I) {
        let idx = self.invariants.len();
        let continuous = invariant.is_continuous();
        self.invariants.push(Box::new(invariant));
        if continuous {
            self.continuous.push(idx);
        }
    }

    /// Checks all invariants.
    pub fn check_all(&self, context: &InvariantContext) -> Vec<InvariantViolation> {
        let mut violations = Vec::new();

        for inv in &self.invariants {
            self.checks_performed.fetch_add(1, Ordering::SeqCst);
            if let Err(v) = inv.check(context) {
                violations.push(v.clone());
                self.violations.write().push(v);
            }
        }

        violations
    }

    /// Checks only continuous invariants.
    pub fn check_continuous(&self, context: &InvariantContext) -> Vec<InvariantViolation> {
        let mut violations = Vec::new();

        for &idx in &self.continuous {
            if let Some(inv) = self.invariants.get(idx) {
                self.checks_performed.fetch_add(1, Ordering::SeqCst);
                if let Err(v) = inv.check(context) {
                    violations.push(v.clone());
                    self.violations.write().push(v);
                }
            }
        }

        violations
    }

    /// Checks a specific invariant by ID.
    pub fn check_by_id(&self, id: &str, context: &InvariantContext) -> Option<Result<(), InvariantViolation>> {
        for inv in &self.invariants {
            if inv.id() == id {
                self.checks_performed.fetch_add(1, Ordering::SeqCst);
                return Some(inv.check(context));
            }
        }
        None
    }

    /// Returns all violations.
    pub fn violations(&self) -> Vec<InvariantViolation> {
        self.violations.read().clone()
    }

    /// Returns check count.
    pub fn checks_performed(&self) -> u64 {
        self.checks_performed.load(Ordering::SeqCst)
    }

    /// Clears violation history.
    pub fn clear_violations(&self) {
        self.violations.write().clear();
    }
}

impl Default for InvariantChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime invariant checker with automatic context building.
pub struct RuntimeInvariantChecker {
    /// Inner checker.
    checker: InvariantChecker,
    /// Current phase.
    phase: RwLock<String>,
    /// Current step.
    step: AtomicU64,
    /// Enable checking (can be disabled in release builds).
    enabled: bool,
}

impl RuntimeInvariantChecker {
    /// Creates a new runtime checker.
    pub fn new(enabled: bool) -> Self {
        Self {
            checker: InvariantChecker::new(),
            phase: RwLock::new("init".into()),
            step: AtomicU64::new(0),
            enabled,
        }
    }

    /// Adds an invariant.
    pub fn add<I: ProtocolInvariant + 'static>(&mut self, invariant: I) {
        self.checker.add(invariant);
    }

    /// Sets current phase.
    pub fn set_phase(&self, phase: impl Into<String>) {
        *self.phase.write() = phase.into();
    }

    /// Advances step.
    pub fn advance_step(&self) -> u64 {
        self.step.fetch_add(1, Ordering::SeqCst)
    }

    /// Checks all invariants with auto-built context.
    pub fn check(&self, context_builder: impl FnOnce(&mut InvariantContext)) -> Vec<InvariantViolation> {
        if !self.enabled {
            return Vec::new();
        }

        let mut context = InvariantContext::new(
            self.phase.read().clone(),
            self.step.load(Ordering::SeqCst),
        );
        context_builder(&mut context);

        self.checker.check_all(&context)
    }

    /// Checks continuous invariants only.
    pub fn check_continuous(&self, context_builder: impl FnOnce(&mut InvariantContext)) -> Vec<InvariantViolation> {
        if !self.enabled {
            return Vec::new();
        }

        let mut context = InvariantContext::new(
            self.phase.read().clone(),
            self.step.load(Ordering::SeqCst),
        );
        context_builder(&mut context);

        self.checker.check_continuous(&context)
    }

    /// Returns violations.
    pub fn violations(&self) -> Vec<InvariantViolation> {
        self.checker.violations()
    }

    /// Returns inner checker.
    pub fn inner(&self) -> &InvariantChecker {
        &self.checker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_sum_invariant() {
        let invariant = ShareSumInvariant::new("test_share_sum");

        let mut context = InvariantContext::new("computation", 1);
        context.set_global("expected_secret", 10.0);

        // Add three parties whose shares sum to 10.
        context.add_party(
            PartyId::from_index(0),
            PartyInvariantState {
                shares: vec![3.0],
                active: true,
                ..Default::default()
            },
        );
        context.add_party(
            PartyId::from_index(1),
            PartyInvariantState {
                shares: vec![4.0],
                active: true,
                ..Default::default()
            },
        );
        context.add_party(
            PartyId::from_index(2),
            PartyInvariantState {
                shares: vec![3.0],
                active: true,
                ..Default::default()
            },
        );

        assert!(invariant.check(&context).is_ok());

        // Now break it.
        context.party_state.get_mut(&PartyId::from_index(2)).unwrap().shares = vec![5.0];
        assert!(invariant.check(&context).is_err());
    }

    #[test]
    fn test_mac_consistency_invariant() {
        let invariant = MACConsistencyInvariant::new("test_mac", 5.0);

        let mut context = InvariantContext::new("computation", 1);

        // Valid MACs.
        context.add_party(
            PartyId::from_index(0),
            PartyInvariantState {
                shares: vec![2.0, 3.0],
                macs: vec![10.0, 15.0], // 5.0 * share
                active: true,
                ..Default::default()
            },
        );

        assert!(invariant.check(&context).is_ok());

        // Invalid MAC.
        context.party_state.get_mut(&PartyId::from_index(0)).unwrap().macs[1] = 14.0;
        assert!(invariant.check(&context).is_err());
    }

    #[test]
    fn test_beaver_triple_invariant() {
        let invariant = BeaverTripleInvariant::new("test_beaver");

        let mut context = InvariantContext::new("preprocessing", 0);
        context.set_global("triple_count", 2.0);
        context.set_global("triple_0_a", 3.0);
        context.set_global("triple_0_b", 4.0);
        context.set_global("triple_0_c", 12.0);
        context.set_global("triple_1_a", 5.0);
        context.set_global("triple_1_b", 6.0);
        context.set_global("triple_1_c", 30.0);

        assert!(invariant.check(&context).is_ok());

        // Break triple 1.
        context.set_global("triple_1_c", 31.0);
        assert!(invariant.check(&context).is_err());
    }

    #[test]
    fn test_invariant_checker() {
        let mut checker = InvariantChecker::new();
        checker.add(ShareSumInvariant::new("share_sum"));
        checker.add(MessageBalanceInvariant::new("msg_balance"));

        let mut context = InvariantContext::new("test", 0);
        context.add_party(
            PartyId::from_index(0),
            PartyInvariantState {
                messages_sent: 5,
                messages_received: 5,
                ..Default::default()
            },
        );

        let violations = checker.check_all(&context);
        assert!(violations.is_empty());
        assert!(checker.checks_performed() >= 2);
    }

    #[test]
    fn test_runtime_checker() {
        let mut checker = RuntimeInvariantChecker::new(true);
        checker.add(ShareSumInvariant::new("share_sum").with_tolerance(0.1));

        checker.set_phase("computation");
        checker.advance_step();

        let violations = checker.check(|ctx| {
            ctx.set_global("expected_secret", 10.0);
            ctx.add_party(
                PartyId::from_index(0),
                PartyInvariantState {
                    shares: vec![5.0],
                    active: true,
                    ..Default::default()
                },
            );
            ctx.add_party(
                PartyId::from_index(1),
                PartyInvariantState {
                    shares: vec![5.0],
                    active: true,
                    ..Default::default()
                },
            );
        });

        assert!(violations.is_empty());
    }
}
