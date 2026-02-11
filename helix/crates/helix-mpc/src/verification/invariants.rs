//! Protocol invariants for runtime checking.
//!
//! Invariants are properties that must hold at specific points during protocol
//! execution. They can be checked at runtime (debug builds) or verified
//! statically.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::field::Fr;
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
    /// Share values (for sum checking) — field elements.
    pub shares: Vec<Fr>,
    /// Commitment values.
    pub commitments: Vec<[u8; 32]>,
    /// MAC values — field elements.
    pub macs: Vec<Fr>,
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
    /// Expected secret (Fr field element).
    pub expected_secret: Option<Fr>,
}

impl ShareSumInvariant {
    /// Creates a new share sum invariant.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            expected_secret: None,
        }
    }

    /// Sets the expected secret value for checking.
    pub fn with_expected(mut self, secret: Fr) -> Self {
        self.expected_secret = Some(secret);
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
        let expected = match &self.expected_secret {
            Some(v) => v,
            None => return Ok(()), // No secret to check against.
        };

        // Sum all party shares at each position.
        let mut share_sums: HashMap<usize, Fr> = HashMap::new();

        for state in context.party_state.values() {
            for (i, share) in state.shares.iter().enumerate() {
                let entry = share_sums.entry(i).or_insert(Fr::ZERO);
                *entry = Fr::add(entry, share);
            }
        }

        // Check each position using exact Fr comparison.
        for (i, sum) in share_sums {
            if !sum.ct_eq(expected).to_bool() {
                return Err(InvariantViolation {
                    invariant_id: self.id.clone(),
                    description: format!("Share sum at position {} incorrect", i),
                    expected: Some(format!("{}", expected.to_f64())),
                    actual: Some(format!("{}", sum.to_f64())),
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
/// Uses exact Fr field arithmetic: mac_i = alpha * share_i.
#[derive(Debug)]
pub struct MACConsistencyInvariant {
    /// Invariant ID.
    pub id: String,
    /// Global MAC key (sum of party key shares) as Fr.
    pub alpha: Fr,
}

impl MACConsistencyInvariant {
    /// Creates a new MAC consistency invariant.
    pub fn new(id: impl Into<String>, alpha: Fr) -> Self {
        Self {
            id: id.into(),
            alpha,
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

            for (i, (share, mac)) in state.shares.iter().zip(state.macs.iter()).enumerate() {
                let expected_mac = Fr::mul(&self.alpha, share);
                if !mac.ct_eq(&expected_mac).to_bool() {
                    return Err(InvariantViolation {
                        invariant_id: self.id.clone(),
                        description: format!("MAC verification failed at position {}", i),
                        expected: Some(format!("{}", expected_mac.to_f64())),
                        actual: Some(format!("{}", mac.to_f64())),
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

/// Beaver triple invariant: a * b = c (exact Fr check).
#[derive(Debug)]
pub struct BeaverTripleInvariant {
    /// Invariant ID.
    pub id: String,
    /// Triples to check: (a, b, c) where a * b must equal c.
    pub triples: Vec<(Fr, Fr, Fr)>,
    /// Whether to use fixed_mul (for fixed-point encoded values).
    pub use_fixed_mul: bool,
}

impl BeaverTripleInvariant {
    /// Creates a new Beaver triple invariant.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            triples: Vec::new(),
            use_fixed_mul: false,
        }
    }

    /// Adds a triple to check.
    pub fn with_triple(mut self, a: Fr, b: Fr, c: Fr) -> Self {
        self.triples.push((a, b, c));
        self
    }

    /// Use fixed-point multiplication (for TrustedDealer triples).
    pub fn with_fixed_mul(mut self) -> Self {
        self.use_fixed_mul = true;
        self
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
        for (i, (a, b, c)) in self.triples.iter().enumerate() {
            let expected_c = if self.use_fixed_mul {
                a.fixed_mul(b)
            } else {
                Fr::mul(a, b)
            };

            if !c.ct_eq(&expected_c).to_bool() {
                return Err(InvariantViolation {
                    invariant_id: self.id.clone(),
                    description: format!("Triple {} fails a*b=c", i),
                    expected: Some(format!("{}", expected_c.to_f64())),
                    actual: Some(format!("{}", c.to_f64())),
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
        let secret = Fr::from_f64(10.0);
        let invariant = ShareSumInvariant::new("test_share_sum")
            .with_expected(secret.clone());

        let mut context = InvariantContext::new("computation", 1);

        // Add three parties whose shares sum to 10.
        let s0 = Fr::from_f64(3.0);
        let s1 = Fr::from_f64(4.0);
        let s2 = Fr::sub(&secret, &Fr::add(&s0, &s1));

        context.add_party(
            PartyId::from_index(0),
            PartyInvariantState {
                shares: vec![s0],
                active: true,
                ..Default::default()
            },
        );
        context.add_party(
            PartyId::from_index(1),
            PartyInvariantState {
                shares: vec![s1],
                active: true,
                ..Default::default()
            },
        );
        context.add_party(
            PartyId::from_index(2),
            PartyInvariantState {
                shares: vec![s2],
                active: true,
                ..Default::default()
            },
        );

        assert!(invariant.check(&context).is_ok());

        // Now break it.
        context.party_state.get_mut(&PartyId::from_index(2)).unwrap().shares = vec![Fr::from_f64(5.0)];
        assert!(invariant.check(&context).is_err());
    }

    #[test]
    fn test_mac_consistency_invariant() {
        let alpha = Fr::from_f64(5.0);
        let invariant = MACConsistencyInvariant::new("test_mac", alpha.clone());

        let mut context = InvariantContext::new("computation", 1);

        let s0 = Fr::from_f64(2.0);
        let s1 = Fr::from_f64(3.0);

        // Valid MACs: alpha * share.
        context.add_party(
            PartyId::from_index(0),
            PartyInvariantState {
                shares: vec![s0.clone(), s1.clone()],
                macs: vec![Fr::mul(&alpha, &s0), Fr::mul(&alpha, &s1)],
                active: true,
                ..Default::default()
            },
        );

        assert!(invariant.check(&context).is_ok());

        // Invalid MAC.
        context.party_state.get_mut(&PartyId::from_index(0)).unwrap().macs[1] = Fr::from_f64(14.0);
        assert!(invariant.check(&context).is_err());
    }

    #[test]
    fn test_beaver_triple_invariant() {
        let a1 = Fr::from_f64(3.0);
        let b1 = Fr::from_f64(4.0);
        let c1 = Fr::mul(&a1, &b1);
        let a2 = Fr::from_f64(5.0);
        let b2 = Fr::from_f64(6.0);
        let c2 = Fr::mul(&a2, &b2);

        let invariant = BeaverTripleInvariant::new("test_beaver")
            .with_triple(a1, b1, c1)
            .with_triple(a2.clone(), b2.clone(), c2);

        let context = InvariantContext::new("preprocessing", 0);
        assert!(invariant.check(&context).is_ok());

        // Break triple 1.
        let bad = BeaverTripleInvariant::new("test_beaver_bad")
            .with_triple(a2, b2, Fr::from_f64(31.0));
        assert!(bad.check(&context).is_err());
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
        let secret = Fr::from_f64(10.0);
        let mut checker = RuntimeInvariantChecker::new(true);
        checker.add(ShareSumInvariant::new("share_sum").with_expected(secret.clone()));

        checker.set_phase("computation");
        checker.advance_step();

        let s0 = Fr::from_f64(5.0);
        let s1 = Fr::sub(&secret, &s0);

        let violations = checker.check(|ctx| {
            ctx.add_party(
                PartyId::from_index(0),
                PartyInvariantState {
                    shares: vec![s0.clone()],
                    active: true,
                    ..Default::default()
                },
            );
            ctx.add_party(
                PartyId::from_index(1),
                PartyInvariantState {
                    shares: vec![s1.clone()],
                    active: true,
                    ..Default::default()
                },
            );
        });

        assert!(violations.is_empty());
    }
}
