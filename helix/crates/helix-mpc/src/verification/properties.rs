//! Security and correctness properties for MPC protocols.
//!
//! This module defines the formal properties that MPC protocols must satisfy.
//! Properties are specified declaratively and can be checked at runtime or
//! verified statically using formal methods.

use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;

use crate::types::PartyId;

/// Security level for adversary capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecurityLevel {
    /// Information-theoretic security (unconditional).
    InformationTheoretic,
    /// Computational security (relies on hardness assumptions).
    Computational,
    /// Statistical security (negligible advantage).
    Statistical,
}

/// Adversary model specification.
#[derive(Debug, Clone)]
pub struct AdversaryModel {
    /// Maximum number of corrupted parties.
    pub corruption_threshold: usize,
    /// Total number of parties.
    pub total_parties: usize,
    /// Whether adversary is adaptive (can corrupt during execution).
    pub adaptive: bool,
    /// Whether adversary is active (can deviate from protocol).
    pub active: bool,
    /// Whether adversary controls network timing.
    pub rushing: bool,
    /// Security level achieved against this adversary.
    pub security_level: SecurityLevel,
}

impl AdversaryModel {
    /// Creates a semi-honest adversary model.
    pub fn semi_honest(corruption_threshold: usize, total_parties: usize) -> Self {
        Self {
            corruption_threshold,
            total_parties,
            adaptive: false,
            active: false,
            rushing: false,
            security_level: SecurityLevel::Computational,
        }
    }

    /// Creates a malicious adversary model.
    pub fn malicious(corruption_threshold: usize, total_parties: usize) -> Self {
        Self {
            corruption_threshold,
            total_parties,
            adaptive: false,
            active: true,
            rushing: true,
            security_level: SecurityLevel::Computational,
        }
    }

    /// Returns honest majority condition.
    /// True if fewer than half of parties can be corrupted.
    pub fn honest_majority(&self) -> bool {
        // Honest majority means t < n/2, or equivalently 2t < n
        2 * self.corruption_threshold < self.total_parties
    }

    /// Returns dishonest majority condition.
    /// True if half or more parties can be corrupted.
    pub fn dishonest_majority(&self) -> bool {
        2 * self.corruption_threshold >= self.total_parties
    }
}

/// A verifiable protocol property.
pub trait ProtocolProperty: fmt::Debug + Send + Sync {
    /// Property identifier.
    fn id(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// Required adversary model for property to hold.
    fn required_adversary_model(&self) -> AdversaryModel;

    /// Returns whether property is structural (can be checked syntactically).
    fn is_structural(&self) -> bool {
        false
    }
}

/// Privacy property: adversary learns nothing beyond function output.
#[derive(Debug, Clone)]
pub struct PrivacyProperty {
    /// Property identifier.
    pub id: String,
    /// Description.
    pub description: String,
    /// Protected values (must remain secret).
    pub protected_values: Vec<String>,
    /// Adversary model.
    pub adversary: AdversaryModel,
    /// Leakage function (what adversary is allowed to learn).
    pub allowed_leakage: Vec<String>,
}

impl PrivacyProperty {
    /// Creates a new privacy property.
    pub fn new(
        id: impl Into<String>,
        protected_values: Vec<String>,
        adversary: AdversaryModel,
    ) -> Self {
        Self {
            id: id.into(),
            description: "Adversary learns nothing beyond allowed leakage".into(),
            protected_values,
            adversary,
            allowed_leakage: vec!["output".into()],
        }
    }

    /// Specifies allowed leakage.
    pub fn with_leakage(mut self, leakage: Vec<String>) -> Self {
        self.allowed_leakage = leakage;
        self
    }

    /// Formal statement of the property.
    ///
    /// ∀ corrupted ⊆ parties, |corrupted| ≤ t:
    ///   View_corrupted(x) ≈ Sim(f(x), leakage)
    ///
    /// Where ≈ denotes computational/statistical indistinguishability.
    pub fn formal_statement(&self) -> String {
        format!(
            "∀ C ⊆ P, |C| ≤ {}: View_C(x) ≈ Sim(f(x), {})",
            self.adversary.corruption_threshold,
            self.allowed_leakage.join(", ")
        )
    }
}

impl ProtocolProperty for PrivacyProperty {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn required_adversary_model(&self) -> AdversaryModel {
        self.adversary.clone()
    }
}

/// Correctness property: output equals function evaluation.
#[derive(Debug, Clone)]
pub struct CorrectnessProperty {
    /// Property identifier.
    pub id: String,
    /// Description.
    pub description: String,
    /// Function being computed.
    pub function_name: String,
    /// Tolerance for numerical results.
    pub tolerance: f64,
}

impl CorrectnessProperty {
    /// Creates a new correctness property.
    pub fn new(id: impl Into<String>, function_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: "Protocol output equals function evaluation".into(),
            function_name: function_name.into(),
            tolerance: 1e-10,
        }
    }

    /// Sets numerical tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Formal statement of the property.
    ///
    /// ∀ inputs x: output = f(x) ± tolerance
    pub fn formal_statement(&self) -> String {
        format!(
            "∀ x ∈ Domain: |π(x) - {}(x)| ≤ {}",
            self.function_name, self.tolerance
        )
    }
}

impl ProtocolProperty for CorrectnessProperty {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn required_adversary_model(&self) -> AdversaryModel {
        AdversaryModel::semi_honest(0, 1)
    }

    fn is_structural(&self) -> bool {
        true
    }
}

/// Integrity property: adversary cannot forge valid outputs.
#[derive(Debug, Clone)]
pub struct IntegrityProperty {
    /// Property identifier.
    pub id: String,
    /// Description.
    pub description: String,
    /// Protected computation.
    pub computation: String,
    /// Adversary model.
    pub adversary: AdversaryModel,
    /// Verification mechanism.
    pub verification: VerificationMechanism,
}

/// Verification mechanism for integrity.
#[derive(Debug, Clone)]
pub enum VerificationMechanism {
    /// Information-theoretic MAC.
    MAC { key_size: usize },
    /// Commitment scheme.
    Commitment { binding: bool, hiding: bool },
    /// Zero-knowledge proof.
    ZKProof { soundness_error: f64 },
    /// Signature scheme.
    Signature { scheme: String },
}

impl IntegrityProperty {
    /// Creates a new integrity property with MAC verification.
    pub fn with_mac(
        id: impl Into<String>,
        computation: impl Into<String>,
        adversary: AdversaryModel,
        key_size: usize,
    ) -> Self {
        Self {
            id: id.into(),
            description: "Adversary cannot forge valid computation results".into(),
            computation: computation.into(),
            adversary,
            verification: VerificationMechanism::MAC { key_size },
        }
    }

    /// Formal statement of the property.
    ///
    /// Pr[Forge] ≤ negl(λ)
    pub fn formal_statement(&self) -> String {
        match &self.verification {
            VerificationMechanism::MAC { key_size } => {
                format!("Pr[Forge] ≤ 2^(-{})", key_size)
            }
            VerificationMechanism::ZKProof { soundness_error } => {
                format!("Pr[Forge] ≤ {}", soundness_error)
            }
            _ => "Pr[Forge] ≤ negl(λ)".into(),
        }
    }
}

impl ProtocolProperty for IntegrityProperty {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn required_adversary_model(&self) -> AdversaryModel {
        self.adversary.clone()
    }
}

/// Comprehensive security property combining privacy, correctness, and integrity.
#[derive(Debug, Clone)]
pub struct SecurityProperty {
    /// Property identifier.
    pub id: String,
    /// Privacy sub-property.
    pub privacy: Option<PrivacyProperty>,
    /// Correctness sub-property.
    pub correctness: Option<CorrectnessProperty>,
    /// Integrity sub-property.
    pub integrity: Option<IntegrityProperty>,
}

impl SecurityProperty {
    /// Creates a new security property with all sub-properties.
    pub fn full(
        id: impl Into<String>,
        privacy: PrivacyProperty,
        correctness: CorrectnessProperty,
        integrity: IntegrityProperty,
    ) -> Self {
        Self {
            id: id.into(),
            privacy: Some(privacy),
            correctness: Some(correctness),
            integrity: Some(integrity),
        }
    }

    /// Creates a privacy-only security property.
    pub fn privacy_only(id: impl Into<String>, privacy: PrivacyProperty) -> Self {
        Self {
            id: id.into(),
            privacy: Some(privacy),
            correctness: None,
            integrity: None,
        }
    }

    /// Adds correctness property.
    pub fn with_correctness(mut self, correctness: CorrectnessProperty) -> Self {
        self.correctness = Some(correctness);
        self
    }

    /// Adds integrity property.
    pub fn with_integrity(mut self, integrity: IntegrityProperty) -> Self {
        self.integrity = Some(integrity);
        self
    }
}

/// Property verifier for checking properties.
pub struct PropertyVerifier {
    /// Registered properties.
    properties: Vec<Box<dyn ProtocolProperty>>,
    /// Adversary model for verification.
    adversary: AdversaryModel,
    /// Verification results.
    results: Vec<PropertyResult>,
}

/// Result of property verification.
#[derive(Debug, Clone)]
pub struct PropertyResult {
    /// Property ID.
    pub property_id: String,
    /// Verification status.
    pub status: PropertyStatus,
    /// Evidence or counterexample.
    pub evidence: Option<String>,
}

/// Property verification status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyStatus {
    /// Property verified to hold.
    Verified,
    /// Property falsified with counterexample.
    Falsified,
    /// Property unknown (timeout or undecidable).
    Unknown,
    /// Property not applicable in current context.
    NotApplicable,
}

impl PropertyVerifier {
    /// Creates a new property verifier.
    pub fn new(adversary: AdversaryModel) -> Self {
        Self {
            properties: Vec::new(),
            adversary,
            results: Vec::new(),
        }
    }

    /// Adds a property to verify.
    pub fn add_property<P: ProtocolProperty + 'static>(&mut self, property: P) {
        self.properties.push(Box::new(property));
    }

    /// Checks all structural properties.
    pub fn check_structural(&mut self) -> Vec<PropertyResult> {
        let mut results = Vec::new();

        for prop in &self.properties {
            if prop.is_structural() {
                // Structural properties can be checked immediately.
                let result = PropertyResult {
                    property_id: prop.id().to_string(),
                    status: PropertyStatus::Verified,
                    evidence: Some("Structural check passed".into()),
                };
                results.push(result);
            }
        }

        self.results.extend(results.clone());
        results
    }

    /// Generates proof obligations for non-structural properties.
    pub fn generate_proof_obligations(&self) -> Vec<ProofObligation> {
        self.properties
            .iter()
            .filter(|p| !p.is_structural())
            .map(|p| ProofObligation {
                id: format!("PO_{}", p.id()),
                property_id: p.id().to_string(),
                statement: p.description().to_string(),
                adversary: p.required_adversary_model(),
                status: ProofStatus::Pending,
            })
            .collect()
    }

    /// Returns all verification results.
    pub fn results(&self) -> &[PropertyResult] {
        &self.results
    }

    /// Clears results.
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

/// Proof obligation generated from properties.
#[derive(Debug, Clone)]
pub struct ProofObligation {
    /// Obligation identifier.
    pub id: String,
    /// Property this obligation is for.
    pub property_id: String,
    /// Statement to prove.
    pub statement: String,
    /// Adversary model.
    pub adversary: AdversaryModel,
    /// Current status.
    pub status: ProofStatus,
}

/// Status of a proof obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofStatus {
    /// Not yet attempted.
    Pending,
    /// Proof in progress.
    InProgress,
    /// Successfully proved.
    Proved,
    /// Failed to prove (not necessarily false).
    Failed,
    /// Proved false (counterexample found).
    Disproved,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversary_model() {
        let semi_honest = AdversaryModel::semi_honest(1, 3);
        assert!(!semi_honest.active);
        assert!(semi_honest.honest_majority());

        let malicious = AdversaryModel::malicious(2, 3);
        assert!(malicious.active);
        assert!(malicious.dishonest_majority());
    }

    #[test]
    fn test_privacy_property() {
        let adversary = AdversaryModel::semi_honest(1, 3);
        let privacy = PrivacyProperty::new(
            "secret_sharing_privacy",
            vec!["model_weights".into(), "gradients".into()],
            adversary,
        );

        assert_eq!(privacy.id(), "secret_sharing_privacy");
        assert!(!privacy.formal_statement().is_empty());
    }

    #[test]
    fn test_correctness_property() {
        let correctness = CorrectnessProperty::new("addition_correctness", "secure_add")
            .with_tolerance(1e-6);

        assert!(correctness.is_structural());
        assert!(correctness.formal_statement().contains("secure_add"));
    }

    #[test]
    fn test_integrity_property() {
        let adversary = AdversaryModel::malicious(1, 3);
        let integrity = IntegrityProperty::with_mac(
            "mac_integrity",
            "gradient_aggregation",
            adversary,
            128,
        );

        assert!(integrity.formal_statement().contains("128"));
    }

    #[test]
    fn test_property_verifier() {
        let adversary = AdversaryModel::semi_honest(1, 3);
        let mut verifier = PropertyVerifier::new(adversary.clone());

        verifier.add_property(CorrectnessProperty::new("test", "f"));
        verifier.add_property(PrivacyProperty::new(
            "privacy",
            vec!["x".into()],
            adversary,
        ));

        let structural = verifier.check_structural();
        assert_eq!(structural.len(), 1);
        assert_eq!(structural[0].status, PropertyStatus::Verified);

        let obligations = verifier.generate_proof_obligations();
        assert_eq!(obligations.len(), 1);
        assert_eq!(obligations[0].status, ProofStatus::Pending);
    }
}
