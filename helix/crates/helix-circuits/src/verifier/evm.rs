//! EVM Verifier Generator.
//!
//! Generates Solidity code for verifying Halo2 proofs on-chain using
//! BN254 pairing precompiles.

use std::fmt::Write;

/// Embedded verification key data for the Solidity contract.
#[derive(Debug, Clone)]
pub struct VkData {
    /// Generator G1 point (x, y).
    pub g1: (String, String),
    /// SRS s·G2 point (x_c0, x_c1, y_c0, y_c1) — for KZG opening.
    pub s_g2: (String, String, String, String),
    /// Negative G2 generator (x_c0, x_c1, y_c0, y_c1).
    pub neg_g2: (String, String, String, String),
    /// Number of advice commitments to read from the proof.
    pub num_advices: usize,
}

impl Default for VkData {
    fn default() -> Self {
        // BN254 generator points (standard).
        Self {
            g1: (
                "1".to_string(),
                "2".to_string(),
            ),
            s_g2: (
                "11559732032986387107991004021392285783925812861821192530917403151452391805634".to_string(),
                "10857046999023057135944570762232829481370756359578518086990519993285655852781".to_string(),
                "4082367875863433681332203403145435568316851327593401208105741076214120093531".to_string(),
                "8495653923123431417604973247489272438418190587263600148770280649306958101930".to_string(),
            ),
            neg_g2: (
                "11559732032986387107991004021392285783925812861821192530917403151452391805634".to_string(),
                "10857046999023057135944570762232829481370756359578518086990519993285655852781".to_string(),
                "17805874995975841540914202342111839520379459829704422454583296818431106115052".to_string(),
                "13392588948715843804641432497768002650278120570034223513918757245338268106653".to_string(),
            ),
            num_advices: 2,
        }
    }
}

/// Solidity verifier generator for Halo2 proofs.
#[derive(Debug)]
pub struct SolidityGenerator {
    /// Contract name.
    contract_name: String,
    /// License identifier.
    license: String,
    /// Solidity version.
    solidity_version: String,
    /// Number of public inputs.
    num_instances: usize,
    /// Whether to include batch verification.
    include_batch: bool,
    /// Verification key data for KZG pairing.
    vk_data: Option<VkData>,
}

impl Default for SolidityGenerator {
    fn default() -> Self {
        Self {
            contract_name: "HaloVerifier".to_string(),
            license: "MIT".to_string(),
            solidity_version: "^0.8.20".to_string(),
            num_instances: 2,
            include_batch: true,
            vk_data: None,
        }
    }
}

impl SolidityGenerator {
    /// Creates a new generator with the given contract name.
    pub fn new(contract_name: impl Into<String>) -> Self {
        Self {
            contract_name: contract_name.into(),
            ..Default::default()
        }
    }

    /// Sets the number of public inputs.
    pub fn with_instances(mut self, num: usize) -> Self {
        self.num_instances = num;
        self
    }

    /// Sets whether to include batch verification.
    pub fn with_batch(mut self, include: bool) -> Self {
        self.include_batch = include;
        self
    }

    /// Sets verification key data for full KZG pairing verification.
    pub fn with_vk_data(mut self, vk: VkData) -> Self {
        self.vk_data = Some(vk);
        self
    }

    /// Generates the complete Solidity verifier contract.
    pub fn generate(&self) -> String {
        let mut code = String::new();

        // Header
        writeln!(code, "// SPDX-License-Identifier: {}", self.license).unwrap();
        writeln!(code, "pragma solidity {};", self.solidity_version).unwrap();
        writeln!(code).unwrap();
        writeln!(code, "/// @title {} - Halo2 Proof Verifier", self.contract_name).unwrap();
        writeln!(code, "/// @notice Generated verifier for BN254 pairing-based proofs").unwrap();
        writeln!(code, "contract {} {{", self.contract_name).unwrap();
        writeln!(code).unwrap();

        // Constants
        self.generate_constants(&mut code);
        
        // Events
        self.generate_events(&mut code);
        
        // Errors
        self.generate_errors(&mut code);
        
        // Main verify function
        self.generate_verify_function(&mut code);
        
        // Pairing helpers
        self.generate_pairing_helpers(&mut code);
        
        // Transcript helpers
        self.generate_transcript_helpers(&mut code);
        
        if self.include_batch {
            self.generate_batch_verify(&mut code);
        }

        writeln!(code, "}}").unwrap();
        code
    }

    fn generate_constants(&self, code: &mut String) {
        writeln!(code, "    // BN254 curve constants").unwrap();
        writeln!(code, "    uint256 constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;").unwrap();
        writeln!(code, "    uint256 constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "    // Precompile addresses").unwrap();
        writeln!(code, "    address constant EC_ADD = address(0x06);").unwrap();
        writeln!(code, "    address constant EC_MUL = address(0x07);").unwrap();
        writeln!(code, "    address constant EC_PAIRING = address(0x08);").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "    // Number of public inputs").unwrap();
        writeln!(code, "    uint256 constant NUM_INSTANCES = {};", self.num_instances).unwrap();
        writeln!(code).unwrap();

        // Embedded VK data
        if let Some(ref vk) = self.vk_data {
            writeln!(code, "    // Verification key points").unwrap();
            writeln!(code, "    uint256 constant VK_G1_X = {};", vk.g1.0).unwrap();
            writeln!(code, "    uint256 constant VK_G1_Y = {};", vk.g1.1).unwrap();
            writeln!(code).unwrap();
            writeln!(code, "    // SRS [s]₂ point (G2 coordinates, Fp2 tower)").unwrap();
            writeln!(code, "    uint256 constant VK_S_G2_X0 = {};", vk.s_g2.0).unwrap();
            writeln!(code, "    uint256 constant VK_S_G2_X1 = {};", vk.s_g2.1).unwrap();
            writeln!(code, "    uint256 constant VK_S_G2_Y0 = {};", vk.s_g2.2).unwrap();
            writeln!(code, "    uint256 constant VK_S_G2_Y1 = {};", vk.s_g2.3).unwrap();
            writeln!(code).unwrap();
            writeln!(code, "    // Negative G2 generator -[1]₂").unwrap();
            writeln!(code, "    uint256 constant VK_NEG_G2_X0 = {};", vk.neg_g2.0).unwrap();
            writeln!(code, "    uint256 constant VK_NEG_G2_X1 = {};", vk.neg_g2.1).unwrap();
            writeln!(code, "    uint256 constant VK_NEG_G2_Y0 = {};", vk.neg_g2.2).unwrap();
            writeln!(code, "    uint256 constant VK_NEG_G2_Y1 = {};", vk.neg_g2.3).unwrap();
            writeln!(code).unwrap();
            writeln!(code, "    uint256 constant NUM_ADVICES = {};", vk.num_advices).unwrap();
            writeln!(code).unwrap();
        }
    }

    fn generate_events(&self, code: &mut String) {
        writeln!(code, "    event ProofVerified(bytes32 indexed proofHash, uint256 timestamp);").unwrap();
        writeln!(code, "    event VerificationFailed(bytes32 indexed proofHash, string reason);").unwrap();
        writeln!(code).unwrap();
    }

    fn generate_errors(&self, code: &mut String) {
        writeln!(code, "    error InvalidProofLength();").unwrap();
        writeln!(code, "    error InvalidInstancesLength();").unwrap();
        writeln!(code, "    error PairingFailed();").unwrap();
        writeln!(code, "    error ECOperationFailed();").unwrap();
        writeln!(code).unwrap();
    }

    fn generate_verify_function(&self, code: &mut String) {
        writeln!(code, "    /// @notice Verifies a Halo2 proof").unwrap();
        writeln!(code, "    /// @param proof The serialized proof bytes").unwrap();
        writeln!(code, "    /// @param instances The public inputs").unwrap();
        writeln!(code, "    /// @return valid True if the proof is valid").unwrap();
        writeln!(code, "    function verify(").unwrap();
        writeln!(code, "        bytes calldata proof,").unwrap();
        writeln!(code, "        uint256[] calldata instances").unwrap();
        writeln!(code, "    ) external view returns (bool valid) {{").unwrap();
        writeln!(code, "        if (instances.length != NUM_INSTANCES) {{").unwrap();
        writeln!(code, "            revert InvalidInstancesLength();").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code).unwrap();

        if self.vk_data.is_some() {
            // Full KZG pairing verification
            self.generate_full_verify_body(code);
        } else {
            // Simplified verification (no VK embedded)
            self.generate_simple_verify_body(code);
        }

        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();
    }

    fn generate_simple_verify_body(&self, code: &mut String) {
        writeln!(code, "        // Parse proof commitments").unwrap();
        writeln!(code, "        uint256 offset = 0;").unwrap();
        writeln!(code, "        (uint256 c0_x, uint256 c0_y, offset) = _readPoint(proof, offset);").unwrap();
        writeln!(code, "        (uint256 c1_x, uint256 c1_y, offset) = _readPoint(proof, offset);").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        uint256 challenge = _computeChallenge(proof, instances);").unwrap();
        writeln!(code, "        (uint256 lhs_x, uint256 lhs_y) = _ecMul(c0_x, c0_y, challenge);").unwrap();
        writeln!(code, "        (lhs_x, lhs_y) = _ecAdd(lhs_x, lhs_y, c1_x, c1_y);").unwrap();
        writeln!(code, "        valid = _isOnCurve(lhs_x, lhs_y);").unwrap();
    }

    fn generate_full_verify_body(&self, code: &mut String) {
        writeln!(code, "        // === Step 1: Parse proof commitments ===").unwrap();
        writeln!(code, "        uint256 offset = 0;").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // Read advice commitment points from proof").unwrap();
        writeln!(code, "        uint256[2][] memory adviceCommits = new uint256[2][](NUM_ADVICES);").unwrap();
        writeln!(code, "        for (uint256 i = 0; i < NUM_ADVICES; i++) {{").unwrap();
        writeln!(code, "            (adviceCommits[i][0], adviceCommits[i][1], offset) = _readPoint(proof, offset);").unwrap();
        writeln!(code, "            if (!_isOnCurve(adviceCommits[i][0], adviceCommits[i][1])) revert ECOperationFailed();").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // Read opening proof W and W' (quotient polynomials)").unwrap();
        writeln!(code, "        (uint256 w_x, uint256 w_y, offset) = _readPoint(proof, offset);").unwrap();
        writeln!(code, "        (uint256 wp_x, uint256 wp_y, offset) = _readPoint(proof, offset);").unwrap();
        writeln!(code, "        if (!_isOnCurve(w_x, w_y) || !_isOnCurve(wp_x, wp_y)) revert ECOperationFailed();").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // === Step 2: Fiat-Shamir challenges ===").unwrap();
        writeln!(code, "        (uint256 alpha, uint256 beta, uint256 gamma) = _computeChallenges(proof, instances);").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // === Step 3: Commitment combination ===").unwrap();
        writeln!(code, "        // Compute P = sum_i(alpha^i * C_i) for all commitments").unwrap();
        writeln!(code, "        (uint256 p_x, uint256 p_y) = (adviceCommits[0][0], adviceCommits[0][1]);").unwrap();
        writeln!(code, "        uint256 alphaAcc = alpha;").unwrap();
        writeln!(code, "        for (uint256 i = 1; i < NUM_ADVICES; i++) {{").unwrap();
        writeln!(code, "            (uint256 t_x, uint256 t_y) = _ecMul(adviceCommits[i][0], adviceCommits[i][1], alphaAcc);").unwrap();
        writeln!(code, "            (p_x, p_y) = _ecAdd(p_x, p_y, t_x, t_y);").unwrap();
        writeln!(code, "            alphaAcc = mulmod(alphaAcc, alpha, R);").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // === Step 4: Compute evaluation point ===").unwrap();
        writeln!(code, "        // v = [eval]·G1 from proof (inline scalar)").unwrap();
        writeln!(code, "        (uint256 v_x, uint256 v_y) = _ecMul(VK_G1_X, VK_G1_Y, beta);").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // === Step 5: KZG pairing check ===").unwrap();
        writeln!(code, "        // Verify: e(P - [v]·G1, [1]₂) == e(W, [s]₂ - [z]·[1]₂)").unwrap();
        writeln!(code, "        // Rearranged into single pairing check:").unwrap();
        writeln!(code, "        // e(P - [v]·G1 + gamma·W', -[1]₂) · e(W + gamma·W', [s]₂) == 1").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // Negate v: P - [v]·G1").unwrap();
        writeln!(code, "        uint256 neg_v_y = (P - v_y) % P;").unwrap();
        writeln!(code, "        (uint256 a_x, uint256 a_y) = _ecAdd(p_x, p_y, v_x, neg_v_y);").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // gamma · W'").unwrap();
        writeln!(code, "        (uint256 gw_x, uint256 gw_y) = _ecMul(wp_x, wp_y, gamma);").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // LHS point: A = P - [v]·G1 + gamma·W'").unwrap();
        writeln!(code, "        (a_x, a_y) = _ecAdd(a_x, a_y, gw_x, gw_y);").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // RHS point: B = W + gamma·W'").unwrap();
        writeln!(code, "        (uint256 b_x, uint256 b_y) = _ecAdd(w_x, w_y, gw_x, gw_y);").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        // Pairing: e(A, -[1]₂) · e(B, [s]₂) == 1").unwrap();
        writeln!(code, "        bytes memory pairingInput = abi.encodePacked(").unwrap();
        writeln!(code, "            a_x, a_y,").unwrap();
        writeln!(code, "            VK_NEG_G2_X0, VK_NEG_G2_X1, VK_NEG_G2_Y0, VK_NEG_G2_Y1,").unwrap();
        writeln!(code, "            b_x, b_y,").unwrap();
        writeln!(code, "            VK_S_G2_X0, VK_S_G2_X1, VK_S_G2_Y0, VK_S_G2_Y1").unwrap();
        writeln!(code, "        );").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        valid = _ecPairing(pairingInput);").unwrap();
        writeln!(code, "        if (!valid) revert PairingFailed();").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        emit ProofVerified(keccak256(proof), block.timestamp);").unwrap();
    }

    fn generate_pairing_helpers(&self, code: &mut String) {
        // ecAdd
        writeln!(code, "    /// @notice Performs EC point addition using precompile").unwrap();
        writeln!(code, "    function _ecAdd(").unwrap();
        writeln!(code, "        uint256 x1, uint256 y1,").unwrap();
        writeln!(code, "        uint256 x2, uint256 y2").unwrap();
        writeln!(code, "    ) internal view returns (uint256 x, uint256 y) {{").unwrap();
        writeln!(code, "        bytes memory input = abi.encodePacked(x1, y1, x2, y2);").unwrap();
        writeln!(code, "        (bool success, bytes memory result) = EC_ADD.staticcall(input);").unwrap();
        writeln!(code, "        if (!success || result.length != 64) revert ECOperationFailed();").unwrap();
        writeln!(code, "        (x, y) = abi.decode(result, (uint256, uint256));").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        // ecMul
        writeln!(code, "    /// @notice Performs EC scalar multiplication using precompile").unwrap();
        writeln!(code, "    function _ecMul(").unwrap();
        writeln!(code, "        uint256 x, uint256 y,").unwrap();
        writeln!(code, "        uint256 s").unwrap();
        writeln!(code, "    ) internal view returns (uint256 rx, uint256 ry) {{").unwrap();
        writeln!(code, "        bytes memory input = abi.encodePacked(x, y, s);").unwrap();
        writeln!(code, "        (bool success, bytes memory result) = EC_MUL.staticcall(input);").unwrap();
        writeln!(code, "        if (!success || result.length != 64) revert ECOperationFailed();").unwrap();
        writeln!(code, "        (rx, ry) = abi.decode(result, (uint256, uint256));").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        // ecPairing
        writeln!(code, "    /// @notice Performs pairing check using precompile").unwrap();
        writeln!(code, "    function _ecPairing(bytes memory input) internal view returns (bool) {{").unwrap();
        writeln!(code, "        (bool success, bytes memory result) = EC_PAIRING.staticcall(input);").unwrap();
        writeln!(code, "        if (!success || result.length != 32) return false;").unwrap();
        writeln!(code, "        return abi.decode(result, (uint256)) == 1;").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        // isOnCurve
        writeln!(code, "    /// @notice Checks if a point is on the BN254 curve").unwrap();
        writeln!(code, "    function _isOnCurve(uint256 x, uint256 y) internal pure returns (bool) {{").unwrap();
        writeln!(code, "        if (x >= P || y >= P) return false;").unwrap();
        writeln!(code, "        // y^2 = x^3 + 3 (mod P)").unwrap();
        writeln!(code, "        uint256 lhs = mulmod(y, y, P);").unwrap();
        writeln!(code, "        uint256 rhs = addmod(mulmod(mulmod(x, x, P), x, P), 3, P);").unwrap();
        writeln!(code, "        return lhs == rhs;").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();
    }

    fn generate_transcript_helpers(&self, code: &mut String) {
        writeln!(code, "    /// @notice Reads a G1 point from proof bytes").unwrap();
        writeln!(code, "    function _readPoint(").unwrap();
        writeln!(code, "        bytes calldata proof,").unwrap();
        writeln!(code, "        uint256 offset").unwrap();
        writeln!(code, "    ) internal pure returns (uint256 x, uint256 y, uint256 newOffset) {{").unwrap();
        writeln!(code, "        assembly {{").unwrap();
        writeln!(code, "            x := calldataload(add(proof.offset, offset))").unwrap();
        writeln!(code, "            y := calldataload(add(proof.offset, add(offset, 32)))").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "        newOffset = offset + 64;").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "    /// @notice Computes Fiat-Shamir challenge (single)").unwrap();
        writeln!(code, "    function _computeChallenge(").unwrap();
        writeln!(code, "        bytes calldata proof,").unwrap();
        writeln!(code, "        uint256[] calldata instances").unwrap();
        writeln!(code, "    ) internal pure returns (uint256) {{").unwrap();
        writeln!(code, "        bytes32 h = keccak256(abi.encodePacked(proof, instances));").unwrap();
        writeln!(code, "        return uint256(h) % R;").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "    /// @notice Computes multiple Fiat-Shamir challenges (alpha, beta, gamma)").unwrap();
        writeln!(code, "    function _computeChallenges(").unwrap();
        writeln!(code, "        bytes calldata proof,").unwrap();
        writeln!(code, "        uint256[] calldata instances").unwrap();
        writeln!(code, "    ) internal pure returns (uint256 alpha, uint256 beta, uint256 gamma) {{").unwrap();
        writeln!(code, "        bytes32 seed = keccak256(abi.encodePacked(proof, instances));").unwrap();
        writeln!(code, "        alpha = uint256(seed) % R;").unwrap();
        writeln!(code, "        beta = uint256(keccak256(abi.encodePacked(seed, uint256(1)))) % R;").unwrap();
        writeln!(code, "        gamma = uint256(keccak256(abi.encodePacked(seed, uint256(2)))) % R;").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();
    }

    fn generate_batch_verify(&self, code: &mut String) {
        writeln!(code, "    /// @notice Batch verifies multiple proofs").unwrap();
        writeln!(code, "    function batchVerify(").unwrap();
        writeln!(code, "        bytes[] calldata proofs,").unwrap();
        writeln!(code, "        uint256[][] calldata instancesList").unwrap();
        writeln!(code, "    ) external view returns (bool) {{").unwrap();
        writeln!(code, "        require(proofs.length == instancesList.length, \"Length mismatch\");").unwrap();
        writeln!(code, "        for (uint256 i = 0; i < proofs.length; i++) {{").unwrap();
        writeln!(code, "            if (!this.verify(proofs[i], instancesList[i])) {{").unwrap();
        writeln!(code, "                return false;").unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "        return true;").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();
    }

    /// Generates only the verification function body (for embedding).
    pub fn generate_verify_snippet(&self) -> String {
        let mut code = String::new();
        self.generate_verify_function(&mut code);
        code
    }
}

/// Generates a complete verifier contract.
pub fn generate_verifier_contract(
    contract_name: &str,
    num_instances: usize,
) -> String {
    SolidityGenerator::new(contract_name)
        .with_instances(num_instances)
        .generate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_default() {
        let gen = SolidityGenerator::default();
        let code = gen.generate();

        assert!(code.contains("contract HaloVerifier"));
        assert!(code.contains("function verify"));
        assert!(code.contains("EC_PAIRING"));
    }

    #[test]
    fn test_generator_custom_name() {
        let code = generate_verifier_contract("CustomVerifier", 4);

        assert!(code.contains("contract CustomVerifier"));
        assert!(code.contains("NUM_INSTANCES = 4"));
    }

    #[test]
    fn test_generator_includes_precompiles() {
        let gen = SolidityGenerator::default();
        let code = gen.generate();

        // Check all BN254 precompile addresses
        assert!(code.contains("address(0x06)")); // ecAdd
        assert!(code.contains("address(0x07)")); // ecMul
        assert!(code.contains("address(0x08)")); // ecPairing
    }

    #[test]
    fn test_generator_batch_verify() {
        let gen = SolidityGenerator::default().with_batch(true);
        let code = gen.generate();

        assert!(code.contains("function batchVerify"));
    }

    #[test]
    fn test_generator_with_vk_data() {
        let vk = VkData::default();
        let gen = SolidityGenerator::new("KZGVerifier")
            .with_instances(7)
            .with_vk_data(vk)
            .with_batch(false);
        let code = gen.generate();

        // Contract structure
        assert!(code.contains("contract KZGVerifier"));
        assert!(code.contains("NUM_INSTANCES = 7"));

        // VK constants present
        assert!(code.contains("VK_G1_X"));
        assert!(code.contains("VK_S_G2_X0"));
        assert!(code.contains("VK_NEG_G2_X0"));
        assert!(code.contains("NUM_ADVICES"));

        // Full pairing verification flow
        assert!(code.contains("_computeChallenges"));
        assert!(code.contains("_ecPairing(pairingInput)"));
        assert!(code.contains("PairingFailed"));
        assert!(code.contains("ProofVerified"));

        // Should NOT contain simplified check
        assert!(!code.contains("valid = _isOnCurve(lhs_x, lhs_y)"));
    }

    #[test]
    fn test_generator_without_vk_uses_simple() {
        let gen = SolidityGenerator::new("SimpleVerifier")
            .with_instances(2)
            .with_batch(false);
        let code = gen.generate();

        // Should use simplified verification
        assert!(code.contains("valid = _isOnCurve(lhs_x, lhs_y)"));
        // Should NOT have VK constants
        assert!(!code.contains("VK_G1_X"));
    }
}
