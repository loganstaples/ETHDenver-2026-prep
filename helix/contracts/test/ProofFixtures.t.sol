// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";

/// @title ProofFixtures
/// @notice Library for loading Rust-generated proof fixtures in Foundry tests
/// @dev Parses JSON fixtures from contracts/fixtures/rust_proofs/
library ProofFixtures {
    // ============ Errors ============
    error FixtureNotFound(string path);
    error InvalidFixtureFormat(string message);

    // ============ Structs ============

    /// @notice Parsed proof fixture data
    struct ProofFixture {
        string description;
        bytes proofBytes;
        uint256[] publicInputs;
        uint256 stepNumber;
        uint256 modelId;
        uint256 roundId;
        bool shouldVerify;
    }

    /// @notice Batch of proof fixtures for multi-step testing
    struct BatchFixture {
        ProofFixture[] proofs;
        uint256 totalSteps;
    }

    // ============ Constants ============

    /// @notice Base path for fixtures relative to project root
    string constant FIXTURES_BASE_PATH = "fixtures/rust_proofs/";

    /// @notice Expected proof length in bytes
    uint256 constant EXPECTED_PROOF_LENGTH = 320;

    /// @notice Expected number of public inputs
    uint256 constant EXPECTED_PUBLIC_INPUTS = 7;
}

/// @title ProofFixturesLoader
/// @notice Test helper contract for loading proof fixtures
/// @dev Inherits from Test to use vm.readFile and vm.parseJson
contract ProofFixturesLoader is Test {
    using ProofFixtures for *;

    // ============ Events ============
    event FixtureLoaded(string path, uint256 proofLength, uint256 numPublicInputs);
    event BatchLoaded(uint256 numProofs);

    // ============ Loading Functions ============

    /// @notice Loads a single proof fixture from JSON file
    /// @param filename Name of the JSON file (without path prefix)
    /// @return fixture The parsed proof fixture
    function loadProofFixture(string memory filename) public returns (ProofFixtures.ProofFixture memory fixture) {
        string memory path = string.concat(ProofFixtures.FIXTURES_BASE_PATH, filename);
        string memory json = vm.readFile(path);

        // Parse proof bytes
        bytes memory proofHex = vm.parseJson(json, ".proof.bytes_hex");
        fixture.proofBytes = abi.decode(proofHex, (bytes));

        // Parse public inputs array
        fixture.publicInputs = new uint256[](7);
        bytes memory inputsData = vm.parseJson(json, ".public_inputs.values_array");
        bytes32[] memory inputsRaw = abi.decode(inputsData, (bytes32[]));
        for (uint i = 0; i < 7; i++) {
            fixture.publicInputs[i] = uint256(inputsRaw[i]);
        }

        // Parse metadata
        bytes memory stepData = vm.parseJson(json, ".step_number");
        fixture.stepNumber = abi.decode(stepData, (uint256));

        bytes memory modelData = vm.parseJson(json, ".metadata.model_id");
        fixture.modelId = abi.decode(modelData, (uint256));

        bytes memory roundData = vm.parseJson(json, ".metadata.round_id");
        fixture.roundId = abi.decode(roundData, (uint256));

        bytes memory descData = vm.parseJson(json, ".description");
        fixture.description = abi.decode(descData, (string));

        // Default to true for validity (invalid proofs have explicit flag)
        fixture.shouldVerify = true;

        emit FixtureLoaded(path, fixture.proofBytes.length, fixture.publicInputs.length);
    }

    /// @notice Loads an invalid proof fixture (expected to fail verification)
    /// @param filename Name of the JSON file
    /// @return fixture The parsed proof fixture with shouldVerify = false
    function loadInvalidProofFixture(string memory filename) public returns (ProofFixtures.ProofFixture memory fixture) {
        fixture = loadProofFixture(filename);
        fixture.shouldVerify = false;
    }

    /// @notice Loads multiple proof fixtures for batch testing
    /// @param filenames Array of JSON filenames to load
    /// @return batch The batch of parsed fixtures
    function loadBatchFixtures(string[] memory filenames) public returns (ProofFixtures.BatchFixture memory batch) {
        batch.proofs = new ProofFixtures.ProofFixture[](filenames.length);
        batch.totalSteps = filenames.length;

        for (uint i = 0; i < filenames.length; i++) {
            batch.proofs[i] = loadProofFixture(filenames[i]);
        }

        emit BatchLoaded(filenames.length);
    }

    /// @notice Creates proof bytes directly from hex string (for inline tests)
    /// @param hexString The hex-encoded proof bytes
    /// @return proof The decoded proof bytes
    function proofFromHex(string memory hexString) public pure returns (bytes memory proof) {
        proof = vm.parseBytes(hexString);
    }

    /// @notice Creates public inputs array from individual values
    /// @return inputs The public inputs array
    function createPublicInputs(
        uint256 oldHashLo,
        uint256 oldHashHi,
        uint256 newHashLo,
        uint256 newHashHi,
        uint256 loss,
        uint256 errorBound,
        uint256 stepNumber
    ) public pure returns (uint256[] memory inputs) {
        inputs = new uint256[](7);
        inputs[0] = oldHashLo;
        inputs[1] = oldHashHi;
        inputs[2] = newHashLo;
        inputs[3] = newHashHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = stepNumber;
    }

    /// @notice Computes the commitment hash matching Solidity's _hashPair
    /// @param lo Lower 128 bits
    /// @param hi Upper 128 bits
    /// @return commitment The keccak256 hash as uint256
    function computeCommitment(uint256 lo, uint256 hi) public pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    /// @notice Validates fixture format
    /// @param fixture The fixture to validate
    /// @return valid True if fixture passes format validation
    function validateFixtureFormat(ProofFixtures.ProofFixture memory fixture) public pure returns (bool valid) {
        if (fixture.proofBytes.length < ProofFixtures.EXPECTED_PROOF_LENGTH) {
            return false;
        }
        if (fixture.publicInputs.length != ProofFixtures.EXPECTED_PUBLIC_INPUTS) {
            return false;
        }
        return true;
    }

    // ============ Helper: Load Standard Test Fixtures ============

    /// @notice Loads the standard valid step 1 fixture
    function loadValidStep1() public returns (ProofFixtures.ProofFixture memory) {
        return loadProofFixture("valid_step_1.json");
    }

    /// @notice Loads the standard valid step 2 fixture
    function loadValidStep2() public returns (ProofFixtures.ProofFixture memory) {
        return loadProofFixture("valid_step_2.json");
    }

    /// @notice Loads the standard valid step 3 fixture
    function loadValidStep3() public returns (ProofFixtures.ProofFixture memory) {
        return loadProofFixture("valid_step_3.json");
    }

    /// @notice Loads the standard invalid proof fixture
    function loadInvalidProof() public returns (ProofFixtures.ProofFixture memory) {
        return loadInvalidProofFixture("invalid_proof.json");
    }

    /// @notice Loads all 3 valid step fixtures as a batch
    function loadAllValidSteps() public returns (ProofFixtures.BatchFixture memory) {
        string[] memory files = new string[](3);
        files[0] = "valid_step_1.json";
        files[1] = "valid_step_2.json";
        files[2] = "valid_step_3.json";
        return loadBatchFixtures(files);
    }
}

/// @title ProofFixtureHardcoded
/// @notice Provides hardcoded proof data for tests that can't use file I/O
/// @dev Use this when running tests in environments without filesystem access
library ProofFixtureHardcoded {

    /// @notice BN254 G1 generator point (1, 2) - valid curve point
    uint256 constant G1_X = 1;
    uint256 constant G1_Y = 2;

    /// @notice A second valid G1 point (from SRS)
    uint256 constant G1_2_X = 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2;
    uint256 constant G1_2_Y = 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed;

    /// @notice A third valid G1 point
    uint256 constant G1_3_X = 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b;
    uint256 constant G1_3_Y = 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa;

    /// @notice Creates a valid 320-byte proof with real BN254 curve points
    /// @return proof Valid proof bytes
    function createValidProof() internal pure returns (bytes memory proof) {
        proof = new bytes(320);

        // Point 1: Generator (1, 2)
        assembly {
            mstore(add(proof, 32), G1_X)
            mstore(add(proof, 64), G1_Y)
        }

        // Point 2: Valid SRS point
        assembly {
            mstore(add(proof, 96), G1_2_X)
            mstore(add(proof, 128), G1_2_Y)
        }

        // Point 3: Another valid point
        assembly {
            mstore(add(proof, 160), G1_3_X)
            mstore(add(proof, 192), G1_3_Y)
        }

        // W point: Generator
        assembly {
            mstore(add(proof, 224), G1_X)
            mstore(add(proof, 256), G1_Y)
        }

        // W' point: Generator
        assembly {
            mstore(add(proof, 288), G1_X)
            mstore(add(proof, 320), G1_Y)
        }
    }

    /// @notice Creates an invalid proof with a point not on the BN254 curve
    /// @return proof Invalid proof bytes (first point has wrong y-coordinate)
    function createInvalidProof() internal pure returns (bytes memory proof) {
        proof = new bytes(320);

        // Invalid point: x=1, y=3 is NOT on the curve y^2 = x^3 + 3
        // (1^3 + 3 = 4, but 3^2 = 9 != 4)
        assembly {
            mstore(add(proof, 32), 1)
            mstore(add(proof, 64), 3)
        }

        // Fill rest with valid points
        assembly {
            mstore(add(proof, 96), G1_X)
            mstore(add(proof, 128), G1_Y)
            mstore(add(proof, 160), G1_X)
            mstore(add(proof, 192), G1_Y)
            mstore(add(proof, 224), G1_X)
            mstore(add(proof, 256), G1_Y)
            mstore(add(proof, 288), G1_X)
            mstore(add(proof, 320), G1_Y)
        }
    }

    /// @notice Creates a corrupted proof (random bytes flipped)
    /// @param seed Random seed for corruption
    /// @return proof Corrupted proof bytes
    function createCorruptedProof(uint256 seed) internal pure returns (bytes memory proof) {
        proof = createValidProof();

        // Corrupt a random byte
        uint256 corruptIndex = (seed % 320);
        assembly {
            let ptr := add(proof, add(32, corruptIndex))
            let current := mload(ptr)
            // XOR with random value to corrupt
            mstore(ptr, xor(current, seed))
        }
    }

    /// @notice Creates valid public inputs for step 1
    /// @return inputs Public inputs array
    function createValidPublicInputsStep1() internal pure returns (uint256[] memory inputs) {
        inputs = new uint256[](7);
        inputs[0] = 0x3039;  // oldHashLo (12345)
        inputs[1] = 0x3042;  // oldHashHi (12354)
        inputs[2] = 0x7b16;  // newHashLo (31510)
        inputs[3] = 0x7b32;  // newHashHi (31538)
        inputs[4] = 1000;    // loss
        inputs[5] = 10;      // errorBound
        inputs[6] = 1;       // stepNumber
    }

    /// @notice Creates valid public inputs for step 2
    function createValidPublicInputsStep2() internal pure returns (uint256[] memory inputs) {
        inputs = new uint256[](7);
        inputs[0] = 0x7b16;  // oldHashLo (matches step 1 newHashLo)
        inputs[1] = 0x7b32;  // oldHashHi
        inputs[2] = 0xc350;  // newHashLo (50000)
        inputs[3] = 0xc36e;  // newHashHi
        inputs[4] = 800;     // loss (decreased)
        inputs[5] = 8;       // errorBound
        inputs[6] = 2;       // stepNumber
    }

    /// @notice Creates valid public inputs for step 3
    function createValidPublicInputsStep3() internal pure returns (uint256[] memory inputs) {
        inputs = new uint256[](7);
        inputs[0] = 0xc350;  // oldHashLo
        inputs[1] = 0xc36e;  // oldHashHi
        inputs[2] = 0x10b8e; // newHashLo (68494)
        inputs[3] = 0x10bb2; // newHashHi
        inputs[4] = 700;     // loss (decreased)
        inputs[5] = 5;       // errorBound
        inputs[6] = 3;       // stepNumber
    }
}
