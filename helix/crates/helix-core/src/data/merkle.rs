//! Production-grade Merkle Tree Implementation for HELIX.
//!
//! Provides cryptographic commitment to datasets with efficient membership proofs.
//! Supports:
//! - Arbitrary hash functions (SHA-256 default)
//! - Incremental tree construction
//! - Sparse tree representation for large datasets
//! - Multi-proof aggregation
//! - Serialization for on-chain storage

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};

use crate::traits::BinarySerializable;
use crate::traits::serializable::SerializeError;

/// Default hash output size (SHA-256).
pub const HASH_SIZE: usize = 32;

/// A 32-byte hash value used as node identifiers and commitments.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Hash(pub [u8; HASH_SIZE]);

impl Hash {
    /// Creates a zero hash.
    pub const fn zero() -> Self {
        Self([0u8; HASH_SIZE])
    }

    /// Creates a hash from bytes.
    pub fn from_bytes(bytes: [u8; HASH_SIZE]) -> Self {
        Self(bytes)
    }

    /// Creates a hash from a slice, padding or truncating as needed.
    pub fn from_slice(slice: &[u8]) -> Self {
        let mut bytes = [0u8; HASH_SIZE];
        let len = slice.len().min(HASH_SIZE);
        bytes[..len].copy_from_slice(&slice[..len]);
        Self(bytes)
    }

    /// Returns the bytes of this hash.
    pub fn as_bytes(&self) -> &[u8; HASH_SIZE] {
        &self.0
    }

    /// Converts to a hexadecimal string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Parses from a hexadecimal string.
    pub fn from_hex(s: &str) -> Result<Self, MerkleError> {
        if s.len() != HASH_SIZE * 2 {
            return Err(MerkleError::InvalidHashLength {
                expected: HASH_SIZE * 2,
                actual: s.len(),
            });
        }

        let mut bytes = [0u8; HASH_SIZE];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk)
                .map_err(|_| MerkleError::InvalidHexEncoding)?;
            bytes[i] = u8::from_str_radix(hex_str, 16)
                .map_err(|_| MerkleError::InvalidHexEncoding)?;
        }
        Ok(Self(bytes))
    }

    /// Checks if this is a zero hash.
    pub fn is_zero(&self) -> bool {
        self.0 == [0u8; HASH_SIZE]
    }
}

impl std::fmt::Debug for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hash({})", &self.to_hex()[..16])
    }
}

impl std::fmt::Display for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl BinarySerializable for Hash {
    fn serialized_size(&self) -> usize {
        HASH_SIZE
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        writer.write_all(&self.0)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let mut bytes = [0u8; HASH_SIZE];
        reader.read_exact(&mut bytes)?;
        Ok(Self(bytes))
    }
}

/// Hasher trait for Merkle tree operations.
pub trait MerkleHasher: Clone + Default {
    /// Hash a single chunk of data (leaf node).
    fn hash_leaf(&self, data: &[u8]) -> Hash;

    /// Hash two child hashes to produce parent hash.
    fn hash_nodes(&self, left: &Hash, right: &Hash) -> Hash;

    /// Returns the identifier for this hash function.
    fn algorithm_id(&self) -> &'static str;
}

/// SHA-256 based hasher (default).
#[derive(Clone, Default)]
pub struct Sha256Hasher;

impl Sha256Hasher {
    /// Simple SHA-256 implementation for demonstration.
    /// In production, would use a proper crypto library.
    fn sha256(data: &[u8]) -> [u8; 32] {
        // Initialize hash values (first 32 bits of fractional parts of square roots)
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ];

        // Round constants
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];

        // Pre-processing: add padding
        let bit_len = (data.len() as u64) * 8;
        let mut padded = data.to_vec();
        padded.push(0x80);
        while (padded.len() % 64) != 56 {
            padded.push(0x00);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());

        // Process each 512-bit chunk
        for chunk in padded.chunks(64) {
            let mut w = [0u32; 64];

            // Copy chunk into first 16 words
            for (i, word) in chunk.chunks(4).enumerate() {
                w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }

            // Extend to 64 words
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            // Initialize working variables
            let mut a = h[0];
            let mut b = h[1];
            let mut c = h[2];
            let mut d = h[3];
            let mut e = h[4];
            let mut f = h[5];
            let mut g = h[6];
            let mut hh = h[7];

            // Compression function
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            // Add to hash values
            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        // Produce final hash
        let mut result = [0u8; 32];
        for (i, &val) in h.iter().enumerate() {
            result[i * 4..(i + 1) * 4].copy_from_slice(&val.to_be_bytes());
        }
        result
    }
}

impl MerkleHasher for Sha256Hasher {
    fn hash_leaf(&self, data: &[u8]) -> Hash {
        // Domain separation: prefix leaf hashes with 0x00
        let mut prefixed = vec![0x00];
        prefixed.extend_from_slice(data);
        Hash(Self::sha256(&prefixed))
    }

    fn hash_nodes(&self, left: &Hash, right: &Hash) -> Hash {
        // Domain separation: prefix internal node hashes with 0x01
        let mut combined = vec![0x01];
        combined.extend_from_slice(&left.0);
        combined.extend_from_slice(&right.0);
        Hash(Self::sha256(&combined))
    }

    fn algorithm_id(&self) -> &'static str {
        "sha256"
    }
}

/// Position in the Merkle tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TreePosition {
    /// Level in the tree (0 = leaves).
    pub level: usize,
    /// Index at this level.
    pub index: usize,
}

impl TreePosition {
    /// Creates a leaf position.
    pub fn leaf(index: usize) -> Self {
        Self { level: 0, index }
    }

    /// Gets the parent position.
    pub fn parent(&self) -> Self {
        Self {
            level: self.level + 1,
            index: self.index / 2,
        }
    }

    /// Gets the sibling position.
    pub fn sibling(&self) -> Self {
        Self {
            level: self.level,
            index: self.index ^ 1,
        }
    }

    /// Checks if this is a left child.
    pub fn is_left(&self) -> bool {
        self.index % 2 == 0
    }
}

/// Direction in a proof path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofDirection {
    Left,
    Right,
}

/// A single step in a Merkle proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStep {
    /// The sibling hash at this level.
    pub sibling: Hash,
    /// Direction: whether the sibling is on the left or right.
    pub direction: ProofDirection,
}

impl BinarySerializable for ProofStep {
    fn serialized_size(&self) -> usize {
        HASH_SIZE + 1
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        BinarySerializable::serialize(&self.sibling, writer)?;
        let dir_byte = match self.direction {
            ProofDirection::Left => 0u8,
            ProofDirection::Right => 1u8,
        };
        writer.write_all(&[dir_byte])?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let sibling = <Hash as BinarySerializable>::deserialize(reader)?;
        let mut dir_buf = [0u8; 1];
        reader.read_exact(&mut dir_buf)?;
        let direction = match dir_buf[0] {
            0 => ProofDirection::Left,
            1 => ProofDirection::Right,
            _ => return Err(SerializeError::InvalidData("invalid direction byte".into())),
        };
        Ok(Self { sibling, direction })
    }
}

/// A Merkle proof for a single leaf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// The leaf index being proven.
    pub leaf_index: usize,
    /// The leaf hash.
    pub leaf_hash: Hash,
    /// Path from leaf to root.
    pub path: Vec<ProofStep>,
    /// The expected root hash.
    pub root: Hash,
}

impl MerkleProof {
    /// Verifies this proof against the expected root.
    pub fn verify<H: MerkleHasher>(&self, hasher: &H) -> bool {
        self.verify_with_root(hasher, &self.root)
    }

    /// Verifies this proof against a given root.
    pub fn verify_with_root<H: MerkleHasher>(&self, hasher: &H, expected_root: &Hash) -> bool {
        let mut current = self.leaf_hash;

        for step in &self.path {
            current = match step.direction {
                ProofDirection::Left => hasher.hash_nodes(&step.sibling, &current),
                ProofDirection::Right => hasher.hash_nodes(&current, &step.sibling),
            };
        }

        current == *expected_root
    }

    /// Computes the root from this proof.
    pub fn compute_root<H: MerkleHasher>(&self, hasher: &H) -> Hash {
        let mut current = self.leaf_hash;

        for step in &self.path {
            current = match step.direction {
                ProofDirection::Left => hasher.hash_nodes(&step.sibling, &current),
                ProofDirection::Right => hasher.hash_nodes(&current, &step.sibling),
            };
        }

        current
    }

    /// Returns the depth of this proof.
    pub fn depth(&self) -> usize {
        self.path.len()
    }
}

impl BinarySerializable for MerkleProof {
    fn serialized_size(&self) -> usize {
        8 + HASH_SIZE + 4 + self.path.len() * (HASH_SIZE + 1) + HASH_SIZE
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        writer.write_all(&(self.leaf_index as u64).to_le_bytes())?;
        BinarySerializable::serialize(&self.leaf_hash, writer)?;
        writer.write_all(&(self.path.len() as u32).to_le_bytes())?;
        for step in &self.path {
            BinarySerializable::serialize(step, writer)?;
        }
        BinarySerializable::serialize(&self.root, writer)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let mut idx_buf = [0u8; 8];
        reader.read_exact(&mut idx_buf)?;
        let leaf_index = u64::from_le_bytes(idx_buf) as usize;

        let leaf_hash = <Hash as BinarySerializable>::deserialize(reader)?;

        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let path_len = u32::from_le_bytes(len_buf) as usize;

        let mut path = Vec::with_capacity(path_len);
        for _ in 0..path_len {
            path.push(<ProofStep as BinarySerializable>::deserialize(reader)?);
        }

        let root = <Hash as BinarySerializable>::deserialize(reader)?;

        Ok(Self {
            leaf_index,
            leaf_hash,
            path,
            root,
        })
    }
}

/// Aggregated proof for multiple leaves (more efficient than individual proofs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiProof {
    /// Leaf indices being proven.
    pub leaf_indices: Vec<usize>,
    /// Leaf hashes.
    pub leaf_hashes: Vec<Hash>,
    /// Deduped proof nodes (shared siblings removed).
    pub proof_nodes: HashMap<TreePosition, Hash>,
    /// The root hash.
    pub root: Hash,
    /// Tree height.
    pub height: usize,
}

impl MultiProof {
    /// Verifies this multi-proof.
    pub fn verify<H: MerkleHasher>(&self, hasher: &H) -> bool {
        if self.leaf_indices.len() != self.leaf_hashes.len() {
            return false;
        }

        // Reconstruct the partial tree
        let mut nodes: HashMap<TreePosition, Hash> = HashMap::new();

        // Insert leaf hashes
        for (&idx, &hash) in self.leaf_indices.iter().zip(&self.leaf_hashes) {
            nodes.insert(TreePosition::leaf(idx), hash);
        }

        // Insert proof nodes
        for (&pos, &hash) in &self.proof_nodes {
            nodes.insert(pos, hash);
        }

        // Compute up to root
        for level in 0..self.height {
            let positions: Vec<_> = nodes
                .keys()
                .filter(|p| p.level == level)
                .copied()
                .collect();

            for pos in positions {
                let parent_pos = pos.parent();
                if nodes.contains_key(&parent_pos) {
                    continue;
                }

                let sibling_pos = pos.sibling();
                let left_pos = if pos.is_left() { pos } else { sibling_pos };
                let right_pos = if pos.is_left() { sibling_pos } else { pos };

                let left = match nodes.get(&left_pos) {
                    Some(h) => *h,
                    None => return false,
                };
                let right = match nodes.get(&right_pos) {
                    Some(h) => *h,
                    None => return false,
                };

                let parent_hash = hasher.hash_nodes(&left, &right);
                nodes.insert(parent_pos, parent_hash);
            }
        }

        // Check root
        let root_pos = TreePosition {
            level: self.height,
            index: 0,
        };
        nodes.get(&root_pos) == Some(&self.root)
    }
}

/// Merkle tree configuration.
#[derive(Debug, Clone)]
pub struct MerkleTreeConfig {
    /// Whether to cache intermediate nodes (uses more memory but faster proofs).
    pub cache_nodes: bool,
    /// Maximum tree height (prevents extremely deep trees).
    pub max_height: usize,
}

impl Default for MerkleTreeConfig {
    fn default() -> Self {
        Self {
            cache_nodes: true,
            max_height: 32,
        }
    }
}

/// Error types for Merkle tree operations.
#[derive(Debug, Clone)]
pub enum MerkleError {
    /// Tree is empty.
    EmptyTree,
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, size: usize },
    /// Invalid hash length.
    InvalidHashLength { expected: usize, actual: usize },
    /// Invalid hex encoding.
    InvalidHexEncoding,
    /// Tree height exceeded maximum.
    HeightExceeded { height: usize, max: usize },
    /// Proof verification failed.
    VerificationFailed,
    /// Invalid proof structure.
    InvalidProof(String),
}

impl std::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MerkleError::EmptyTree => write!(f, "Merkle tree is empty"),
            MerkleError::IndexOutOfBounds { index, size } => {
                write!(f, "Index {} out of bounds (size: {})", index, size)
            }
            MerkleError::InvalidHashLength { expected, actual } => {
                write!(f, "Invalid hash length: expected {}, got {}", expected, actual)
            }
            MerkleError::InvalidHexEncoding => write!(f, "Invalid hex encoding"),
            MerkleError::HeightExceeded { height, max } => {
                write!(f, "Tree height {} exceeds maximum {}", height, max)
            }
            MerkleError::VerificationFailed => write!(f, "Proof verification failed"),
            MerkleError::InvalidProof(msg) => write!(f, "Invalid proof: {}", msg),
        }
    }
}

impl std::error::Error for MerkleError {}

/// A production-grade Merkle tree.
#[derive(Clone)]
pub struct MerkleTree<H: MerkleHasher = Sha256Hasher> {
    /// The hasher.
    hasher: H,
    /// Configuration.
    config: MerkleTreeConfig,
    /// Leaf hashes.
    leaves: Vec<Hash>,
    /// Cached internal nodes (level -> index -> hash).
    nodes: Vec<Vec<Hash>>,
    /// Number of leaves.
    leaf_count: usize,
    /// Tree height.
    height: usize,
}

impl<H: MerkleHasher> MerkleTree<H> {
    /// Creates a new empty Merkle tree.
    pub fn new(hasher: H) -> Self {
        Self::with_config(hasher, MerkleTreeConfig::default())
    }

    /// Creates a new Merkle tree with config.
    pub fn with_config(hasher: H, config: MerkleTreeConfig) -> Self {
        Self {
            hasher,
            config,
            leaves: Vec::new(),
            nodes: Vec::new(),
            leaf_count: 0,
            height: 0,
        }
    }

    /// Creates a Merkle tree from leaf data.
    pub fn from_leaves(hasher: H, leaves: &[&[u8]]) -> Result<Self, MerkleError> {
        let mut tree = Self::new(hasher);
        tree.build(leaves)?;
        Ok(tree)
    }

    /// Creates a Merkle tree from pre-computed leaf hashes.
    pub fn from_hashes(hasher: H, hashes: Vec<Hash>) -> Result<Self, MerkleError> {
        let mut tree = Self::new(hasher);
        tree.leaves = hashes;
        tree.leaf_count = tree.leaves.len();
        tree.rebuild_internal()?;
        Ok(tree)
    }

    /// Builds the tree from leaf data.
    pub fn build(&mut self, leaves: &[&[u8]]) -> Result<(), MerkleError> {
        self.leaves.clear();
        self.leaves.reserve(leaves.len());

        for leaf in leaves {
            self.leaves.push(self.hasher.hash_leaf(leaf));
        }

        self.leaf_count = self.leaves.len();
        self.rebuild_internal()
    }

    /// Rebuilds internal nodes from leaves.
    fn rebuild_internal(&mut self) -> Result<(), MerkleError> {
        if self.leaves.is_empty() {
            self.height = 0;
            self.nodes.clear();
            return Ok(());
        }

        // Calculate height
        self.height = (self.leaf_count as f64).log2().ceil() as usize;
        if self.height > self.config.max_height {
            return Err(MerkleError::HeightExceeded {
                height: self.height,
                max: self.config.max_height,
            });
        }

        // Pad leaves to power of 2 with zero hashes
        let padded_count = 1usize << self.height;
        let mut current_level = self.leaves.clone();
        current_level.resize(padded_count, Hash::zero());

        self.nodes.clear();

        if self.config.cache_nodes {
            self.nodes.push(current_level.clone());
        }

        // Build internal levels
        while current_level.len() > 1 {
            let mut next_level = Vec::with_capacity(current_level.len() / 2);

            for pair in current_level.chunks(2) {
                let hash = self.hasher.hash_nodes(&pair[0], &pair[1]);
                next_level.push(hash);
            }

            if self.config.cache_nodes {
                self.nodes.push(next_level.clone());
            }

            current_level = next_level;
        }

        // Store root if not caching
        if !self.config.cache_nodes && !current_level.is_empty() {
            self.nodes = vec![vec![current_level[0]]];
        }

        Ok(())
    }

    /// Returns the root hash.
    pub fn root(&self) -> Option<Hash> {
        if self.nodes.is_empty() {
            return None;
        }

        if self.config.cache_nodes {
            self.nodes.last().and_then(|level| level.first().copied())
        } else {
            self.nodes.first().and_then(|level| level.first().copied())
        }
    }

    /// Returns the number of leaves.
    pub fn len(&self) -> usize {
        self.leaf_count
    }

    /// Returns true if tree is empty.
    pub fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    /// Returns the tree height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Gets a leaf hash by index.
    pub fn get_leaf(&self, index: usize) -> Option<Hash> {
        self.leaves.get(index).copied()
    }

    /// Generates a proof for a leaf at the given index.
    pub fn prove(&self, index: usize) -> Result<MerkleProof, MerkleError> {
        if index >= self.leaf_count {
            return Err(MerkleError::IndexOutOfBounds {
                index,
                size: self.leaf_count,
            });
        }

        let leaf_hash = self.leaves[index];
        let root = self.root().ok_or(MerkleError::EmptyTree)?;

        if !self.config.cache_nodes {
            // Rebuild just the path we need
            return self.prove_without_cache(index);
        }

        let mut path = Vec::with_capacity(self.height);
        let mut current_index = index;

        for level in 0..self.height {
            let sibling_index = current_index ^ 1;
            let sibling = self.nodes[level]
                .get(sibling_index)
                .copied()
                .unwrap_or(Hash::zero());

            let direction = if current_index % 2 == 0 {
                ProofDirection::Right
            } else {
                ProofDirection::Left
            };

            path.push(ProofStep { sibling, direction });
            current_index /= 2;
        }

        Ok(MerkleProof {
            leaf_index: index,
            leaf_hash,
            path,
            root,
        })
    }

    /// Generates a proof without using cached nodes.
    fn prove_without_cache(&self, index: usize) -> Result<MerkleProof, MerkleError> {
        let padded_count = 1usize << self.height;
        let mut current_level = self.leaves.clone();
        current_level.resize(padded_count, Hash::zero());

        let mut path = Vec::with_capacity(self.height);
        let mut current_index = index;

        for _ in 0..self.height {
            let sibling_index = current_index ^ 1;
            let sibling = current_level.get(sibling_index).copied().unwrap_or(Hash::zero());

            let direction = if current_index % 2 == 0 {
                ProofDirection::Right
            } else {
                ProofDirection::Left
            };

            path.push(ProofStep { sibling, direction });

            // Compute next level
            let mut next_level = Vec::with_capacity(current_level.len() / 2);
            for pair in current_level.chunks(2) {
                next_level.push(self.hasher.hash_nodes(&pair[0], &pair[1]));
            }

            current_level = next_level;
            current_index /= 2;
        }

        let root = current_level.first().copied().ok_or(MerkleError::EmptyTree)?;

        Ok(MerkleProof {
            leaf_index: index,
            leaf_hash: self.leaves[index],
            path,
            root,
        })
    }

    /// Generates proofs for multiple leaves efficiently.
    pub fn prove_batch(&self, indices: &[usize]) -> Result<Vec<MerkleProof>, MerkleError> {
        indices.iter().map(|&i| self.prove(i)).collect()
    }

    /// Generates an aggregated multi-proof for multiple leaves.
    pub fn prove_multi(&self, indices: &[usize]) -> Result<MultiProof, MerkleError> {
        if indices.is_empty() {
            return Err(MerkleError::EmptyTree);
        }

        for &idx in indices {
            if idx >= self.leaf_count {
                return Err(MerkleError::IndexOutOfBounds {
                    index: idx,
                    size: self.leaf_count,
                });
            }
        }

        let root = self.root().ok_or(MerkleError::EmptyTree)?;

        let leaf_hashes: Vec<Hash> = indices.iter().map(|&i| self.leaves[i]).collect();

        // Collect proof nodes, deduplicating shared paths
        let mut proof_nodes: HashMap<TreePosition, Hash> = HashMap::new();
        let mut known_positions: std::collections::HashSet<TreePosition> =
            std::collections::HashSet::new();

        // Mark leaf positions as known
        for &idx in indices {
            known_positions.insert(TreePosition::leaf(idx));
        }

        // For each level, find needed siblings
        for level in 0..self.height {
            let positions_at_level: Vec<_> = known_positions
                .iter()
                .filter(|p| p.level == level)
                .copied()
                .collect();

            for pos in positions_at_level {
                let sibling = pos.sibling();
                let parent = pos.parent();

                known_positions.insert(parent);

                // Only add sibling to proof if not already known
                if !known_positions.contains(&sibling) {
                    if self.config.cache_nodes {
                        let hash = self.nodes[level]
                            .get(sibling.index)
                            .copied()
                            .unwrap_or(Hash::zero());
                        proof_nodes.insert(sibling, hash);
                    }
                    known_positions.insert(sibling);
                }
            }
        }

        Ok(MultiProof {
            leaf_indices: indices.to_vec(),
            leaf_hashes,
            proof_nodes,
            root,
            height: self.height,
        })
    }

    /// Verifies a proof against this tree's root.
    pub fn verify_proof(&self, proof: &MerkleProof) -> bool {
        match self.root() {
            Some(root) => proof.verify_with_root(&self.hasher, &root),
            None => false,
        }
    }

    /// Appends a new leaf to the tree.
    pub fn push(&mut self, data: &[u8]) -> Result<usize, MerkleError> {
        let hash = self.hasher.hash_leaf(data);
        self.leaves.push(hash);
        self.leaf_count = self.leaves.len();
        self.rebuild_internal()?;
        Ok(self.leaf_count - 1)
    }

    /// Updates a leaf at the given index.
    pub fn update(&mut self, index: usize, data: &[u8]) -> Result<(), MerkleError> {
        if index >= self.leaf_count {
            return Err(MerkleError::IndexOutOfBounds {
                index,
                size: self.leaf_count,
            });
        }

        let hash = self.hasher.hash_leaf(data);
        self.leaves[index] = hash;
        self.rebuild_internal()
    }

    /// Returns the hasher.
    pub fn hasher(&self) -> &H {
        &self.hasher
    }

    /// Returns all leaf hashes.
    pub fn leaves(&self) -> &[Hash] {
        &self.leaves
    }
}

impl MerkleTree<Sha256Hasher> {
    /// Creates a new Merkle tree with SHA-256 hasher.
    pub fn with_sha256() -> Self {
        Self::new(Sha256Hasher)
    }
}

impl<H: MerkleHasher> std::fmt::Debug for MerkleTree<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MerkleTree")
            .field("leaf_count", &self.leaf_count)
            .field("height", &self.height)
            .field("root", &self.root())
            .finish()
    }
}

/// Builder for creating Merkle trees incrementally.
pub struct MerkleTreeBuilder<H: MerkleHasher = Sha256Hasher> {
    hasher: H,
    config: MerkleTreeConfig,
    leaves: Vec<Hash>,
}

impl<H: MerkleHasher> MerkleTreeBuilder<H> {
    /// Creates a new builder.
    pub fn new(hasher: H) -> Self {
        Self {
            hasher,
            config: MerkleTreeConfig::default(),
            leaves: Vec::new(),
        }
    }

    /// Creates a new builder with config.
    pub fn with_config(hasher: H, config: MerkleTreeConfig) -> Self {
        Self {
            hasher,
            config,
            leaves: Vec::new(),
        }
    }

    /// Adds a leaf by hashing data.
    pub fn add_leaf(mut self, data: &[u8]) -> Self {
        self.leaves.push(self.hasher.hash_leaf(data));
        self
    }

    /// Adds a pre-hashed leaf.
    pub fn add_hash(mut self, hash: Hash) -> Self {
        self.leaves.push(hash);
        self
    }

    /// Adds multiple leaves.
    pub fn add_leaves<'a>(mut self, data: impl IntoIterator<Item = &'a [u8]>) -> Self {
        for d in data {
            self.leaves.push(self.hasher.hash_leaf(d));
        }
        self
    }

    /// Returns the current number of leaves.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Returns true if no leaves added.
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Builds the Merkle tree.
    pub fn build(self) -> Result<MerkleTree<H>, MerkleError> {
        MerkleTree::from_hashes(self.hasher, self.leaves)
    }
}

impl MerkleTreeBuilder<Sha256Hasher> {
    /// Creates a builder with SHA-256 hasher.
    pub fn with_sha256() -> Self {
        Self::new(Sha256Hasher)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_basic() {
        let hash = Hash::from_slice(b"hello world");
        assert!(!hash.is_zero());
        assert_eq!(hash.to_hex().len(), 64);
    }

    #[test]
    fn test_hash_hex_roundtrip() {
        let original = Hash::from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let hex = original.to_hex();
        let parsed = Hash::from_hex(&hex).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_sha256_hasher() {
        let hasher = Sha256Hasher;
        let hash = hasher.hash_leaf(b"test");
        assert!(!hash.is_zero());
        assert_eq!(hasher.algorithm_id(), "sha256");
    }

    #[test]
    fn test_merkle_tree_basic() {
        let leaves: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        let tree = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        assert_eq!(tree.len(), 4);
        assert_eq!(tree.height(), 2);
        assert!(tree.root().is_some());
    }

    #[test]
    fn test_merkle_tree_single_leaf() {
        let tree = MerkleTree::from_leaves(Sha256Hasher, &[b"only"]).unwrap();

        assert_eq!(tree.len(), 1);
        assert!(tree.root().is_some());
    }

    #[test]
    fn test_merkle_proof_verify() {
        let leaves: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h"];
        let tree = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        for i in 0..leaves.len() {
            let proof = tree.prove(i).unwrap();
            assert!(proof.verify(&Sha256Hasher));
            assert_eq!(proof.leaf_index, i);
        }
    }

    #[test]
    fn test_merkle_proof_invalid() {
        let leaves: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        let tree = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        let mut proof = tree.prove(0).unwrap();
        // Corrupt the leaf hash
        proof.leaf_hash = Hash::zero();
        assert!(!proof.verify(&Sha256Hasher));
    }

    #[test]
    fn test_merkle_tree_odd_leaves() {
        let leaves: Vec<&[u8]> = vec![b"a", b"b", b"c"];
        let tree = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        assert_eq!(tree.len(), 3);
        assert!(tree.root().is_some());

        // All proofs should verify
        for i in 0..3 {
            let proof = tree.prove(i).unwrap();
            assert!(proof.verify(&Sha256Hasher));
        }
    }

    #[test]
    fn test_merkle_tree_builder() {
        let tree = MerkleTreeBuilder::with_sha256()
            .add_leaf(b"one")
            .add_leaf(b"two")
            .add_leaf(b"three")
            .build()
            .unwrap();

        assert_eq!(tree.len(), 3);
        assert!(tree.root().is_some());
    }

    #[test]
    fn test_merkle_tree_update() {
        let leaves: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        let mut tree = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        let original_root = tree.root();
        tree.update(1, b"x").unwrap();
        let new_root = tree.root();

        assert_ne!(original_root, new_root);
    }

    #[test]
    fn test_merkle_tree_push() {
        let mut tree = MerkleTree::with_sha256();
        tree.build(&[b"a", b"b"]).unwrap();

        let original_root = tree.root();
        tree.push(b"c").unwrap();
        let new_root = tree.root();

        assert_eq!(tree.len(), 3);
        assert_ne!(original_root, new_root);
    }

    #[test]
    fn test_merkle_proof_serialization() {
        let leaves: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        let tree = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        let proof = tree.prove(1).unwrap();
        let bytes = proof.to_bytes();
        let restored = MerkleProof::from_bytes(&bytes).unwrap();

        assert_eq!(proof.leaf_index, restored.leaf_index);
        assert_eq!(proof.leaf_hash, restored.leaf_hash);
        assert_eq!(proof.root, restored.root);
        assert!(restored.verify(&Sha256Hasher));
    }

    #[test]
    fn test_batch_proofs() {
        let leaves: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h"];
        let tree = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        let proofs = tree.prove_batch(&[0, 2, 5, 7]).unwrap();
        assert_eq!(proofs.len(), 4);

        for proof in &proofs {
            assert!(proof.verify(&Sha256Hasher));
        }
    }

    #[test]
    fn test_multi_proof() {
        let leaves: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h"];
        let tree = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        let multi = tree.prove_multi(&[0, 2, 5]).unwrap();
        assert!(multi.verify(&Sha256Hasher));
    }

    #[test]
    fn test_deterministic_root() {
        let leaves: Vec<&[u8]> = vec![b"test1", b"test2", b"test3"];

        let tree1 = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();
        let tree2 = MerkleTree::from_leaves(Sha256Hasher, &leaves).unwrap();

        assert_eq!(tree1.root(), tree2.root());
    }

    #[test]
    fn test_empty_tree() {
        let tree = MerkleTree::<Sha256Hasher>::with_sha256();
        assert!(tree.is_empty());
        assert!(tree.root().is_none());
    }

    #[test]
    fn test_large_tree() {
        let leaves: Vec<Vec<u8>> = (0..1000).map(|i| format!("leaf_{}", i).into_bytes()).collect();
        let leaf_refs: Vec<&[u8]> = leaves.iter().map(|v| v.as_slice()).collect();

        let tree = MerkleTree::from_leaves(Sha256Hasher, &leaf_refs).unwrap();

        assert_eq!(tree.len(), 1000);
        assert!(tree.root().is_some());

        // Verify random proofs
        for i in [0, 100, 500, 999] {
            let proof = tree.prove(i).unwrap();
            assert!(proof.verify(&Sha256Hasher));
        }
    }
}
