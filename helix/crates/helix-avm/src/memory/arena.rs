//! Tensor Memory Arena for allocation pooling.
//!
//! Provides a typed arena for `Vec<BoundedValue<f64>>` to avoid per-tensor
//! system allocator hits on hot paths. During a forward/backward pass (~50
//! tensor allocations), reusing pre-allocated chunks significantly reduces
//! allocation overhead.
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::memory::arena::TensorArena;
//!
//! let mut arena = TensorArena::new(1024);
//!
//! // Allocate from the arena
//! let slice = arena.allocate(256);
//! // ... use the slice ...
//!
//! // Reset reclaims all allocations at once
//! arena.reset();
//! ```

use helix_core::types::BoundedValue;

/// A handle into the arena's storage.
#[derive(Debug, Clone, Copy)]
pub struct ArenaSlice {
    /// Index of the chunk in the arena.
    pub chunk_idx: usize,
    /// Start offset within the chunk.
    pub offset: usize,
    /// Number of elements in this slice.
    pub len: usize,
}

/// A typed arena allocator for tensor backing storage.
///
/// Pre-allocates chunks of `BoundedValue<f64>` and hands out slices from them.
/// All allocations can be reclaimed at once via `reset()`, making this ideal
/// for forward/backward pass scopes.
pub struct TensorArena {
    /// Pre-allocated chunks of storage.
    chunks: Vec<Vec<BoundedValue<f64>>>,
    /// Free list: (chunk_idx, offset, len) of returned slices.
    free_list: Vec<(usize, usize, usize)>,
    /// Default chunk size (number of BoundedValue<f64> elements).
    chunk_size: usize,
    /// Current write position in the active chunk.
    current_offset: usize,
    /// Total elements allocated (for stats).
    total_allocated: usize,
    /// Total elements capacity across all chunks.
    total_capacity: usize,
}

impl TensorArena {
    /// Creates a new arena with the given chunk size.
    ///
    /// Each chunk holds `chunk_size` `BoundedValue<f64>` elements.
    /// A good default is 4096 (covers most layer activations).
    pub fn new(chunk_size: usize) -> Self {
        let initial_chunk = vec![BoundedValue::<f64>::exact(0.0); chunk_size];
        Self {
            chunks: vec![initial_chunk],
            free_list: Vec::new(),
            chunk_size,
            current_offset: 0,
            total_allocated: 0,
            total_capacity: chunk_size,
        }
    }

    /// Allocates a slice of `size` elements from the arena.
    ///
    /// Returns an `ArenaSlice` handle that can be used with `get`/`get_mut`
    /// to access the underlying storage.
    pub fn allocate(&mut self, size: usize) -> ArenaSlice {
        // Try the free list first
        if let Some(idx) = self.free_list.iter().position(|&(_, _, len)| len >= size) {
            let (chunk_idx, offset, len) = self.free_list.remove(idx);
            self.total_allocated += size;
            // If the free block is larger, return the remainder to the free list
            if len > size {
                self.free_list.push((chunk_idx, offset + size, len - size));
            }
            return ArenaSlice {
                chunk_idx,
                offset,
                len: size,
            };
        }

        // Try to allocate from the current chunk
        let current_chunk = self.chunks.len() - 1;
        let chunk_len = self.chunks[current_chunk].len();

        if self.current_offset + size <= chunk_len {
            let slice = ArenaSlice {
                chunk_idx: current_chunk,
                offset: self.current_offset,
                len: size,
            };
            self.current_offset += size;
            self.total_allocated += size;
            return slice;
        }

        // Need a new chunk — make it at least chunk_size or the requested size
        let new_chunk_size = self.chunk_size.max(size);
        let new_chunk = vec![BoundedValue::<f64>::exact(0.0); new_chunk_size];
        self.chunks.push(new_chunk);
        self.total_capacity += new_chunk_size;

        let chunk_idx = self.chunks.len() - 1;
        self.current_offset = size;
        self.total_allocated += size;

        ArenaSlice {
            chunk_idx,
            offset: 0,
            len: size,
        }
    }

    /// Returns a slice to the free list for reuse.
    pub fn deallocate(&mut self, slice: ArenaSlice) {
        self.total_allocated = self.total_allocated.saturating_sub(slice.len);
        self.free_list
            .push((slice.chunk_idx, slice.offset, slice.len));
    }

    /// Gets an immutable reference to the data backing an `ArenaSlice`.
    pub fn get(&self, slice: &ArenaSlice) -> &[BoundedValue<f64>] {
        &self.chunks[slice.chunk_idx][slice.offset..slice.offset + slice.len]
    }

    /// Gets a mutable reference to the data backing an `ArenaSlice`.
    pub fn get_mut(&mut self, slice: &ArenaSlice) -> &mut [BoundedValue<f64>] {
        &mut self.chunks[slice.chunk_idx][slice.offset..slice.offset + slice.len]
    }

    /// Resets the arena, reclaiming all allocations at once.
    ///
    /// This does not free memory — it simply resets the write pointer and
    /// clears the free list. All previously returned `ArenaSlice` handles
    /// become invalid after a reset.
    pub fn reset(&mut self) {
        self.free_list.clear();
        self.current_offset = 0;
        self.total_allocated = 0;
        // Keep only the first chunk to reduce fragmentation
        if self.chunks.len() > 1 {
            // Merge capacity into a single large chunk
            let total = self.total_capacity;
            self.chunks.clear();
            self.chunks
                .push(vec![BoundedValue::<f64>::exact(0.0); total]);
            self.total_capacity = total;
        }
    }

    /// Returns the total number of elements currently allocated.
    pub fn allocated(&self) -> usize {
        self.total_allocated
    }

    /// Returns the total capacity across all chunks.
    pub fn capacity(&self) -> usize {
        self.total_capacity
    }

    /// Returns the number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Returns memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        self.total_capacity * std::mem::size_of::<BoundedValue<f64>>()
    }
}

/// Scoped arena guard that resets the arena on drop.
///
/// Use `with_arena()` to create a scope where arena allocations are
/// automatically reclaimed when the scope exits.
pub struct ArenaGuard<'a> {
    arena: &'a mut TensorArena,
}

impl<'a> ArenaGuard<'a> {
    /// Allocates from the arena within this scope.
    pub fn allocate(&mut self, size: usize) -> ArenaSlice {
        self.arena.allocate(size)
    }

    /// Gets an immutable reference to arena data.
    pub fn get(&self, slice: &ArenaSlice) -> &[BoundedValue<f64>] {
        self.arena.get(slice)
    }

    /// Gets a mutable reference to arena data.
    pub fn get_mut(&mut self, slice: &ArenaSlice) -> &mut [BoundedValue<f64>] {
        self.arena.get_mut(slice)
    }
}

impl<'a> Drop for ArenaGuard<'a> {
    fn drop(&mut self) {
        self.arena.reset();
    }
}

/// Creates a scoped arena guard that auto-resets on drop.
///
/// # Example
///
/// ```ignore
/// let mut arena = TensorArena::new(4096);
/// {
///     let mut guard = with_arena(&mut arena);
///     let s = guard.allocate(128);
///     // ... use the allocation ...
/// } // arena is reset here
/// ```
pub fn with_arena(arena: &mut TensorArena) -> ArenaGuard<'_> {
    ArenaGuard { arena }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut arena = TensorArena::new(1024);
        let slice = arena.allocate(100);
        assert_eq!(slice.len, 100);
        assert_eq!(slice.chunk_idx, 0);
        assert_eq!(slice.offset, 0);
        assert_eq!(arena.allocated(), 100);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut arena = TensorArena::new(1024);
        let s1 = arena.allocate(100);
        let s2 = arena.allocate(200);
        let s3 = arena.allocate(300);

        assert_eq!(s1.offset, 0);
        assert_eq!(s2.offset, 100);
        assert_eq!(s3.offset, 300);
        assert_eq!(arena.allocated(), 600);
    }

    #[test]
    fn test_new_chunk_on_overflow() {
        let mut arena = TensorArena::new(100);
        let _s1 = arena.allocate(80);
        let s2 = arena.allocate(50); // Doesn't fit in current chunk

        assert_eq!(s2.chunk_idx, 1);
        assert_eq!(arena.num_chunks(), 2);
    }

    #[test]
    fn test_deallocate_and_reuse() {
        let mut arena = TensorArena::new(1024);
        let s1 = arena.allocate(100);
        arena.deallocate(s1);
        assert_eq!(arena.allocated(), 0);

        // Should reuse the freed space
        let s2 = arena.allocate(50);
        assert_eq!(s2.chunk_idx, s1.chunk_idx);
        assert_eq!(s2.offset, s1.offset);
    }

    #[test]
    fn test_reset() {
        let mut arena = TensorArena::new(1024);
        let _s1 = arena.allocate(100);
        let _s2 = arena.allocate(200);
        assert_eq!(arena.allocated(), 300);

        arena.reset();
        assert_eq!(arena.allocated(), 0);

        // Can allocate again from the start
        let s3 = arena.allocate(50);
        assert_eq!(s3.offset, 0);
    }

    #[test]
    fn test_get_and_get_mut() {
        let mut arena = TensorArena::new(1024);
        let slice = arena.allocate(10);

        // Write data
        let data = arena.get_mut(&slice);
        for (i, val) in data.iter_mut().enumerate() {
            *val = BoundedValue::exact(i as f64);
        }

        // Read back
        let data = arena.get(&slice);
        assert!((data[0].value() - 0.0).abs() < 1e-10);
        assert!((data[5].value() - 5.0).abs() < 1e-10);
        assert!((data[9].value() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_large_allocation() {
        let mut arena = TensorArena::new(100);
        // Request more than chunk_size
        let slice = arena.allocate(500);
        assert_eq!(slice.len, 500);
        assert!(arena.capacity() >= 500);
    }

    #[test]
    fn test_arena_guard_auto_reset() {
        let mut arena = TensorArena::new(1024);

        {
            let mut guard = with_arena(&mut arena);
            let _s = guard.allocate(100);
        } // Guard dropped, arena reset

        assert_eq!(arena.allocated(), 0);
    }

    #[test]
    fn test_memory_bytes() {
        let arena = TensorArena::new(1024);
        let expected = 1024 * std::mem::size_of::<BoundedValue<f64>>();
        assert_eq!(arena.memory_bytes(), expected);
    }
}
