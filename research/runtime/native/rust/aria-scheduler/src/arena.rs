use crate::error::AriaError;

/// A bump-allocator arena backed by a single contiguous `Vec<u8>`.
///
/// Allocations are cache-line aligned (default 64 bytes) and the arena can be
/// `reset()` without freeing the backing buffer, making it ideal for
/// per-inference scratch memory.
pub struct Arena {
    buffer: Vec<u8>,
    alignment: usize,
    offset: usize,
}

impl Arena {
    /// Create a new arena with the given capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            alignment: 64,
            offset: 0,
        }
    }

    /// Create a new arena with a custom alignment (must be a power of two).
    pub fn with_alignment(capacity: usize, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "alignment must be a power of two");
        Self {
            buffer: vec![0u8; capacity],
            alignment,
            offset: 0,
        }
    }

    /// Allocate a contiguous, aligned slice of `count` f32 values from the arena.
    ///
    /// Returns a mutable slice into the arena's buffer. The returned memory is
    /// zeroed.
    pub fn alloc_f32(&mut self, count: usize) -> Result<&mut [f32], AriaError> {
        let byte_count = count * std::mem::size_of::<f32>();

        // Round offset up to the required alignment.
        let aligned_offset = (self.offset + self.alignment - 1) & !(self.alignment - 1);
        let end = aligned_offset + byte_count;

        if end > self.buffer.len() {
            return Err(AriaError::ArenaOOM {
                requested: byte_count,
                available: self.buffer.len().saturating_sub(aligned_offset),
            });
        }

        self.offset = end;

        // Zero the allocated region.
        self.buffer[aligned_offset..end].fill(0);

        // Safety: the slice is properly aligned to `alignment` (>= 4 for f32),
        // the lifetime is tied to `&mut self`, and no other reference can alias
        // because we advanced `self.offset` past this region.
        let ptr = self.buffer[aligned_offset..end].as_mut_ptr() as *mut f32;
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, count) })
    }

    /// Allocate a contiguous, aligned region for `count` f32 values and return
    /// a raw pointer and count.  Unlike `alloc_f32`, this does NOT borrow
    /// `&mut self` beyond this call, so callers can hold multiple pointers
    /// into the arena simultaneously.
    ///
    /// # Safety
    /// The returned pointer is valid until the arena is reset or dropped.
    /// The caller must ensure no two mutable references to the same region
    /// exist at the same time, and must not use the pointer after `reset()`.
    pub fn alloc_f32_raw(&mut self, count: usize) -> Result<(*mut f32, usize), AriaError> {
        let byte_count = count * std::mem::size_of::<f32>();
        let aligned_offset = (self.offset + self.alignment - 1) & !(self.alignment - 1);
        let end = aligned_offset + byte_count;

        if end > self.buffer.len() {
            return Err(AriaError::ArenaOOM {
                requested: byte_count,
                available: self.buffer.len().saturating_sub(aligned_offset),
            });
        }

        self.offset = end;
        self.buffer[aligned_offset..end].fill(0);
        let ptr = self.buffer[aligned_offset..end].as_mut_ptr() as *mut f32;
        Ok((ptr, count))
    }

    /// Reset the arena, allowing all previously allocated memory to be reused.
    /// Does not free or reallocate the backing buffer.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// The high-water mark: the maximum number of bytes ever allocated
    /// (approximated by the current offset since we never shrink).
    pub fn peak_bytes(&self) -> usize {
        self.offset
    }

    /// The number of bytes currently in use.
    pub fn used_bytes(&self) -> usize {
        self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_write() {
        let mut arena = Arena::new(4096);
        let slice = arena.alloc_f32(4).unwrap();
        assert_eq!(slice.len(), 4);
        slice[0] = 1.0;
        slice[3] = 42.0;
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[3], 42.0);
    }

    #[test]
    fn test_reset() {
        let mut arena = Arena::new(256);
        let _ = arena.alloc_f32(8).unwrap();
        assert!(arena.used_bytes() > 0);
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        // Can allocate again after reset.
        let _ = arena.alloc_f32(8).unwrap();
    }

    #[test]
    fn test_oom() {
        let mut arena = Arena::new(64);
        // 64 bytes of alignment padding + 1024 floats = way too much.
        let result = arena.alloc_f32(1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_peak_bytes() {
        let mut arena = Arena::new(4096);
        let _ = arena.alloc_f32(16).unwrap(); // 64 bytes
        let peak = arena.peak_bytes();
        assert!(peak >= 64);
    }
}
