use std::fmt;

/// Error types for the Aria scheduler, mapping to NR_ERR_* ABI codes.
#[derive(Debug)]
pub enum AriaError {
    /// The graph IR JSON is malformed or semantically invalid.
    InvalidIR(String),
    /// The graph contains a cycle and cannot be topologically sorted.
    CyclicGraph,
    /// An operation name has no registered kernel.
    UnsupportedOp(String),
    /// A kernel or execution step failed at runtime.
    ExecutionFailed(String),
    /// The arena allocator ran out of pre-allocated memory.
    ArenaOOM { requested: usize, available: usize },
    /// A raw ABI error code propagated from C/FFI.
    AbiError(i32),
}

impl fmt::Display for AriaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AriaError::InvalidIR(msg) => write!(f, "invalid IR: {}", msg),
            AriaError::CyclicGraph => write!(f, "graph contains a cycle"),
            AriaError::UnsupportedOp(op) => write!(f, "unsupported op: {}", op),
            AriaError::ExecutionFailed(msg) => write!(f, "execution failed: {}", msg),
            AriaError::ArenaOOM {
                requested,
                available,
            } => write!(
                f,
                "arena OOM: requested {} bytes but only {} available",
                requested, available
            ),
            AriaError::AbiError(code) => write!(f, "ABI error code {}", code),
        }
    }
}

impl std::error::Error for AriaError {}

impl AriaError {
    /// Map to the NR_ERR_* integer codes defined in runner_abi.h.
    ///
    /// Convention (mirrors the C header):
    ///   -1  generic / invalid IR
    ///   -2  cyclic graph
    ///   -3  unsupported op
    ///   -4  execution failure
    ///   -5  arena OOM
    ///   *   pass-through for AbiError
    pub fn to_abi_code(&self) -> i32 {
        match self {
            AriaError::InvalidIR(_) => -1,
            AriaError::CyclicGraph => -2,
            AriaError::UnsupportedOp(_) => -3,
            AriaError::ExecutionFailed(_) => -4,
            AriaError::ArenaOOM { .. } => -5,
            AriaError::AbiError(code) => *code,
        }
    }
}
