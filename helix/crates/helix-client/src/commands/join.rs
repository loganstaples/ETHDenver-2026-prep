//! Join command implementation

/// Join command handler - marker type for CLI
pub struct JoinCommand;

impl JoinCommand {
    /// Create a new join command
    pub fn new() -> Self {
        Self
    }
}

impl Default for JoinCommand {
    fn default() -> Self {
        Self::new()
    }
}
