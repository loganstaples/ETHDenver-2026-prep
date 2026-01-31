//! Init command implementation

/// Init command handler - marker type for CLI
pub struct InitCommand;

impl InitCommand {
    /// Create a new init command
    pub fn new() -> Self {
        Self
    }
}

impl Default for InitCommand {
    fn default() -> Self {
        Self::new()
    }
}
