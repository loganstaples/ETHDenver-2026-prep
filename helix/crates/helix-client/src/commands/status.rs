//! Status command implementation

/// Status command handler - marker type for CLI
pub struct StatusCommand;

impl StatusCommand {
    /// Create a new status command
    pub fn new() -> Self {
        Self
    }
}

impl Default for StatusCommand {
    fn default() -> Self {
        Self::new()
    }
}
