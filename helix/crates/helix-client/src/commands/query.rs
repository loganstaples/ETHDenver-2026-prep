//! Query command implementation

/// Query command handler - marker type for CLI
pub struct QueryCommand;

impl QueryCommand {
    /// Create a new query command
    pub fn new() -> Self {
        Self
    }
}

impl Default for QueryCommand {
    fn default() -> Self {
        Self::new()
    }
}
