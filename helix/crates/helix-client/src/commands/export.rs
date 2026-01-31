//! Export command implementation

/// Export command handler - marker type for CLI
pub struct ExportCommand;

impl ExportCommand {
    /// Create a new export command
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExportCommand {
    fn default() -> Self {
        Self::new()
    }
}
