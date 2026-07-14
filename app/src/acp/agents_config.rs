//! User configuration for external ACP agents.
//!
//! Agents are defined in `agents.json` in the Warp home config directory
//! (see [`warp_core::paths::warp_home_acp_agents_config_file_path`]), as a
//! map of agent name to launch definition:
//!
//! ```json
//! {
//!     "claude": { "command": "claude-code-acp" },
//!     "gemini": { "command": "gemini", "args": ["--experimental-acp"] }
//! }
//! ```
//!
//! Well-known agents are also offered as built-in presets so the picker is
//! useful before the user has written any config; a user-defined agent with
//! the same name overrides its preset.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

/// How to launch one external ACP agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcpAgentDefinition {
    /// The executable to run (resolved against `PATH` unless absolute).
    pub command: String,
    /// Arguments passed to the command.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Environment variables set on the agent process, on top of the
    /// session's environment.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,
    /// Working directory for the agent process. Defaults to the pane's
    /// working directory when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_directory: Option<String>,
}

impl AcpAgentDefinition {
    fn new(command: impl Into<String>, args: &[&str]) -> Self {
        Self {
            command: command.into(),
            args: args.iter().map(|arg| (*arg).to_string()).collect(),
            env: HashMap::new(),
            working_directory: None,
        }
    }
}

/// The set of configured ACP agents, keyed by display name. Ordered so the
/// picker is stable across reloads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AcpAgentsConfig {
    pub agents: BTreeMap<String, AcpAgentDefinition>,
}

impl AcpAgentsConfig {
    /// Parses an `agents.json` document: a JSON object mapping agent names
    /// to definitions.
    pub fn parse(json: &str) -> Result<Self> {
        let agents: BTreeMap<String, AcpAgentDefinition> =
            serde_json::from_str(json).context("Failed to parse agents.json")?;
        Ok(Self { agents })
    }

    /// Built-in launch definitions for well-known ACP agents. These appear in
    /// the picker without any user config; availability still depends on the
    /// binary being installed.
    pub fn builtin_presets() -> Self {
        let agents = BTreeMap::from([
            (
                "Claude Code".to_string(),
                AcpAgentDefinition::new("claude-code-acp", &[]),
            ),
            (
                "Gemini CLI".to_string(),
                AcpAgentDefinition::new("gemini", &["--experimental-acp"]),
            ),
        ]);
        Self { agents }
    }

    /// Returns the presets merged with the user's config; user-defined
    /// agents win on name collisions.
    pub fn with_presets(user: Self) -> Self {
        let mut merged = Self::builtin_presets();
        merged.agents.extend(user.agents);
        merged
    }
}

#[cfg(test)]
#[path = "agents_config_tests.rs"]
mod tests;
