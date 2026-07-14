//! Singleton model owning the set of configured ACP agents.
//!
//! Loads `agents.json` (merged over the built-in presets) and builds launch
//! commands for the picker. Connection lifecycle is owned by the
//! conversation that spawns an agent, not by this manager.

use std::io::ErrorKind;
use std::path::Path;

use anyhow::{anyhow, Result};
use command::r#async::Command;
use warp_core::paths::warp_home_acp_agents_config_file_path;
use warpui::{Entity, ModelContext, SingletonEntity};

use super::agents_config::{AcpAgentDefinition, AcpAgentsConfig};

pub enum AcpAgentManagerEvent {
    /// The agent list changed (config file loaded or reloaded).
    ConfigReloaded,
}

pub struct AcpAgentManager {
    /// Presets merged with the user's `agents.json` definitions.
    config: AcpAgentsConfig,
}

impl AcpAgentManager {
    pub fn new(ctx: &mut ModelContext<Self>) -> Self {
        let mut manager = Self {
            config: AcpAgentsConfig::builtin_presets(),
        };
        manager.reload(ctx);
        manager
    }

    /// The configured agents, keyed by display name.
    pub fn agents(&self) -> &AcpAgentsConfig {
        &self.config
    }

    /// Re-reads `agents.json` off the UI thread and applies the result.
    /// A missing file leaves just the presets; a malformed file is logged
    /// and ignored so a typo can't empty the picker.
    pub fn reload(&mut self, ctx: &mut ModelContext<Self>) {
        let Some(path) = warp_home_acp_agents_config_file_path() else {
            return;
        };
        ctx.spawn(
            async move {
                match async_fs::read_to_string(&path).await {
                    Ok(contents) => match AcpAgentsConfig::parse(&contents) {
                        Ok(config) => Some(config),
                        Err(e) => {
                            log::warn!("ACP: ignoring malformed {}: {e:#}", path.display());
                            None
                        }
                    },
                    Err(e) if e.kind() == ErrorKind::NotFound => Some(AcpAgentsConfig::default()),
                    Err(e) => {
                        log::warn!("ACP: failed to read {}: {e}", path.display());
                        None
                    }
                }
            },
            |me, parsed, ctx| {
                if let Some(user_config) = parsed {
                    me.config = AcpAgentsConfig::with_presets(user_config);
                    ctx.emit(AcpAgentManagerEvent::ConfigReloaded);
                }
            },
        );
    }

    /// Builds the launch command for a configured agent. `cwd` is used when
    /// the definition doesn't pin its own working directory.
    pub fn launch_command(&self, agent_name: &str, cwd: Option<&Path>) -> Result<Command> {
        let definition = self
            .config
            .agents
            .get(agent_name)
            .ok_or_else(|| anyhow!("Unknown ACP agent: {agent_name}"))?;
        Ok(build_launch_command(definition, cwd))
    }
}

fn build_launch_command(definition: &AcpAgentDefinition, cwd: Option<&Path>) -> Command {
    let mut command = Command::new(&definition.command);
    command.args(&definition.args);
    command.envs(&definition.env);
    match (&definition.working_directory, cwd) {
        (Some(dir), _) => {
            command.current_dir(dir);
        }
        (None, Some(dir)) => {
            command.current_dir(dir);
        }
        (None, None) => {}
    }
    command
}

impl Entity for AcpAgentManager {
    type Event = AcpAgentManagerEvent;
}

impl SingletonEntity for AcpAgentManager {}
