//! Native support for external agents speaking the Agent Client Protocol
//! (ACP, <https://agentclientprotocol.com>). The protocol client lives in
//! `crates/acp`; this module hosts the app-side integration: agent
//! configuration, connection lifecycle, and the bridge onto Warp's agent
//! conversation UI.

pub mod agent_manager;
pub mod agents_config;
pub mod conversation;
pub mod delegate;
