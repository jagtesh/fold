//! Backend-agnostic state for one ACP agent conversation.
//!
//! [`AcpConversation`] owns the agent connection and folds the
//! `session/update` stream into an ordered list of [`AcpConversationItem`]s
//! — the UI-independent shape the conversation view (or the blocklist
//! bridge) renders from. All protocol I/O happens on the background
//! executor; item mutations happen on the model thread via events.

use std::path::PathBuf;
use std::sync::Arc;

use acp::{schema, AcpAgentConnection, ProtocolVersion};
use anyhow::anyhow;
use command::r#async::Command;
use warpui::{Entity, ModelContext};

use super::delegate::{AutoDenyPermissionResolver, LocalAcpDelegate, PermissionResolver};

/// Lifecycle of an ACP conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcpConversationStatus {
    /// Spawning the process and running `initialize` / `session/new`.
    Starting,
    /// Session established; no prompt turn in flight.
    Idle,
    /// The agent requires authentication before a session can be created.
    /// Holds the auth methods it advertised.
    NeedsAuthentication,
    /// A prompt turn is in flight.
    Running,
    /// The last turn ended (stop reason retained for the UI).
    Stopped(schema::StopReason),
    /// The connection or a turn failed.
    Failed(String),
}

/// One renderable unit of conversation content, folded from ACP
/// `session/update` notifications.
#[derive(Debug, Clone, PartialEq)]
pub enum AcpConversationItem {
    UserMessage { markdown: String },
    AgentMessage { markdown: String },
    AgentThought { markdown: String },
    ToolCall(schema::ToolCall),
    Plan(schema::Plan),
}

pub enum AcpConversationEvent {
    /// Status or items changed; re-render.
    Updated,
}

pub struct AcpConversation {
    connection: Option<Arc<AcpAgentConnection>>,
    status: AcpConversationStatus,
    session_id: Option<schema::SessionId>,
    auth_methods: Vec<schema::AuthMethod>,
    items: Vec<AcpConversationItem>,
}

impl AcpConversation {
    /// Spawns the agent process and establishes a session rooted at `cwd`.
    /// `permission_resolver` decides `session/request_permission` outcomes;
    /// pass the UI resolver once one exists, or [`AutoDenyPermissionResolver`].
    pub fn start(
        launch_command: Command,
        cwd: PathBuf,
        permission_resolver: Option<Arc<dyn PermissionResolver>>,
        ctx: &mut ModelContext<Self>,
    ) -> Self {
        let executor = ctx.background_executor();
        let resolver = permission_resolver.unwrap_or_else(|| Arc::new(AutoDenyPermissionResolver));
        let delegate = Arc::new(LocalAcpDelegate::new(executor.clone(), resolver));

        let mut conversation = Self {
            connection: None,
            status: AcpConversationStatus::Starting,
            session_id: None,
            auth_methods: Vec::new(),
            items: Vec::new(),
        };

        match AcpAgentConnection::spawn(launch_command, delegate, executor, None) {
            Ok(connection) => {
                let connection = Arc::new(connection);
                conversation.connection = Some(connection.clone());
                conversation.establish_session(connection, cwd, ctx);
            }
            Err(e) => {
                conversation.status = AcpConversationStatus::Failed(format!("{e:#}"));
            }
        }
        conversation
    }

    pub fn status(&self) -> &AcpConversationStatus {
        &self.status
    }

    pub fn items(&self) -> &[AcpConversationItem] {
        &self.items
    }

    pub fn auth_methods(&self) -> &[schema::AuthMethod] {
        &self.auth_methods
    }

    /// Sends a prompt turn with the user's markdown text.
    pub fn prompt(&mut self, text: String, ctx: &mut ModelContext<Self>) {
        let Some(connection) = self.connection.clone() else {
            return;
        };
        let Some(session_id) = self.session_id.clone() else {
            log::warn!("ACP: prompt before session established");
            return;
        };
        self.items.push(AcpConversationItem::UserMessage {
            markdown: text.clone(),
        });
        self.status = AcpConversationStatus::Running;
        ctx.emit(AcpConversationEvent::Updated);

        let request = schema::PromptRequest::new(
            session_id,
            vec![schema::ContentBlock::Text(schema::TextContent::new(text))],
        );
        ctx.spawn(
            async move { connection.prompt(request).await },
            |me, result, ctx| {
                match result {
                    Ok(response) => {
                        me.status = AcpConversationStatus::Stopped(response.stop_reason);
                    }
                    Err(e) => {
                        me.status = AcpConversationStatus::Failed(format!("{e:#}"));
                    }
                }
                ctx.emit(AcpConversationEvent::Updated);
            },
        );
    }

    /// Cancels the in-flight turn, if any.
    pub fn cancel(&self) {
        if let (Some(connection), Some(session_id)) = (&self.connection, &self.session_id) {
            if let Err(e) = connection.cancel(session_id.clone()) {
                log::warn!("ACP: cancel failed: {e:#}");
            }
        }
    }

    /// Shuts down the agent process.
    pub fn shutdown(&self, ctx: &mut ModelContext<Self>) {
        if let Some(connection) = self.connection.clone() {
            ctx.background_executor()
                .spawn(async move {
                    if let Err(e) = connection.shutdown().await {
                        log::warn!("ACP: shutdown failed: {e:#}");
                    }
                })
                .detach();
        }
    }

    /// Runs `initialize` + `session/new` and starts draining session updates.
    fn establish_session(
        &mut self,
        connection: Arc<AcpAgentConnection>,
        cwd: PathBuf,
        ctx: &mut ModelContext<Self>,
    ) {
        // Drain session updates into the model as they stream in.
        ctx.spawn_stream_local(
            connection.session_updates(),
            |me, notification, ctx| {
                me.apply_update(notification.update);
                ctx.emit(AcpConversationEvent::Updated);
            },
            |_, _| {},
        );

        let handshake = async move {
            let capabilities = schema::ClientCapabilities::new()
                .fs(schema::FileSystemCapabilities::new()
                    .read_text_file(true)
                    .write_text_file(true))
                .terminal(true);
            let initialize = connection
                .initialize(
                    schema::InitializeRequest::new(ProtocolVersion::V1)
                        .client_capabilities(capabilities),
                )
                .await?;

            match connection
                .new_session(schema::NewSessionRequest::new(cwd))
                .await
            {
                Ok(response) => Ok((initialize.auth_methods, Some(response.session_id))),
                // A session/new failure with advertised auth methods means
                // the agent wants authentication first.
                Err(e) if !initialize.auth_methods.is_empty() => {
                    log::info!("ACP: session/new failed, assuming auth required: {e:#}");
                    Ok((initialize.auth_methods, None))
                }
                Err(e) => Err(anyhow!(e)),
            }
        };
        ctx.spawn(handshake, |me, result, ctx| {
            match result {
                Ok((auth_methods, Some(session_id))) => {
                    me.auth_methods = auth_methods;
                    me.session_id = Some(session_id);
                    me.status = AcpConversationStatus::Idle;
                }
                Ok((auth_methods, None)) => {
                    me.auth_methods = auth_methods;
                    me.status = AcpConversationStatus::NeedsAuthentication;
                }
                Err(e) => {
                    me.status = AcpConversationStatus::Failed(format!("{e:#}"));
                }
            }
            ctx.emit(AcpConversationEvent::Updated);
        });
    }

    /// Folds one `session/update` into the item list.
    fn apply_update(&mut self, update: schema::SessionUpdate) {
        match update {
            schema::SessionUpdate::UserMessageChunk(chunk) => {
                // Echoed user content (e.g. on session/load replay).
                if let Some(text) = content_block_text(&chunk.content) {
                    match self.items.last_mut() {
                        Some(AcpConversationItem::UserMessage { markdown }) => {
                            markdown.push_str(text);
                        }
                        _ => self.items.push(AcpConversationItem::UserMessage {
                            markdown: text.to_string(),
                        }),
                    }
                }
            }
            schema::SessionUpdate::AgentMessageChunk(chunk) => {
                if let Some(text) = content_block_text(&chunk.content) {
                    match self.items.last_mut() {
                        Some(AcpConversationItem::AgentMessage { markdown }) => {
                            markdown.push_str(text);
                        }
                        _ => self.items.push(AcpConversationItem::AgentMessage {
                            markdown: text.to_string(),
                        }),
                    }
                }
            }
            schema::SessionUpdate::AgentThoughtChunk(chunk) => {
                if let Some(text) = content_block_text(&chunk.content) {
                    match self.items.last_mut() {
                        Some(AcpConversationItem::AgentThought { markdown }) => {
                            markdown.push_str(text);
                        }
                        _ => self.items.push(AcpConversationItem::AgentThought {
                            markdown: text.to_string(),
                        }),
                    }
                }
            }
            schema::SessionUpdate::ToolCall(tool_call) => {
                self.items.push(AcpConversationItem::ToolCall(tool_call));
            }
            schema::SessionUpdate::ToolCallUpdate(update) => {
                self.apply_tool_call_update(update);
            }
            schema::SessionUpdate::Plan(plan) => {
                // A plan update replaces the previous plan snapshot.
                if let Some(AcpConversationItem::Plan(existing)) = self
                    .items
                    .iter_mut()
                    .rev()
                    .find(|item| matches!(item, AcpConversationItem::Plan(_)))
                {
                    *existing = plan;
                } else {
                    self.items.push(AcpConversationItem::Plan(plan));
                }
            }
            // Not yet surfaced in the UI; ignore.
            schema::SessionUpdate::AvailableCommandsUpdate(_)
            | schema::SessionUpdate::CurrentModeUpdate(_)
            | schema::SessionUpdate::ConfigOptionUpdate(_)
            | schema::SessionUpdate::SessionInfoUpdate(_)
            | schema::SessionUpdate::UsageUpdate(_) => {}
            // SessionUpdate is #[non_exhaustive]; ignore unknown updates.
            _ => {}
        }
    }

    /// Merges a `tool_call_update` into the matching [`AcpConversationItem::ToolCall`].
    fn apply_tool_call_update(&mut self, update: schema::ToolCallUpdate) {
        let target = self.items.iter_mut().rev().find_map(|item| match item {
            AcpConversationItem::ToolCall(tool_call)
                if tool_call.tool_call_id == update.tool_call_id =>
            {
                Some(tool_call)
            }
            _ => None,
        });
        let Some(tool_call) = target else {
            log::debug!(
                "ACP: tool_call_update for unknown tool call {}",
                update.tool_call_id
            );
            return;
        };
        let fields = update.fields;
        if let Some(title) = fields.title {
            tool_call.title = title;
        }
        if let Some(kind) = fields.kind {
            tool_call.kind = kind;
        }
        if let Some(status) = fields.status {
            tool_call.status = status;
        }
        if let Some(content) = fields.content {
            tool_call.content = content;
        }
        if let Some(locations) = fields.locations {
            tool_call.locations = locations;
        }
        if let Some(raw_input) = fields.raw_input {
            tool_call.raw_input = Some(raw_input);
        }
        if let Some(raw_output) = fields.raw_output {
            tool_call.raw_output = Some(raw_output);
        }
    }
}

/// Extracts renderable markdown/plain text from a content block, if any.
fn content_block_text(content: &schema::ContentBlock) -> Option<&str> {
    match content {
        schema::ContentBlock::Text(text) => Some(&text.text),
        schema::ContentBlock::ResourceLink(link) => Some(&link.uri),
        schema::ContentBlock::Image(_)
        | schema::ContentBlock::Audio(_)
        | schema::ContentBlock::Resource(_) => None,
        _ => None,
    }
}

impl Entity for AcpConversation {
    type Event = AcpConversationEvent;
}

#[cfg(test)]
#[path = "conversation_tests.rs"]
mod tests;
