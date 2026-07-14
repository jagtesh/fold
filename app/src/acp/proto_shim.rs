//! Maps ACP session events onto Warp's native multi-agent protobuf stream.
//!
//! Warp's conversation store is protobuf-native: everything downstream of a
//! stream of [`api::ResponseEvent`]s (history model, streaming markdown,
//! tool cards, persistence, restore) is backend-agnostic. This shim lets an
//! ACP agent drive that pipeline by synthesizing the same events the
//! multi-agent backend would produce:
//!
//! - turn start -> `StreamInit`
//! - `agent_message_chunk` -> `AddMessagesToTask` (first chunk) then
//!   `AppendToMessageContent` with a `message.agent_output.text` field mask
//! - `agent_thought_chunk` -> same, as `AgentReasoning`
//! - `tool_call` / `tool_call_update` -> a `Server`-tool `ToolCall` message
//!   carrying the ACP tool call as an opaque JSON payload (display-only:
//!   ACP tools are agent-executed, so the client must never try to run them)
//! - turn end -> `StreamFinished`
//!
//! The shim is pure state-in/events-out so it can be unit tested without
//! the controller; the ACP response-stream driver owns feeding its output
//! into the blocklist controller.

use acp::schema;
use prost_types::FieldMask;
use uuid::Uuid;
use warp_multi_agent_api as api;

/// Field-mask path for appending streamed agent text.
const AGENT_OUTPUT_TEXT_MASK: &str = "message.agent_output.text";
/// Field-mask path for appending streamed reasoning text.
const AGENT_REASONING_MASK: &str = "message.agent_reasoning.reasoning";
/// Field-mask path for replacing a tool call on update.
const TOOL_CALL_MASK: &str = "message.tool_call";

/// Which streaming message (if any) the shim is currently appending to.
enum StreamingTarget {
    None,
    Text { message_id: String },
    Reasoning { message_id: String },
}

/// Stateful translator from ACP `session/update`s to native
/// [`api::ResponseEvent`]s for one prompt turn.
pub struct AcpProtoShim {
    task_id: String,
    request_id: String,
    target: StreamingTarget,
}

impl AcpProtoShim {
    pub fn new(task_id: String, request_id: String) -> Self {
        Self {
            task_id,
            request_id,
            target: StreamingTarget::None,
        }
    }

    /// The `StreamInit` event that must open the synthesized stream.
    pub fn stream_init(&self, conversation_id: String) -> api::ResponseEvent {
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::Init(
                api::response_event::StreamInit {
                    conversation_id,
                    request_id: self.request_id.clone(),
                    run_id: String::new(),
                },
            )),
        }
    }

    /// Translates one ACP session update into client actions, if it has a
    /// native representation.
    pub fn map_update(&mut self, update: schema::SessionUpdate) -> Option<api::ResponseEvent> {
        let actions = match update {
            schema::SessionUpdate::AgentMessageChunk(chunk) => {
                let text = content_block_text(&chunk.content)?;
                Some(self.append_streaming_text(text, StreamingKind::Text))
            }
            schema::SessionUpdate::AgentThoughtChunk(chunk) => {
                let text = content_block_text(&chunk.content)?;
                Some(self.append_streaming_text(text, StreamingKind::Reasoning))
            }
            schema::SessionUpdate::ToolCall(tool_call) => {
                self.target = StreamingTarget::None;
                let message_id = tool_call.tool_call_id.to_string();
                Some(vec![add_messages_action(
                    &self.task_id,
                    vec![tool_call_message(
                        &self.task_id,
                        &self.request_id,
                        &message_id,
                        &tool_call_payload(&tool_call),
                    )],
                )])
            }
            schema::SessionUpdate::ToolCallUpdate(update) => {
                // Tool-call messages are keyed by the ACP tool_call_id, so an
                // update replaces the payload in place via field mask.
                let message_id = update.tool_call_id.to_string();
                let payload = serde_json::to_string(&update).unwrap_or_default();
                Some(vec![api::ClientAction {
                    action: Some(api::client_action::Action::UpdateTaskMessage(
                        api::client_action::UpdateTaskMessage {
                            task_id: self.task_id.clone(),
                            message: Some(tool_call_message(
                                &self.task_id,
                                &self.request_id,
                                &message_id,
                                &payload,
                            )),
                            mask: Some(FieldMask {
                                paths: vec![TOOL_CALL_MASK.to_string()],
                            }),
                        },
                    )),
                }])
            }
            // Plans, mode/config/info/usage updates and user echoes have no
            // native client-action mapping yet.
            schema::SessionUpdate::UserMessageChunk(_)
            | schema::SessionUpdate::Plan(_)
            | schema::SessionUpdate::AvailableCommandsUpdate(_)
            | schema::SessionUpdate::CurrentModeUpdate(_)
            | schema::SessionUpdate::ConfigOptionUpdate(_)
            | schema::SessionUpdate::SessionInfoUpdate(_)
            | schema::SessionUpdate::UsageUpdate(_) => None,
            // SessionUpdate is #[non_exhaustive].
            _ => None,
        }?;

        Some(api::ResponseEvent {
            r#type: Some(api::response_event::Type::ClientActions(
                api::response_event::ClientActions { actions },
            )),
        })
    }

    /// The `StreamFinished` event that must close the synthesized stream.
    pub fn stream_finished(&self, stop_reason: schema::StopReason) -> api::ResponseEvent {
        use api::response_event::stream_finished::{Done, Other, ReachedMaxTokenLimit, Reason};

        let reason = match stop_reason {
            schema::StopReason::EndTurn
            | schema::StopReason::MaxTurnRequests
            | schema::StopReason::Cancelled => Reason::Done(Done::default()),
            schema::StopReason::MaxTokens => Reason::MaxTokenLimit(ReachedMaxTokenLimit::default()),
            schema::StopReason::Refusal => Reason::Other(Other::default()),
            // StopReason is #[non_exhaustive].
            _ => Reason::Done(Done::default()),
        };
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::Finished(
                api::response_event::StreamFinished {
                    reason: Some(reason),
                    ..Default::default()
                },
            )),
        }
    }

    /// First chunk of a message creates it; subsequent chunks append via
    /// field mask. Switching between text and reasoning starts a new message.
    fn append_streaming_text(&mut self, text: &str, kind: StreamingKind) -> Vec<api::ClientAction> {
        let continuing_message_id = match (&self.target, kind) {
            (StreamingTarget::Text { message_id }, StreamingKind::Text) => Some(message_id.clone()),
            (StreamingTarget::Reasoning { message_id }, StreamingKind::Reasoning) => {
                Some(message_id.clone())
            }
            _ => None,
        };

        match continuing_message_id {
            Some(message_id) => {
                let (message, mask_path) = match kind {
                    StreamingKind::Text => (
                        streaming_text_message(
                            &self.task_id,
                            &self.request_id,
                            &message_id,
                            text,
                            StreamingKind::Text,
                        ),
                        AGENT_OUTPUT_TEXT_MASK,
                    ),
                    StreamingKind::Reasoning => (
                        streaming_text_message(
                            &self.task_id,
                            &self.request_id,
                            &message_id,
                            text,
                            StreamingKind::Reasoning,
                        ),
                        AGENT_REASONING_MASK,
                    ),
                };
                vec![api::ClientAction {
                    action: Some(api::client_action::Action::AppendToMessageContent(
                        api::client_action::AppendToMessageContent {
                            task_id: self.task_id.clone(),
                            message: Some(message),
                            mask: Some(FieldMask {
                                paths: vec![mask_path.to_string()],
                            }),
                        },
                    )),
                }]
            }
            None => {
                let message_id = Uuid::new_v4().to_string();
                self.target = match kind {
                    StreamingKind::Text => StreamingTarget::Text {
                        message_id: message_id.clone(),
                    },
                    StreamingKind::Reasoning => StreamingTarget::Reasoning {
                        message_id: message_id.clone(),
                    },
                };
                vec![add_messages_action(
                    &self.task_id,
                    vec![streaming_text_message(
                        &self.task_id,
                        &self.request_id,
                        &message_id,
                        text,
                        kind,
                    )],
                )]
            }
        }
    }
}

#[derive(Clone, Copy)]
enum StreamingKind {
    Text,
    Reasoning,
}

fn empty_message(task_id: &str, request_id: &str, message_id: &str) -> api::Message {
    api::Message {
        id: message_id.to_string(),
        task_id: task_id.to_string(),
        request_id: request_id.to_string(),
        server_message_data: String::new(),
        citations: vec![],
        fetched_memories: vec![],
        timestamp: None,
        message: None,
    }
}

fn streaming_text_message(
    task_id: &str,
    request_id: &str,
    message_id: &str,
    text: &str,
    kind: StreamingKind,
) -> api::Message {
    let mut message = empty_message(task_id, request_id, message_id);
    message.message = Some(match kind {
        StreamingKind::Text => api::message::Message::AgentOutput(api::message::AgentOutput {
            text: text.to_string(),
        }),
        StreamingKind::Reasoning => {
            api::message::Message::AgentReasoning(api::message::AgentReasoning {
                reasoning: text.to_string(),
                finished_duration: None,
            })
        }
    });
    message
}

/// ACP tool calls are executed by the agent, not the client. The `Server`
/// tool variant is the display-only shape in the native vocabulary, so the
/// ACP tool call rides along as its opaque payload.
fn tool_call_message(
    task_id: &str,
    request_id: &str,
    message_id: &str,
    payload: &str,
) -> api::Message {
    let mut message = empty_message(task_id, request_id, message_id);
    message.message = Some(api::message::Message::ToolCall(api::message::ToolCall {
        tool_call_id: message_id.to_string(),
        tool: Some(api::message::tool_call::Tool::Server(
            api::message::tool_call::Server {
                payload: payload.to_string(),
            },
        )),
    }));
    message
}

fn tool_call_payload(tool_call: &schema::ToolCall) -> String {
    serde_json::to_string(tool_call).unwrap_or_default()
}

fn add_messages_action(task_id: &str, messages: Vec<api::Message>) -> api::ClientAction {
    api::ClientAction {
        action: Some(api::client_action::Action::AddMessagesToTask(
            api::client_action::AddMessagesToTask {
                task_id: task_id.to_string(),
                messages,
            },
        )),
    }
}

/// Extracts streamable text from an ACP content block.
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

#[cfg(test)]
#[path = "proto_shim_tests.rs"]
mod tests;
