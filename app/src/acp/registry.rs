//! Binds native conversations to ACP agent sessions.
//!
//! When a conversation is ACP-backed, its turns must be served by the
//! agent process instead of the multi-agent backend. The registry is the
//! lookup `ResponseStream::spawn_generate` consults: bindings are keyed by
//! the conversation's **root task id**, which is the one conversation-stable
//! identifier present in [`api::RequestParams`] (`params.tasks`).

use std::collections::HashMap;
use std::sync::Arc;

use acp::{schema, AcpAgentConnection};
use futures::channel::oneshot;
use uuid::Uuid;
use warpui::{AppContext, Entity, ModelContext, SingletonEntity};

use super::response_stream_driver::{acp_response_stream, AcpStreamIds};
use crate::ai::agent::{api, AIAgentInput};

/// One conversation's link to its ACP agent session.
#[derive(Clone)]
pub struct AcpBinding {
    pub connection: Arc<AcpAgentConnection>,
    pub session_id: schema::SessionId,
    /// The native conversation id, round-tripped into `StreamInit`.
    pub conversation_id: String,
}

pub enum AcpConversationRegistryEvent {}

/// Singleton mapping root task ids to ACP bindings.
#[derive(Default)]
pub struct AcpConversationRegistry {
    bindings_by_root_task: HashMap<String, AcpBinding>,
}

impl AcpConversationRegistry {
    pub fn register(&mut self, root_task_id: String, binding: AcpBinding) {
        self.bindings_by_root_task.insert(root_task_id, binding);
    }

    pub fn unregister(&mut self, root_task_id: &str) -> Option<AcpBinding> {
        self.bindings_by_root_task.remove(root_task_id)
    }

    /// Returns the binding when any of the request's tasks belongs to an
    /// ACP-backed conversation.
    fn binding_for_tasks<'a>(
        &self,
        tasks: impl IntoIterator<Item = &'a warp_multi_agent_api::Task>,
    ) -> Option<&AcpBinding> {
        tasks
            .into_iter()
            .find_map(|task| self.bindings_by_root_task.get(&task.id))
    }
}

impl Entity for AcpConversationRegistry {
    type Event = AcpConversationRegistryEvent;
}

impl SingletonEntity for AcpConversationRegistry {}

/// Consulted by `ResponseStream::spawn_generate`: when the request belongs
/// to an ACP-backed conversation, returns the synthesized response stream
/// for one agent turn; otherwise hands the cancellation receiver back so
/// the native multi-agent path proceeds unchanged.
pub fn try_acp_response_stream(
    params: &api::RequestParams,
    cancellation_rx: oneshot::Receiver<()>,
    ctx: &AppContext,
) -> Result<api::ResponseStream, oneshot::Receiver<()>> {
    let registry = AcpConversationRegistry::as_ref(ctx);
    let Some(binding) = registry.binding_for_tasks(&params.tasks) else {
        return Err(cancellation_rx);
    };
    let binding = binding.clone();

    let root_task_id = params
        .tasks
        .first()
        .map(|task| task.id.clone())
        .unwrap_or_default();

    Ok(acp_response_stream(
        binding.connection,
        binding.session_id,
        prompt_blocks(&params.input),
        AcpStreamIds {
            conversation_id: binding.conversation_id,
            task_id: root_task_id,
            request_id: Uuid::new_v4().to_string(),
        },
        cancellation_rx,
    ))
}

/// Extracts the promptable text from the request inputs. ACP agents take a
/// plain content-block prompt; context attachments and action results have
/// no ACP representation yet.
fn prompt_blocks(inputs: &[AIAgentInput]) -> Vec<schema::ContentBlock> {
    let text = inputs
        .iter()
        .filter_map(|input| match input {
            AIAgentInput::UserQuery { query, .. }
            | AIAgentInput::AutoCodeDiffQuery { query, .. } => Some(query.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    vec![schema::ContentBlock::Text(schema::TextContent::new(text))]
}

/// Convenience for model contexts (e.g. controller code) that need to
/// register a binding.
pub fn register_binding<T: Entity>(
    root_task_id: String,
    binding: AcpBinding,
    ctx: &mut ModelContext<T>,
) {
    AcpConversationRegistry::handle(ctx).update(ctx, |registry, _| {
        registry.register(root_task_id, binding);
    });
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
