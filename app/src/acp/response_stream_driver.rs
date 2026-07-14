//! Produces a native [`ResponseStream`] from an ACP prompt turn.
//!
//! This is the runtime half of the bridge (the mapping rules live in
//! [`super::proto_shim`]): it sends `session/prompt`, forwards the
//! connection's `session/update` notifications through the shim while the
//! turn is in flight, and closes the stream with `StreamFinished` when the
//! prompt resolves. The returned stream is exactly what
//! `BlocklistAIController` consumes from the multi-agent backend, so the
//! native conversation pipeline renders ACP turns unchanged.

use std::sync::Arc;

use acp::{schema, AcpAgentConnection};
use anyhow::anyhow;
use futures::channel::oneshot;
use futures::{FutureExt as _, StreamExt as _};

use super::proto_shim::AcpProtoShim;
use crate::ai::agent::api::ResponseStream;
use crate::server::server_api::AIApiError;

/// Identifiers tying the synthesized stream to the native conversation
/// structures it feeds.
pub struct AcpStreamIds {
    /// The native conversation this turn belongs to.
    pub conversation_id: String,
    /// The task messages are appended to (the conversation's root task).
    pub task_id: String,
    /// The native request id for this turn.
    pub request_id: String,
}

/// Runs one ACP prompt turn as a native response-event stream.
///
/// Firing `cancellation_rx` forwards `session/cancel` to the agent; per the
/// ACP contract the agent then resolves the prompt with
/// `StopReason::Cancelled`, which closes the stream gracefully.
pub fn acp_response_stream(
    connection: Arc<AcpAgentConnection>,
    session_id: schema::SessionId,
    prompt: Vec<schema::ContentBlock>,
    ids: AcpStreamIds,
    cancellation_rx: oneshot::Receiver<()>,
) -> ResponseStream {
    let AcpStreamIds {
        conversation_id,
        task_id,
        request_id,
    } = ids;
    let mut shim = AcpProtoShim::new(task_id, request_id);

    let stream = async_stream::stream! {
        yield Ok(shim.stream_init(conversation_id));

        let request = schema::PromptRequest::new(session_id.clone(), prompt);
        let mut updates = connection.session_updates().fuse();
        let mut prompt_result = Box::pin(connection.prompt(request).fuse());
        let mut cancellation_rx = cancellation_rx.fuse();

        loop {
            futures::select! {
                result = prompt_result => {
                    // Drain any updates that raced with turn completion so
                    // trailing output isn't dropped. The subscription channel
                    // is filled synchronously by the read loop before the
                    // prompt response resolves, so everything the agent sent
                    // first is already available here.
                    while let Some(Some(notification)) = updates.next().now_or_never() {
                        if notification.session_id == session_id {
                            if let Some(event) = shim.map_update(notification.update) {
                                yield Ok(event);
                            }
                        }
                    }
                    match result {
                        Ok(response) => yield Ok(shim.stream_finished(response.stop_reason)),
                        Err(e) => yield Err(Arc::new(AIApiError::Other(anyhow!(e)))),
                    }
                    break;
                }
                _ = cancellation_rx => {
                    // Ask the agent to stop; keep looping — the prompt will
                    // resolve (with StopReason::Cancelled) and end the stream.
                    if let Err(e) = connection.cancel(session_id.clone()) {
                        log::warn!("ACP: failed to send session/cancel: {e:#}");
                    }
                }
                update = updates.next() => {
                    match update {
                        Some(notification) => {
                            if notification.session_id == session_id {
                                if let Some(event) = shim.map_update(notification.update) {
                                    yield Ok(event);
                                }
                            }
                        }
                        None => {
                            // Update stream closed: the connection is gone.
                            // Await the prompt future for the definitive
                            // outcome instead of spinning on a closed channel.
                            match (&mut prompt_result).await {
                                Ok(response) => {
                                    yield Ok(shim.stream_finished(response.stop_reason));
                                }
                                Err(e) => yield Err(Arc::new(AIApiError::Other(anyhow!(e)))),
                            }
                            break;
                        }
                    }
                }
            }
        }
    };

    Box::pin(stream)
}

#[cfg(test)]
#[path = "response_stream_driver_tests.rs"]
mod tests;
