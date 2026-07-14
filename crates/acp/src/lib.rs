//! ACP (Agent Client Protocol) client.
//!
//! Speaks ACP over newline-delimited JSON-RPC to an external agent process
//! (e.g. `claude-code-acp`, `gemini --experimental-acp`). Warp is the
//! *client* in ACP terms; the spawned process is the *agent*.
//!
//! The wire protocol runs on [`jsonrpc::JsonRpcService`]; this crate adds the
//! NDJSON process transport, typed method wrappers for client -> agent
//! requests, a typed stream of `session/update` notifications, and a
//! delegate trait for agent -> client requests (permission prompts, fs
//! access, terminal management).

mod transport;

use std::sync::Arc;
use std::time::Duration;

pub use agent_client_protocol_schema::{v1 as schema, ProtocolVersion};
use anyhow::{anyhow, Context as _, Result};
use async_trait::async_trait;
use command::r#async::Command;
use jsonrpc::{JsonRpcService, ServerNotificationEvent};
use schema::{AGENT_METHOD_NAMES, CLIENT_METHOD_NAMES};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use simple_logger::SimpleLogger;
pub use transport::NdjsonProcessTransport;
use warpui_core::r#async::executor::Background;

/// JSON-RPC error code returned to the agent for unhandled or failed
/// agent -> client requests (standard "Internal error").
pub const JSON_RPC_INTERNAL_ERROR: i64 = -32603;

/// JSON-RPC error code for methods this client does not implement.
pub const JSON_RPC_METHOD_NOT_FOUND: i64 = -32601;

/// Capacity of the `session/update` notification channel. Updates stream
/// fast during a prompt turn; the consumer drains them onto the UI thread.
const SESSION_UPDATE_CHANNEL_CAPACITY: usize = 1024;

/// How long to give the agent process to exit before escalating to
/// SIGTERM/kill during shutdown.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

/// Handles agent -> client requests. Implemented by the embedding
/// application; every method corresponds to an ACP client capability.
///
/// Implementations run on the background executor and may take arbitrarily
/// long (e.g. waiting for the user to answer a permission prompt) — the
/// JSON-RPC response is only written once the returned future resolves.
#[async_trait]
pub trait AcpClientDelegate: Send + Sync + 'static {
    async fn request_permission(
        &self,
        request: schema::RequestPermissionRequest,
    ) -> Result<schema::RequestPermissionResponse>;

    async fn read_text_file(
        &self,
        request: schema::ReadTextFileRequest,
    ) -> Result<schema::ReadTextFileResponse>;

    async fn write_text_file(
        &self,
        request: schema::WriteTextFileRequest,
    ) -> Result<schema::WriteTextFileResponse>;

    async fn create_terminal(
        &self,
        request: schema::CreateTerminalRequest,
    ) -> Result<schema::CreateTerminalResponse>;

    async fn terminal_output(
        &self,
        request: schema::TerminalOutputRequest,
    ) -> Result<schema::TerminalOutputResponse>;

    async fn wait_for_terminal_exit(
        &self,
        request: schema::WaitForTerminalExitRequest,
    ) -> Result<schema::WaitForTerminalExitResponse>;

    async fn kill_terminal(
        &self,
        request: schema::KillTerminalRequest,
    ) -> Result<schema::KillTerminalResponse>;

    async fn release_terminal(
        &self,
        request: schema::ReleaseTerminalRequest,
    ) -> Result<schema::ReleaseTerminalResponse>;
}

/// A connection to a running ACP agent process.
///
/// Dropping the connection does not kill the agent; call [`Self::shutdown`].
pub struct AcpAgentConnection {
    service: Arc<JsonRpcService>,
    session_updates: async_channel::Receiver<schema::SessionNotification>,
    executor: Arc<Background>,
}

impl AcpAgentConnection {
    /// Spawns the agent process and starts the JSON-RPC read loop.
    ///
    /// `delegate` handles agent -> client requests. If `logger` is provided,
    /// the agent's stderr is mirrored to its file.
    pub fn spawn(
        command: Command,
        delegate: Arc<dyn AcpClientDelegate>,
        executor: Arc<Background>,
        logger: Option<SimpleLogger>,
    ) -> Result<Self> {
        let transport = NdjsonProcessTransport::new(command, executor.clone(), logger)?;
        let service = Arc::new(JsonRpcService::new(
            Box::new(transport),
            executor.clone(),
            JSON_RPC_METHOD_NOT_FOUND,
        ));

        Self::install_delegate(&service, delegate, &executor);

        // Bridge raw session/update notifications into a typed channel.
        let (raw_tx, raw_rx) = async_channel::bounded(SESSION_UPDATE_CHANNEL_CAPACITY);
        let (typed_tx, typed_rx) = async_channel::bounded(SESSION_UPDATE_CHANNEL_CAPACITY);
        let service_for_subscribe = service.clone();
        let executor_clone = executor.clone();
        executor
            .spawn(async move {
                service_for_subscribe
                    .subscribe(CLIENT_METHOD_NAMES.session_update.to_string(), raw_tx)
                    .await;
                executor_clone
                    .spawn(async move {
                        while let Ok(event) = raw_rx.recv().await {
                            let ServerNotificationEvent { params, .. } = event;
                            match serde_json::from_value::<schema::SessionNotification>(params) {
                                Ok(notification) => {
                                    if typed_tx.send(notification).await.is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    log::warn!("ACP: Failed to parse session/update: {e}");
                                }
                            }
                        }
                    })
                    .detach();
            })
            .detach();

        Ok(Self {
            service,
            session_updates: typed_rx,
            executor,
        })
    }

    /// Returns the stream of `session/update` notifications for all sessions
    /// on this connection.
    pub fn session_updates(&self) -> async_channel::Receiver<schema::SessionNotification> {
        self.session_updates.clone()
    }

    /// Performs the `initialize` handshake.
    pub async fn initialize(
        &self,
        request: schema::InitializeRequest,
    ) -> Result<schema::InitializeResponse> {
        self.request(AGENT_METHOD_NAMES.initialize, request).await
    }

    /// Authenticates with the agent using one of the methods it advertised.
    pub async fn authenticate(
        &self,
        request: schema::AuthenticateRequest,
    ) -> Result<schema::AuthenticateResponse> {
        self.request(AGENT_METHOD_NAMES.authenticate, request).await
    }

    /// Creates a new session.
    pub async fn new_session(
        &self,
        request: schema::NewSessionRequest,
    ) -> Result<schema::NewSessionResponse> {
        self.request(AGENT_METHOD_NAMES.session_new, request).await
    }

    /// Loads an existing session (only if the agent advertises `loadSession`).
    pub async fn load_session(
        &self,
        request: schema::LoadSessionRequest,
    ) -> Result<schema::LoadSessionResponse> {
        self.request(AGENT_METHOD_NAMES.session_load, request).await
    }

    /// Sends a prompt turn. Resolves when the turn completes; progress
    /// streams through [`Self::session_updates`] in the meantime.
    pub async fn prompt(&self, request: schema::PromptRequest) -> Result<schema::PromptResponse> {
        self.request(AGENT_METHOD_NAMES.session_prompt, request)
            .await
    }

    /// Cancels the in-flight prompt turn for a session (fire-and-forget).
    pub fn cancel(&self, session_id: schema::SessionId) -> Result<()> {
        let notification = schema::CancelNotification::new(session_id);
        self.service.send_notification(
            AGENT_METHOD_NAMES.session_cancel.to_string(),
            serde_json::to_value(notification)?,
        )
    }

    /// Shuts down the agent process, escalating to SIGTERM/kill after a
    /// grace period.
    pub async fn shutdown(&self) -> Result<()> {
        self.service.shutdown(SHUTDOWN_TIMEOUT).await
    }

    async fn request<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        method: &str,
        request: Req,
    ) -> Result<Resp> {
        let params = serde_json::to_value(request)?;
        let id = self.service.next_id();
        let value = self
            .service
            .send_request(id, method.to_string(), params)
            .await
            .with_context(|| format!("ACP request {method} failed"))?;
        serde_json::from_value(value)
            .with_context(|| format!("Failed to parse ACP response for {method}"))
    }

    /// Routes agent -> client requests to the delegate on the background
    /// executor, writing the JSON-RPC response when the delegate resolves.
    fn install_delegate(
        service: &Arc<JsonRpcService>,
        delegate: Arc<dyn AcpClientDelegate>,
        executor: &Arc<Background>,
    ) {
        let service_for_responder = service.clone();
        let executor = executor.clone();
        service.set_server_request_responder(move |method, params, request_id| {
            let service = service_for_responder.clone();
            let delegate = delegate.clone();
            executor
                .spawn(async move {
                    let result = dispatch_agent_request(&*delegate, &method, params).await;
                    let write_result = match result {
                        Ok(value) => service.respond(request_id, value).await,
                        Err(DispatchError::UnknownMethod) => {
                            service
                                .respond_with_error(
                                    request_id,
                                    JSON_RPC_METHOD_NOT_FOUND,
                                    &format!("Method {method} not implemented"),
                                )
                                .await
                        }
                        Err(DispatchError::Failed(e)) => {
                            log::warn!("ACP: delegate failed for {method}: {e:#}");
                            service
                                .respond_with_error(
                                    request_id,
                                    JSON_RPC_INTERNAL_ERROR,
                                    &e.to_string(),
                                )
                                .await
                        }
                    };
                    if let Err(e) = write_result {
                        log::warn!("ACP: failed to write response for {method}: {e:#}");
                    }
                })
                .detach();
            Ok(())
        });
    }
}

enum DispatchError {
    UnknownMethod,
    Failed(anyhow::Error),
}

impl From<anyhow::Error> for DispatchError {
    fn from(e: anyhow::Error) -> Self {
        DispatchError::Failed(e)
    }
}

async fn dispatch_agent_request(
    delegate: &dyn AcpClientDelegate,
    method: &str,
    params: Value,
) -> Result<Value, DispatchError> {
    fn parse<T: DeserializeOwned>(method: &str, params: Value) -> Result<T, DispatchError> {
        serde_json::from_value(params)
            .map_err(|e| DispatchError::Failed(anyhow!("Invalid params for {method}: {e}")))
    }
    fn encode<T: Serialize>(response: T) -> Result<Value, DispatchError> {
        serde_json::to_value(response)
            .map_err(|e| DispatchError::Failed(anyhow!("Failed to encode response: {e}")))
    }

    if method == CLIENT_METHOD_NAMES.session_request_permission {
        let request = parse(method, params)?;
        encode(delegate.request_permission(request).await?)
    } else if method == CLIENT_METHOD_NAMES.fs_read_text_file {
        let request = parse(method, params)?;
        encode(delegate.read_text_file(request).await?)
    } else if method == CLIENT_METHOD_NAMES.fs_write_text_file {
        let request = parse(method, params)?;
        encode(delegate.write_text_file(request).await?)
    } else if method == CLIENT_METHOD_NAMES.terminal_create {
        let request = parse(method, params)?;
        encode(delegate.create_terminal(request).await?)
    } else if method == CLIENT_METHOD_NAMES.terminal_output {
        let request = parse(method, params)?;
        encode(delegate.terminal_output(request).await?)
    } else if method == CLIENT_METHOD_NAMES.terminal_wait_for_exit {
        let request = parse(method, params)?;
        encode(delegate.wait_for_terminal_exit(request).await?)
    } else if method == CLIENT_METHOD_NAMES.terminal_kill {
        let request = parse(method, params)?;
        encode(delegate.kill_terminal(request).await?)
    } else if method == CLIENT_METHOD_NAMES.terminal_release {
        let request = parse(method, params)?;
        encode(delegate.release_terminal(request).await?)
    } else {
        Err(DispatchError::UnknownMethod)
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
