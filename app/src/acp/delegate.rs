//! Local implementation of [`acp::AcpClientDelegate`].
//!
//! Routes agent -> client requests for a locally-running ACP agent:
//! `fs/*` against the local filesystem, `terminal/*` against hidden child
//! processes, and `session/request_permission` through a pluggable
//! [`PermissionResolver`] (the UI layer supplies one that shows a prompt;
//! [`AutoDenyPermissionResolver`] is the safe default until then).

use std::collections::HashMap;
use std::io::ErrorKind;
use std::sync::Arc;
use std::time::Duration;

use acp::{schema, AcpClientDelegate};
use anyhow::{anyhow, Context as _, Result};
use async_process::Stdio;
use async_trait::async_trait;
use command::r#async::Command;
use futures::io::{AsyncBufReadExt as _, BufReader};
use futures::lock::Mutex;
use warpui::r#async::executor::Background;
use warpui::r#async::Timer;

/// Decides `session/request_permission` outcomes. The UI layer implements
/// this to show the user a prompt; policy layers (execution-profile
/// allowlists) can wrap another resolver to short-circuit known answers.
#[async_trait]
pub trait PermissionResolver: Send + Sync + 'static {
    async fn resolve(
        &self,
        request: &schema::RequestPermissionRequest,
    ) -> Result<schema::RequestPermissionOutcome>;
}

/// Denies every permission request. Used until a prompt UI is attached so
/// an agent can never act as if the user had approved something.
pub struct AutoDenyPermissionResolver;

#[async_trait]
impl PermissionResolver for AutoDenyPermissionResolver {
    async fn resolve(
        &self,
        _request: &schema::RequestPermissionRequest,
    ) -> Result<schema::RequestPermissionOutcome> {
        Ok(schema::RequestPermissionOutcome::Cancelled)
    }
}

/// Interval for polling a terminal child for exit. Exit is also checked on
/// every `terminal/output` request, so this only bounds `wait_for_exit`
/// latency.
const TERMINAL_EXIT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Default cap on retained terminal output when the agent doesn't set
/// `outputByteLimit`.
const DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT: usize = 1024 * 1024;

#[derive(Default)]
struct TerminalState {
    output: String,
    truncated: bool,
    byte_limit: usize,
    exit_status: Option<schema::TerminalExitStatus>,
}

struct TerminalEntry {
    state: Arc<Mutex<TerminalState>>,
    child: Arc<Mutex<Option<async_process::Child>>>,
}

/// [`AcpClientDelegate`] for agents running on the local machine.
pub struct LocalAcpDelegate {
    executor: Arc<Background>,
    permission_resolver: Arc<dyn PermissionResolver>,
    terminals: Mutex<HashMap<schema::TerminalId, TerminalEntry>>,
}

impl LocalAcpDelegate {
    pub fn new(
        executor: Arc<Background>,
        permission_resolver: Arc<dyn PermissionResolver>,
    ) -> Self {
        Self {
            executor,
            permission_resolver,
            terminals: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl AcpClientDelegate for LocalAcpDelegate {
    async fn request_permission(
        &self,
        request: schema::RequestPermissionRequest,
    ) -> Result<schema::RequestPermissionResponse> {
        let outcome = self.permission_resolver.resolve(&request).await?;
        Ok(schema::RequestPermissionResponse::new(outcome))
    }

    async fn read_text_file(
        &self,
        request: schema::ReadTextFileRequest,
    ) -> Result<schema::ReadTextFileResponse> {
        anyhow::ensure!(
            request.path.is_absolute(),
            "fs/read_text_file path must be absolute"
        );
        let contents = async_fs::read_to_string(&request.path)
            .await
            .with_context(|| format!("Failed to read {}", request.path.display()))?;
        let contents = slice_lines(&contents, request.line, request.limit);
        Ok(schema::ReadTextFileResponse::new(contents))
    }

    async fn write_text_file(
        &self,
        request: schema::WriteTextFileRequest,
    ) -> Result<schema::WriteTextFileResponse> {
        anyhow::ensure!(
            request.path.is_absolute(),
            "fs/write_text_file path must be absolute"
        );
        if let Some(parent) = request.path.parent() {
            if let Err(e) = async_fs::create_dir_all(parent).await {
                if e.kind() != ErrorKind::AlreadyExists {
                    return Err(anyhow::Error::new(e)
                        .context(format!("Failed to create {}", parent.display())));
                }
            }
        }
        async_fs::write(&request.path, request.content.as_bytes())
            .await
            .with_context(|| format!("Failed to write {}", request.path.display()))?;
        Ok(schema::WriteTextFileResponse::new())
    }

    async fn create_terminal(
        &self,
        request: schema::CreateTerminalRequest,
    ) -> Result<schema::CreateTerminalResponse> {
        let mut command = Command::new(&request.command);
        command.args(&request.args);
        command.envs(
            request
                .env
                .iter()
                .map(|var| (var.name.clone(), var.value.clone())),
        );
        if let Some(cwd) = &request.cwd {
            command.current_dir(cwd);
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command
            .spawn()
            .with_context(|| format!("Failed to spawn terminal command {}", request.command))?;

        let byte_limit = request
            .output_byte_limit
            .map(|limit| usize::try_from(limit).unwrap_or(usize::MAX))
            .unwrap_or(DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT);
        let state = Arc::new(Mutex::new(TerminalState {
            byte_limit,
            ..Default::default()
        }));

        // Stream stdout and stderr into the shared buffer.
        let mut readers: Vec<Box<dyn futures::io::AsyncRead + Unpin + Send>> = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            readers.push(Box::new(stdout));
        }
        if let Some(stderr) = child.stderr.take() {
            readers.push(Box::new(stderr));
        }
        for reader in readers {
            let state = state.clone();
            self.executor
                .spawn(async move {
                    let mut lines = BufReader::new(reader);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        match lines.read_line(&mut line).await {
                            Ok(0) => break,
                            Ok(_) => {
                                let mut state = state.lock().await;
                                state.output.push_str(&line);
                                let limit = state.byte_limit;
                                if state.output.len() > limit {
                                    truncate_to_byte_limit(&mut state.output, limit);
                                    state.truncated = true;
                                }
                            }
                            Err(e) => {
                                log::debug!("ACP terminal: output stream error: {e}");
                                break;
                            }
                        }
                    }
                })
                .detach();
        }

        let terminal_id = schema::TerminalId::new(uuid::Uuid::new_v4().to_string());
        let entry = TerminalEntry {
            state,
            child: Arc::new(Mutex::new(Some(child))),
        };
        self.terminals
            .lock()
            .await
            .insert(terminal_id.clone(), entry);
        Ok(schema::CreateTerminalResponse::new(terminal_id))
    }

    async fn terminal_output(
        &self,
        request: schema::TerminalOutputRequest,
    ) -> Result<schema::TerminalOutputResponse> {
        let (state, child) = self.terminal_parts(&request.terminal_id).await?;
        check_exit(&state, &child).await;
        let state = state.lock().await;
        let mut response =
            schema::TerminalOutputResponse::new(state.output.clone(), state.truncated);
        response.exit_status = state.exit_status.clone();
        Ok(response)
    }

    async fn wait_for_terminal_exit(
        &self,
        request: schema::WaitForTerminalExitRequest,
    ) -> Result<schema::WaitForTerminalExitResponse> {
        let (state, child) = self.terminal_parts(&request.terminal_id).await?;
        loop {
            check_exit(&state, &child).await;
            if let Some(exit_status) = state.lock().await.exit_status.clone() {
                return Ok(schema::WaitForTerminalExitResponse::new(exit_status));
            }
            Timer::after(TERMINAL_EXIT_POLL_INTERVAL).await;
        }
    }

    async fn kill_terminal(
        &self,
        request: schema::KillTerminalRequest,
    ) -> Result<schema::KillTerminalResponse> {
        let (_, child) = self.terminal_parts(&request.terminal_id).await?;
        let mut child_guard = child.lock().await;
        if let Some(child) = child_guard.as_mut() {
            let _ = child.kill();
        }
        Ok(schema::KillTerminalResponse::new())
    }

    async fn release_terminal(
        &self,
        request: schema::ReleaseTerminalRequest,
    ) -> Result<schema::ReleaseTerminalResponse> {
        let entry = self.terminals.lock().await.remove(&request.terminal_id);
        if let Some(entry) = entry {
            let mut child_guard = entry.child.lock().await;
            if let Some(mut child) = child_guard.take() {
                // Kill unless it already exited (kill on an exited child is a no-op error).
                let _ = child.kill();
            }
        }
        Ok(schema::ReleaseTerminalResponse::new())
    }
}

impl LocalAcpDelegate {
    async fn terminal_parts(
        &self,
        terminal_id: &schema::TerminalId,
    ) -> Result<(
        Arc<Mutex<TerminalState>>,
        Arc<Mutex<Option<async_process::Child>>>,
    )> {
        let terminals = self.terminals.lock().await;
        let entry = terminals
            .get(terminal_id)
            .ok_or_else(|| anyhow!("Unknown terminal: {terminal_id}"))?;
        Ok((entry.state.clone(), entry.child.clone()))
    }
}

/// Records the exit status on `state` if the child has exited.
async fn check_exit(
    state: &Arc<Mutex<TerminalState>>,
    child: &Arc<Mutex<Option<async_process::Child>>>,
) {
    if state.lock().await.exit_status.is_some() {
        return;
    }
    let mut child_guard = child.lock().await;
    let Some(child) = child_guard.as_mut() else {
        return;
    };
    match child.try_status() {
        Ok(Some(status)) => {
            let exit_status = exit_status_to_acp(status);
            state.lock().await.exit_status = Some(exit_status);
        }
        Ok(None) => {}
        Err(e) => log::debug!("ACP terminal: try_status failed: {e}"),
    }
}

fn exit_status_to_acp(status: std::process::ExitStatus) -> schema::TerminalExitStatus {
    let exit_code = status.code().and_then(|code| u32::try_from(code).ok());
    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt as _;
        status.signal().map(|signal| signal.to_string())
    };
    #[cfg(not(unix))]
    let signal: Option<String> = None;
    schema::TerminalExitStatus::new()
        .exit_code(exit_code)
        .signal(signal)
}

/// Applies ACP `fs/read_text_file` line slicing: `line` is the 1-based first
/// line to include, `limit` the maximum number of lines.
fn slice_lines(contents: &str, line: Option<u32>, limit: Option<u32>) -> String {
    if line.is_none() && limit.is_none() {
        return contents.to_string();
    }
    let skip = line.map(|l| l.saturating_sub(1) as usize).unwrap_or(0);
    let lines = contents.lines().skip(skip);
    match limit {
        Some(limit) => lines.take(limit as usize).collect::<Vec<_>>().join("\n"),
        None => lines.collect::<Vec<_>>().join("\n"),
    }
}

/// Truncates from the *front* of `output` so at most `limit` bytes remain,
/// cutting only at a character boundary per the ACP terminal contract.
fn truncate_to_byte_limit(output: &mut String, limit: usize) {
    if output.len() <= limit {
        return;
    }
    let mut start = output.len() - limit;
    while start < output.len() && !output.is_char_boundary(start) {
        start += 1;
    }
    output.replace_range(..start, "");
}

#[cfg(test)]
#[path = "delegate_tests.rs"]
mod tests;
