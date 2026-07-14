use std::sync::Arc;

use async_process::{Child, ChildStdin, ChildStdout, Stdio};
use async_trait::async_trait;
use command::r#async::Command;
use futures::future::FutureExt;
use futures::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use futures::lock::Mutex;
use jsonrpc::Transport;
use simple_logger::SimpleLogger;
use warp_errors::report_error;
use warpui_core::r#async::executor::{Background, BackgroundTask};
use warpui_core::r#async::Timer;

/// Transport for ACP communication over process stdin/stdout using
/// newline-delimited JSON framing (one complete JSON-RPC message per line,
/// as specified by the Agent Client Protocol). Also manages the agent
/// process lifecycle with graceful shutdown.
///
/// This is the NDJSON sibling of the LSP `ProcessTransport`, which uses
/// `Content-Length` framing instead.
#[derive(Clone)]
pub struct NdjsonProcessTransport {
    input: Arc<Mutex<BufReader<ChildStdout>>>,
    output: Arc<Mutex<BufWriter<ChildStdin>>>,
    child: Arc<Mutex<Option<Child>>>,
    stderr_task: Arc<Mutex<Option<BackgroundTask>>>,
}

impl NdjsonProcessTransport {
    /// Spawns the agent process and wires its stdio.
    ///
    /// If `logger` is provided, stderr output is written to that logger's file
    /// in addition to being logged via `log::debug!`.
    pub fn new(
        mut command: Command,
        executor: Arc<Background>,
        logger: Option<SimpleLogger>,
    ) -> anyhow::Result<Self> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn ACP agent process: {e}"))?;

        let child_pid = child.id();
        log::info!("NdjsonProcessTransport: Spawned agent process with pid {child_pid}");

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to get agent stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to get agent stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to get agent stderr"))?;

        // stderr -> logger background task
        let stderr_task = executor.spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut buffer = String::new();
            loop {
                buffer.clear();
                match reader.read_line(&mut buffer).await {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        let message = buffer.trim_end();
                        if let Some(ref logger) = logger {
                            logger.log(format!("[stderr] {message}"));
                        }
                        log::debug!("ACP agent [pid: {child_pid}] stderr: {message}");
                    }
                    Err(e) => {
                        if let Some(ref logger) = logger {
                            logger.log(format!("[error] Error reading stderr: {e}"));
                        }
                        report_error!(
                            anyhow::Error::new(e)
                                .context("NdjsonProcessTransport: Error reading stderr"),
                            extra: { "pid" => %child_pid }
                        );
                        break;
                    }
                }
            }
        });

        Ok(Self {
            input: Arc::new(Mutex::new(BufReader::new(stdout))),
            output: Arc::new(Mutex::new(BufWriter::new(stdin))),
            child: Arc::new(Mutex::new(Some(child))),
            stderr_task: Arc::new(Mutex::new(Some(stderr_task))),
        })
    }
}

#[async_trait]
impl Transport for NdjsonProcessTransport {
    async fn read(&self) -> anyhow::Result<String> {
        loop {
            let mut line = String::new();
            let bytes_read = {
                let mut reader = self.input.lock().await;
                reader.read_line(&mut line).await?
            };
            if bytes_read == 0 {
                // EOF: the service's read loop treats an empty message as a
                // closed transport.
                return Ok(String::new());
            }

            let message = line.trim();
            if message.is_empty() {
                // Skip blank lines between messages.
                continue;
            }
            return Ok(message.to_string());
        }
    }

    async fn write(&self, message: &str) -> anyhow::Result<()> {
        // NDJSON framing requires exactly one message per line. Serialized
        // JSON-RPC messages never contain raw newlines (serde_json escapes
        // them), so reject rather than corrupt the stream if one shows up.
        anyhow::ensure!(
            !message.contains('\n'),
            "NDJSON message must not contain newlines"
        );
        {
            let mut writer = self.output.lock().await;
            writer.write_all(message.as_bytes()).await?;
            writer.write_all(b"\n").await?;
            writer.flush().await?;
        }
        Ok(())
    }

    async fn shutdown(&self, timeout: std::time::Duration) -> anyhow::Result<()> {
        log::info!("ACP: Shutting down agent process.");

        let child = {
            let mut child_guard = self.child.lock().await;
            match child_guard.take() {
                Some(c) => c,
                None => {
                    log::warn!("ACP: Agent process already shut down.");
                    return Ok(());
                }
            }
        };

        let mut child = child;
        let shutdown = child.status();
        let timeout_future = Timer::after(timeout);
        futures::select! {
            _ = shutdown.fuse() => {},
            _ = timeout_future.fuse() => {
                // On *nix platforms, send a SIGTERM with a 2s grace period
                // before killing the process.
                #[cfg(unix)]
                {
                    use nix::sys::signal::{kill, Signal};
                    use nix::unistd::Pid;
                    use std::time::Duration;
                    const SIGTERM_TIMEOUT: Duration = Duration::from_secs(2);
                    if kill(Pid::from_raw(child.id() as i32), Signal::SIGTERM).is_ok() {
                        Timer::after(SIGTERM_TIMEOUT).await;
                    }
                }

                let _ = child.kill();
            }
        }

        // Wait for the stderr task because it owns the last logger clone.
        // Joining it ensures that clone is dropped before restart so the same
        // log path can be registered again without colliding with a stale entry.
        if let Some(stderr_task) = self.stderr_task.lock().await.take() {
            if let Err(e) = stderr_task.await {
                log::warn!("ACP: Failed to join stderr task: {e}");
            }
        }
        log::info!("ACP: Agent process shut down.");
        Ok(())
    }
}
