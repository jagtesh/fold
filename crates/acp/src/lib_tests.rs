use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use command::r#async::Command;
use futures::executor::block_on;
use jsonrpc::Transport;
use serde_json::json;

use super::*;

struct UnimplementedDelegate;

#[async_trait]
impl AcpClientDelegate for UnimplementedDelegate {
    async fn request_permission(
        &self,
        _request: schema::RequestPermissionRequest,
    ) -> Result<schema::RequestPermissionResponse> {
        anyhow::bail!("unimplemented")
    }

    async fn read_text_file(
        &self,
        request: schema::ReadTextFileRequest,
    ) -> Result<schema::ReadTextFileResponse> {
        Ok(schema::ReadTextFileResponse::new(format!(
            "contents of {}",
            request.path.display()
        )))
    }

    async fn write_text_file(
        &self,
        _request: schema::WriteTextFileRequest,
    ) -> Result<schema::WriteTextFileResponse> {
        anyhow::bail!("unimplemented")
    }

    async fn create_terminal(
        &self,
        _request: schema::CreateTerminalRequest,
    ) -> Result<schema::CreateTerminalResponse> {
        anyhow::bail!("unimplemented")
    }

    async fn terminal_output(
        &self,
        _request: schema::TerminalOutputRequest,
    ) -> Result<schema::TerminalOutputResponse> {
        anyhow::bail!("unimplemented")
    }

    async fn wait_for_terminal_exit(
        &self,
        _request: schema::WaitForTerminalExitRequest,
    ) -> Result<schema::WaitForTerminalExitResponse> {
        anyhow::bail!("unimplemented")
    }

    async fn kill_terminal(
        &self,
        _request: schema::KillTerminalRequest,
    ) -> Result<schema::KillTerminalResponse> {
        anyhow::bail!("unimplemented")
    }

    async fn release_terminal(
        &self,
        _request: schema::ReleaseTerminalRequest,
    ) -> Result<schema::ReleaseTerminalResponse> {
        anyhow::bail!("unimplemented")
    }
}

#[test]
fn dispatch_unknown_method_reports_method_not_found() {
    let delegate = UnimplementedDelegate;
    let result = block_on(dispatch_agent_request(
        &delegate,
        "some/unknown_method",
        json!({}),
    ));
    assert!(matches!(result, Err(DispatchError::UnknownMethod)));
}

#[test]
fn dispatch_read_text_file_returns_delegate_response() {
    let delegate = UnimplementedDelegate;
    let result = block_on(dispatch_agent_request(
        &delegate,
        schema::CLIENT_METHOD_NAMES.fs_read_text_file,
        json!({"sessionId": "sess-1", "path": "/tmp/file.txt"}),
    ));
    let value = match result {
        Ok(value) => value,
        Err(DispatchError::UnknownMethod) => panic!("method should be known"),
        Err(DispatchError::Failed(e)) => panic!("dispatch failed: {e}"),
    };
    assert_eq!(value["content"], "contents of /tmp/file.txt");
}

#[test]
fn dispatch_delegate_failure_is_reported() {
    let delegate = UnimplementedDelegate;
    let result = block_on(dispatch_agent_request(
        &delegate,
        schema::CLIENT_METHOD_NAMES.fs_write_text_file,
        json!({"sessionId": "sess-1", "path": "/tmp/file.txt", "content": "x"}),
    ));
    assert!(matches!(result, Err(DispatchError::Failed(_))));
}

#[test]
fn dispatch_invalid_params_is_reported() {
    let delegate = UnimplementedDelegate;
    let result = block_on(dispatch_agent_request(
        &delegate,
        schema::CLIENT_METHOD_NAMES.fs_read_text_file,
        json!({"not": "valid"}),
    ));
    assert!(matches!(result, Err(DispatchError::Failed(_))));
}

/// Round-trips NDJSON frames through a real process (`cat` echoes stdin to
/// stdout) to exercise framing, blank-line skipping, and EOF handling.
#[test]
#[cfg(unix)]
fn ndjson_transport_round_trip_through_cat() {
    let executor = Arc::new(warpui_core::r#async::executor::Background::default());
    let transport = NdjsonProcessTransport::new(Command::new("cat"), executor, None)
        .expect("failed to spawn cat");

    block_on(async {
        let first = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let second = r#"{"jsonrpc":"2.0","method":"session/update","params":{"k":"v"}}"#;
        transport.write(first).await.unwrap();
        transport.write(second).await.unwrap();

        assert_eq!(transport.read().await.unwrap(), first);
        assert_eq!(transport.read().await.unwrap(), second);

        // Messages containing raw newlines must be rejected, not corrupt the stream.
        assert!(transport.write("{\n}").await.is_err());

        transport
            .shutdown(std::time::Duration::from_secs(5))
            .await
            .unwrap();

        // After shutdown/EOF, read returns an empty string.
        assert_eq!(transport.read().await.unwrap(), "");
    });
}
