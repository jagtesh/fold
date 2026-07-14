use super::*;

#[test]
fn slice_lines_defaults_to_full_contents() {
    assert_eq!(slice_lines("a\nb\nc", None, None), "a\nb\nc");
}

#[test]
fn slice_lines_applies_one_based_start_line() {
    assert_eq!(slice_lines("a\nb\nc", Some(2), None), "b\nc");
    assert_eq!(slice_lines("a\nb\nc", Some(1), None), "a\nb\nc");
}

#[test]
fn slice_lines_applies_limit() {
    assert_eq!(slice_lines("a\nb\nc", None, Some(2)), "a\nb");
    assert_eq!(slice_lines("a\nb\nc", Some(2), Some(1)), "b");
    assert_eq!(slice_lines("a\nb\nc", Some(10), Some(2)), "");
}

#[test]
fn truncate_keeps_suffix_within_limit() {
    let mut output = "abcdefgh".to_string();
    truncate_to_byte_limit(&mut output, 3);
    assert_eq!(output, "fgh");
}

#[test]
fn truncate_respects_char_boundaries() {
    // Each snowman is 3 bytes; a 4-byte budget cannot start mid-character,
    // so only the final full character survives.
    let mut output = "☃☃☃".to_string();
    truncate_to_byte_limit(&mut output, 4);
    assert_eq!(output, "☃");
}

#[test]
fn truncate_noop_when_under_limit() {
    let mut output = "abc".to_string();
    truncate_to_byte_limit(&mut output, 10);
    assert_eq!(output, "abc");
}

#[cfg(unix)]
mod terminal {
    use std::sync::Arc;

    use futures::executor::block_on;
    use warpui::r#async::executor::Background;

    use super::*;

    fn delegate() -> LocalAcpDelegate {
        LocalAcpDelegate::new(
            Arc::new(Background::default()),
            Arc::new(AutoDenyPermissionResolver),
        )
    }

    #[test]
    fn terminal_runs_command_and_reports_exit() {
        let delegate = delegate();
        block_on(async {
            let create = delegate
                .create_terminal(schema::CreateTerminalRequest::new(
                    "test-session",
                    "/bin/sh",
                ))
                .await;
            // CreateTerminalRequest::new signature may differ; constructed in
            // the test to fail loudly if the schema changes shape.
            let terminal_id = create.expect("create_terminal failed").terminal_id;

            let exit = delegate
                .wait_for_terminal_exit(schema::WaitForTerminalExitRequest::new(
                    "test-session",
                    terminal_id.clone(),
                ))
                .await
                .expect("wait_for_exit failed");
            // /bin/sh with null stdin exits 0 immediately.
            assert_eq!(exit.exit_status.exit_code, Some(0));

            let released = delegate
                .release_terminal(schema::ReleaseTerminalRequest::new(
                    "test-session",
                    terminal_id,
                ))
                .await;
            assert!(released.is_ok());
        });
    }

    #[test]
    fn terminal_captures_output() {
        let delegate = delegate();
        block_on(async {
            let mut request = schema::CreateTerminalRequest::new("test-session", "/bin/echo");
            request.args = vec!["hello".to_string(), "acp".to_string()];
            let terminal_id = delegate
                .create_terminal(request)
                .await
                .expect("create_terminal failed")
                .terminal_id;

            delegate
                .wait_for_terminal_exit(schema::WaitForTerminalExitRequest::new(
                    "test-session",
                    terminal_id.clone(),
                ))
                .await
                .expect("wait_for_exit failed");

            // Output readers may still be draining just after exit; poll briefly.
            let mut output = String::new();
            for _ in 0..100 {
                let response = delegate
                    .terminal_output(schema::TerminalOutputRequest::new(
                        "test-session",
                        terminal_id.clone(),
                    ))
                    .await
                    .expect("terminal_output failed");
                output = response.output;
                if !output.is_empty() {
                    break;
                }
                warpui::r#async::Timer::after(std::time::Duration::from_millis(10)).await;
            }
            assert_eq!(output.trim_end(), "hello acp");
        });
    }

    #[test]
    fn unknown_terminal_errors() {
        let delegate = delegate();
        block_on(async {
            let result = delegate
                .terminal_output(schema::TerminalOutputRequest::new(
                    "test-session",
                    schema::TerminalId::new("nope"),
                ))
                .await;
            assert!(result.is_err());
        });
    }
}

mod fs {
    use std::sync::Arc;

    use futures::executor::block_on;
    use warpui::r#async::executor::Background;

    use super::*;

    fn delegate() -> LocalAcpDelegate {
        LocalAcpDelegate::new(
            Arc::new(Background::default()),
            Arc::new(AutoDenyPermissionResolver),
        )
    }

    #[test]
    fn write_then_read_round_trips() {
        let delegate = delegate();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("file.txt");
        block_on(async {
            delegate
                .write_text_file(schema::WriteTextFileRequest::new(
                    "test-session",
                    path.clone(),
                    "line1\nline2\nline3",
                ))
                .await
                .expect("write failed");

            let read = delegate
                .read_text_file(schema::ReadTextFileRequest::new("test-session", path))
                .await
                .expect("read failed");
            assert_eq!(read.content, "line1\nline2\nline3");
        });
    }

    #[test]
    fn read_applies_line_and_limit() {
        let delegate = delegate();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "a\nb\nc\nd").unwrap();
        block_on(async {
            let read = delegate
                .read_text_file(
                    schema::ReadTextFileRequest::new("test-session", path)
                        .line(2u32)
                        .limit(2u32),
                )
                .await
                .expect("read failed");
            assert_eq!(read.content, "b\nc");
        });
    }

    #[test]
    fn relative_paths_are_rejected() {
        let delegate = delegate();
        block_on(async {
            let read = delegate
                .read_text_file(schema::ReadTextFileRequest::new(
                    "test-session",
                    "relative/path.txt",
                ))
                .await;
            assert!(read.is_err());
        });
    }

    #[test]
    fn permission_defaults_to_cancelled() {
        let delegate = delegate();
        block_on(async {
            let response = delegate
                .request_permission(schema::RequestPermissionRequest::new(
                    "test-session",
                    schema::ToolCallUpdate::new("tool-1", schema::ToolCallUpdateFields::default()),
                    Vec::new(),
                ))
                .await
                .expect("request_permission failed");
            assert!(matches!(
                response.outcome,
                schema::RequestPermissionOutcome::Cancelled
            ));
        });
    }
}
