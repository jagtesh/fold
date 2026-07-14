use std::sync::Arc;

use acp::AcpAgentConnection;
use command::r#async::Command;
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::StreamExt as _;
use warp_multi_agent_api::client_action::Action;
use warp_multi_agent_api::response_event::Type;
use warpui::r#async::executor::Background;

use super::super::delegate::{AutoDenyPermissionResolver, LocalAcpDelegate};
use super::*;

/// A scripted fake ACP agent: replies to the first request (the prompt,
/// request id 1) with one streamed text chunk and an end_turn result.
#[cfg(unix)]
fn fake_agent_command() -> Command {
    let script = concat!(
        "read line; ",
        r#"printf '%s\n' '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hi"}}}}'; "#,
        r#"printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"stopReason":"end_turn"}}'"#,
    );
    let mut command = Command::new("sh");
    command.args(["-c", script]);
    command
}

#[test]
#[cfg(unix)]
fn driver_streams_init_updates_and_finished_from_fake_agent() {
    let executor = Arc::new(Background::default());
    let delegate = Arc::new(LocalAcpDelegate::new(
        executor.clone(),
        Arc::new(AutoDenyPermissionResolver),
    ));
    let connection = Arc::new(
        AcpAgentConnection::spawn(fake_agent_command(), delegate, executor, None)
            .expect("failed to spawn fake agent"),
    );

    let (_cancel_tx, cancel_rx) = oneshot::channel();
    let stream = acp_response_stream(
        connection,
        schema::SessionId::new("s1"),
        vec![schema::ContentBlock::Text(schema::TextContent::new("go"))],
        AcpStreamIds {
            conversation_id: "conv-1".to_string(),
            task_id: "task-1".to_string(),
            request_id: "req-1".to_string(),
        },
        cancel_rx,
    );

    let events: Vec<_> = block_on(stream.collect());
    assert_eq!(events.len(), 3, "init + one update + finished: {events:?}");

    let Ok(init) = &events[0] else {
        panic!("expected init event");
    };
    let Some(Type::Init(init)) = &init.r#type else {
        panic!("expected StreamInit, got {init:?}");
    };
    assert_eq!(init.conversation_id, "conv-1");

    let Ok(update) = &events[1] else {
        panic!("expected client actions event");
    };
    let Some(Type::ClientActions(actions)) = &update.r#type else {
        panic!("expected ClientActions, got {update:?}");
    };
    let Some(Action::AddMessagesToTask(add)) = &actions.actions[0].action else {
        panic!("expected AddMessagesToTask");
    };
    let Some(warp_multi_agent_api::message::Message::AgentOutput(output)) =
        &add.messages[0].message
    else {
        panic!("expected AgentOutput message");
    };
    assert_eq!(output.text, "hi");

    let Ok(finished) = &events[2] else {
        panic!("expected finished event");
    };
    let Some(Type::Finished(finished)) = &finished.r#type else {
        panic!("expected StreamFinished, got {finished:?}");
    };
    use warp_multi_agent_api::response_event::stream_finished::Reason;
    assert!(matches!(finished.reason, Some(Reason::Done(_))));
}

/// Updates for other sessions on the same connection must not leak into
/// this turn's stream.
#[test]
#[cfg(unix)]
fn driver_filters_updates_from_other_sessions() {
    let script = concat!(
        "read line; ",
        r#"printf '%s\n' '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"other","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"leak"}}}}'; "#,
        r#"printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"stopReason":"end_turn"}}'"#,
    );
    let mut command = Command::new("sh");
    command.args(["-c", script]);

    let executor = Arc::new(Background::default());
    let delegate = Arc::new(LocalAcpDelegate::new(
        executor.clone(),
        Arc::new(AutoDenyPermissionResolver),
    ));
    let connection = Arc::new(
        AcpAgentConnection::spawn(command, delegate, executor, None)
            .expect("failed to spawn fake agent"),
    );

    let (_cancel_tx, cancel_rx) = oneshot::channel();
    let stream = acp_response_stream(
        connection,
        schema::SessionId::new("s1"),
        vec![schema::ContentBlock::Text(schema::TextContent::new("go"))],
        AcpStreamIds {
            conversation_id: "conv-1".to_string(),
            task_id: "task-1".to_string(),
            request_id: "req-1".to_string(),
        },
        cancel_rx,
    );

    let events: Vec<_> = block_on(stream.collect());
    // Just init + finished; the other session's update is filtered out.
    assert_eq!(events.len(), 2, "unexpected events: {events:?}");
    assert!(matches!(
        events[0].as_ref().map(|e| &e.r#type),
        Ok(Some(Type::Init(_)))
    ));
    assert!(matches!(
        events[1].as_ref().map(|e| &e.r#type),
        Ok(Some(Type::Finished(_)))
    ));
}
