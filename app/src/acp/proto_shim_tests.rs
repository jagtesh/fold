use acp::schema;
use warp_multi_agent_api::client_action::Action;
use warp_multi_agent_api::response_event::Type;

use super::*;

fn shim() -> AcpProtoShim {
    AcpProtoShim::new("task-1".to_string(), "req-1".to_string())
}

fn text_chunk(text: &str) -> schema::ContentChunk {
    schema::ContentChunk::new(schema::ContentBlock::Text(schema::TextContent::new(text)))
}

fn actions_of(event: api::ResponseEvent) -> Vec<api::ClientAction> {
    match event.r#type {
        Some(Type::ClientActions(actions)) => actions.actions,
        other => panic!("expected ClientActions, got {other:?}"),
    }
}

#[test]
fn stream_init_carries_conversation_and_request_ids() {
    let event = shim().stream_init("conv-1".to_string());
    let Some(Type::Init(init)) = event.r#type else {
        panic!("expected Init");
    };
    assert_eq!(init.conversation_id, "conv-1");
    assert_eq!(init.request_id, "req-1");
}

#[test]
fn first_text_chunk_adds_message_subsequent_chunks_append() {
    let mut shim = shim();

    let first = shim
        .map_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("Hel")))
        .expect("mapped");
    let actions = actions_of(first);
    let Some(Action::AddMessagesToTask(add)) = &actions[0].action else {
        panic!("expected AddMessagesToTask, got {:?}", actions[0].action);
    };
    assert_eq!(add.task_id, "task-1");
    let message = &add.messages[0];
    let Some(api::message::Message::AgentOutput(output)) = &message.message else {
        panic!("expected AgentOutput");
    };
    assert_eq!(output.text, "Hel");
    let first_message_id = message.id.clone();

    let second = shim
        .map_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("lo")))
        .expect("mapped");
    let actions = actions_of(second);
    let Some(Action::AppendToMessageContent(append)) = &actions[0].action else {
        panic!(
            "expected AppendToMessageContent, got {:?}",
            actions[0].action
        );
    };
    let message = append.message.as_ref().unwrap();
    assert_eq!(message.id, first_message_id);
    assert_eq!(
        append.mask.as_ref().unwrap().paths,
        vec!["message.agent_output.text".to_string()]
    );
}

#[test]
fn switching_between_text_and_reasoning_starts_new_messages() {
    let mut shim = shim();
    shim.map_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("a")))
        .expect("mapped");

    let reasoning = shim
        .map_update(schema::SessionUpdate::AgentThoughtChunk(text_chunk("hmm")))
        .expect("mapped");
    let actions = actions_of(reasoning);
    let Some(Action::AddMessagesToTask(add)) = &actions[0].action else {
        panic!("reasoning after text should start a new message");
    };
    let Some(api::message::Message::AgentReasoning(reasoning)) = &add.messages[0].message else {
        panic!("expected AgentReasoning");
    };
    assert_eq!(reasoning.reasoning, "hmm");

    // And returning to text starts another new message.
    let text_again = shim
        .map_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("b")))
        .expect("mapped");
    let actions = actions_of(text_again);
    assert!(matches!(
        &actions[0].action,
        Some(Action::AddMessagesToTask(_))
    ));
}

#[test]
fn tool_call_becomes_display_only_server_tool() {
    let mut shim = shim();
    let tool_call = schema::ToolCall::new("tc-9", "Reading files");
    let event = shim
        .map_update(schema::SessionUpdate::ToolCall(tool_call))
        .expect("mapped");
    let actions = actions_of(event);
    let Some(Action::AddMessagesToTask(add)) = &actions[0].action else {
        panic!("expected AddMessagesToTask");
    };
    let Some(api::message::Message::ToolCall(tool_call)) = &add.messages[0].message else {
        panic!("expected ToolCall message");
    };
    assert_eq!(tool_call.tool_call_id, "tc-9");
    let Some(api::message::tool_call::Tool::Server(server)) = &tool_call.tool else {
        panic!("expected display-only Server tool");
    };
    assert!(server.payload.contains("Reading files"));
}

#[test]
fn tool_call_update_replaces_message_with_mask() {
    let mut shim = shim();
    shim.map_update(schema::SessionUpdate::ToolCall(schema::ToolCall::new(
        "tc-9", "Running",
    )))
    .expect("mapped");

    let event = shim
        .map_update(schema::SessionUpdate::ToolCallUpdate(
            schema::ToolCallUpdate::new("tc-9", Default::default()),
        ))
        .expect("mapped");
    let actions = actions_of(event);
    let Some(Action::UpdateTaskMessage(update)) = &actions[0].action else {
        panic!("expected UpdateTaskMessage");
    };
    assert_eq!(update.message.as_ref().unwrap().id, "tc-9");
    assert_eq!(
        update.mask.as_ref().unwrap().paths,
        vec!["message.tool_call".to_string()]
    );
}

#[test]
fn interleaved_tool_call_splits_streaming_text() {
    let mut shim = shim();
    shim.map_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("a")))
        .expect("mapped");
    shim.map_update(schema::SessionUpdate::ToolCall(schema::ToolCall::new(
        "tc-1", "Tool",
    )))
    .expect("mapped");

    // Text after a tool call must start a fresh message, not append to the
    // pre-tool-call one.
    let event = shim
        .map_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("b")))
        .expect("mapped");
    let actions = actions_of(event);
    assert!(matches!(
        &actions[0].action,
        Some(Action::AddMessagesToTask(_))
    ));
}

#[test]
fn unmapped_updates_return_none() {
    let mut shim = shim();
    assert!(shim
        .map_update(schema::SessionUpdate::Plan(schema::Plan::new(vec![])))
        .is_none());
    assert!(shim
        .map_update(schema::SessionUpdate::UserMessageChunk(text_chunk("x")))
        .is_none());
}

#[test]
fn stream_finished_maps_stop_reasons() {
    use warp_multi_agent_api::response_event::stream_finished::Reason;

    let shim = shim();
    let done = shim.stream_finished(schema::StopReason::EndTurn);
    let Some(Type::Finished(finished)) = done.r#type else {
        panic!("expected Finished");
    };
    assert!(matches!(finished.reason, Some(Reason::Done(_))));

    let max = shim.stream_finished(schema::StopReason::MaxTokens);
    let Some(Type::Finished(finished)) = max.r#type else {
        panic!("expected Finished");
    };
    assert!(matches!(finished.reason, Some(Reason::MaxTokenLimit(_))));
}
