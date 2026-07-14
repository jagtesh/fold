use acp::schema;

use super::*;

fn empty_conversation() -> AcpConversation {
    AcpConversation {
        connection: None,
        status: AcpConversationStatus::Idle,
        session_id: None,
        auth_methods: Vec::new(),
        items: Vec::new(),
    }
}

fn text_chunk(text: &str) -> schema::ContentChunk {
    schema::ContentChunk::new(schema::ContentBlock::Text(schema::TextContent::new(text)))
}

#[test]
fn agent_message_chunks_accumulate_into_one_item() {
    let mut conversation = empty_conversation();
    conversation.apply_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("Hel")));
    conversation.apply_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("lo")));

    assert_eq!(
        conversation.items(),
        &[AcpConversationItem::AgentMessage {
            markdown: "Hello".to_string()
        }]
    );
}

#[test]
fn thought_and_message_chunks_stay_separate_items() {
    let mut conversation = empty_conversation();
    conversation.apply_update(schema::SessionUpdate::AgentThoughtChunk(text_chunk(
        "thinking",
    )));
    conversation.apply_update(schema::SessionUpdate::AgentMessageChunk(text_chunk(
        "answer",
    )));
    conversation.apply_update(schema::SessionUpdate::AgentThoughtChunk(text_chunk(
        " more",
    )));

    assert_eq!(conversation.items().len(), 3);
    assert!(matches!(
        &conversation.items()[2],
        AcpConversationItem::AgentThought { markdown } if markdown == " more"
    ));
}

#[test]
fn tool_call_update_merges_fields_by_id() {
    let mut conversation = empty_conversation();
    conversation.apply_update(schema::SessionUpdate::ToolCall(schema::ToolCall::new(
        "tc-1",
        "Running ls",
    )));

    let mut fields = schema::ToolCallUpdateFields::default();
    fields.status = Some(schema::ToolCallStatus::Completed);
    fields.title = Some("Ran ls".to_string());
    conversation.apply_update(schema::SessionUpdate::ToolCallUpdate(
        schema::ToolCallUpdate::new("tc-1", fields),
    ));

    let AcpConversationItem::ToolCall(tool_call) = &conversation.items()[0] else {
        panic!("expected tool call item");
    };
    assert_eq!(tool_call.title, "Ran ls");
    assert_eq!(tool_call.status, schema::ToolCallStatus::Completed);
}

#[test]
fn tool_call_update_for_unknown_id_is_ignored() {
    let mut conversation = empty_conversation();
    conversation.apply_update(schema::SessionUpdate::ToolCallUpdate(
        schema::ToolCallUpdate::new("nope", Default::default()),
    ));
    assert!(conversation.items().is_empty());
}

#[test]
fn plan_updates_replace_previous_plan() {
    let mut conversation = empty_conversation();
    let plan_one = schema::Plan::new(vec![]);
    conversation.apply_update(schema::SessionUpdate::Plan(plan_one));
    conversation.apply_update(schema::SessionUpdate::AgentMessageChunk(text_chunk("hi")));

    let plan_two = schema::Plan::new(vec![schema::PlanEntry::new(
        "step 1",
        schema::PlanEntryPriority::Medium,
        schema::PlanEntryStatus::Pending,
    )]);
    conversation.apply_update(schema::SessionUpdate::Plan(plan_two.clone()));

    // Still two items: the plan was replaced in place, not appended.
    assert_eq!(conversation.items().len(), 2);
    assert!(matches!(
        &conversation.items()[0],
        AcpConversationItem::Plan(plan) if *plan == plan_two
    ));
}

#[test]
fn user_message_chunks_accumulate() {
    let mut conversation = empty_conversation();
    conversation.apply_update(schema::SessionUpdate::UserMessageChunk(text_chunk("a")));
    conversation.apply_update(schema::SessionUpdate::UserMessageChunk(text_chunk("b")));
    assert_eq!(
        conversation.items(),
        &[AcpConversationItem::UserMessage {
            markdown: "ab".to_string()
        }]
    );
}
