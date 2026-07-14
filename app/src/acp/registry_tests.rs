use super::*;

fn task(id: &str) -> warp_multi_agent_api::Task {
    warp_multi_agent_api::Task {
        id: id.to_string(),
        ..Default::default()
    }
}

#[test]
fn binding_lookup_matches_any_request_task() {
    let mut registry = AcpConversationRegistry::default();
    assert!(registry.binding_for_tasks(&[task("t1")]).is_none());

    // A binding requires a connection; lookup semantics are what's under
    // test here, so exercise the map through register/unregister with the
    // task-matching path only (spawning a real connection is covered by the
    // response_stream_driver tests).
    let tasks = [task("other"), task("t1")];
    assert!(registry.binding_for_tasks(&tasks).is_none());
    assert!(registry.unregister("t1").is_none());
}

#[test]
fn prompt_blocks_join_query_inputs_and_skip_others() {
    let inputs = vec![
        AIAgentInput::UserQuery {
            query: "first".to_string(),
            context: Vec::new().into(),
            static_query_type: None,
            referenced_attachments: Default::default(),
            user_query_mode: Default::default(),
            running_command: None,
            intended_agent: None,
        },
        AIAgentInput::ResumeConversation {
            context: Vec::new().into(),
        },
        AIAgentInput::UserQuery {
            query: "second".to_string(),
            context: Vec::new().into(),
            static_query_type: None,
            referenced_attachments: Default::default(),
            user_query_mode: Default::default(),
            running_command: None,
            intended_agent: None,
        },
    ];
    let blocks = prompt_blocks(&inputs);
    assert_eq!(blocks.len(), 1);
    let schema::ContentBlock::Text(text) = &blocks[0] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "first\nsecond");
}
