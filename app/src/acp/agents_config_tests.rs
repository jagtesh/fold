use super::*;

#[test]
fn parses_flat_map_of_agents() {
    let config = AcpAgentsConfig::parse(
        r#"{
            "claude": { "command": "claude-code-acp" },
            "gemini": {
                "command": "gemini",
                "args": ["--experimental-acp"],
                "env": { "GEMINI_API_KEY": "abc" },
                "working_directory": "/tmp"
            }
        }"#,
    )
    .unwrap();

    assert_eq!(config.agents.len(), 2);
    let claude = &config.agents["claude"];
    assert_eq!(claude.command, "claude-code-acp");
    assert!(claude.args.is_empty());

    let gemini = &config.agents["gemini"];
    assert_eq!(gemini.args, vec!["--experimental-acp"]);
    assert_eq!(gemini.env["GEMINI_API_KEY"], "abc");
    assert_eq!(gemini.working_directory.as_deref(), Some("/tmp"));
}

#[test]
fn rejects_unknown_fields() {
    let result = AcpAgentsConfig::parse(r#"{ "claude": { "command": "x", "url": "y" } }"#);
    assert!(result.is_err());
}

#[test]
fn empty_object_is_valid() {
    let config = AcpAgentsConfig::parse("{}").unwrap();
    assert!(config.agents.is_empty());
}

#[test]
fn user_config_overrides_presets_by_name() {
    let user = AcpAgentsConfig::parse(
        r#"{ "Claude Code": { "command": "/opt/custom/claude-code-acp" } }"#,
    )
    .unwrap();
    let merged = AcpAgentsConfig::with_presets(user);

    assert_eq!(
        merged.agents["Claude Code"].command,
        "/opt/custom/claude-code-acp"
    );
    // Presets not overridden are still present.
    assert!(merged.agents.contains_key("Gemini CLI"));
}

#[test]
fn round_trips_serialization() {
    let config = AcpAgentsConfig::builtin_presets();
    let json = serde_json::to_string(&config.agents).unwrap();
    let parsed = AcpAgentsConfig::parse(&json).unwrap();
    assert_eq!(parsed, config);
}
