from __future__ import annotations

import pytest

from argos.meta_agent.agents import AgentConfig

#################################
#     Tests for AgentConfig     #
#################################


def test_agent_config_components_and_metadata_are_set() -> None:
    components = {"llm": "gpt-4", "memory": "redis"}
    metadata = {"version": "1.0", "author": "alice"}
    config = AgentConfig(components=components, metadata=metadata)
    assert config.components == components
    assert config.metadata == metadata


def test_agent_config_metadata_defaults_to_empty_dict() -> None:
    config = AgentConfig(components={})
    assert config.metadata is None


def test_agent_config_components_is_required() -> None:
    with pytest.raises(TypeError):
        AgentConfig()


def test_agent_config_components_accepts_arbitrary_values() -> None:
    components = {"llm": 42, "memory": [1, 2, 3], "tool": {"nested": True}}
    config = AgentConfig(components=components)
    assert config.components == components


def test_agent_config_metadata_accepts_arbitrary_values() -> None:
    metadata = {"version": 1, "tags": ["prod", "v2"], "active": True}
    config = AgentConfig(components={}, metadata=metadata)
    assert config.metadata == metadata


def test_agent_config_components_can_be_empty() -> None:
    config = AgentConfig(components={})
    assert config.components == {}


def test_agent_config_repr() -> None:
    config = AgentConfig(components={"model": "gpt-4o", "prompt": "You are a helpful assistant."})
    assert repr(config) == (
        "AgentConfig(components={'model': 'gpt-4o', 'prompt': 'You are a helpful assistant.'}, metadata=None)"
    )


def test_agent_config_repr_empty() -> None:
    assert repr(AgentConfig(components={})) == "AgentConfig(components={}, metadata=None)"


def test_agent_config_equality() -> None:
    assert AgentConfig(components={"llm": "gpt-4"}) == AgentConfig(components={"llm": "gpt-4"})


@pytest.mark.parametrize(
    ("cfg1", "cfg2"),
    [
        pytest.param(
            AgentConfig(components={"llm": "gpt-4"}),
            AgentConfig(components={"llm": "claude"}),
            id="different_components",
        ),
        pytest.param(
            AgentConfig(components={}, metadata={"version": "1.0"}),
            AgentConfig(components={}, metadata={"version": "2.0"}),
            id="different_metadata",
        ),
    ],
)
def test_agent_config_inequality(cfg1: AgentConfig, cfg2: AgentConfig) -> None:
    assert cfg1 != cfg2
