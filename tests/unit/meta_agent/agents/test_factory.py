"""Unit tests for BaseAgentFactory."""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from argos.meta_agent.agents import Agent, AgentConfig, BaseAgent, BaseAgentFactory


class UpperCaseAgentFactory(BaseAgentFactory):
    """Concrete factory that always returns an Agent wrapping
    str.upper."""

    def create(self, config: AgentConfig) -> BaseAgent:  # noqa: ARG002
        return Agent(RunnableLambda(str.upper))


class EchoAgentFactory(BaseAgentFactory):
    """Concrete factory that uses a component from config to select a
    transform."""

    def create(self, config: AgentConfig) -> BaseAgent:
        transform = config.components.get("transform", str)
        return Agent(RunnableLambda(transform))


#####################################
#     Tests for BaseAgentFactory    #
#####################################


def test_base_agent_factory_concrete_can_be_instantiated() -> None:
    factory = UpperCaseAgentFactory()
    assert isinstance(factory, BaseAgentFactory)


def test_base_agent_factory_create_returns_agent() -> None:
    factory = UpperCaseAgentFactory()
    agent = factory.create(AgentConfig(components={}))
    assert isinstance(agent, Agent)


def test_base_agent_factory_create_agent_produces_correct_predictions() -> None:
    factory = UpperCaseAgentFactory()
    agent = factory.create(AgentConfig(components={}))
    assert agent.predict(["hello", "world"]) == ["HELLO", "WORLD"]


def test_base_agent_factory_create_uses_config_components() -> None:
    factory = EchoAgentFactory()
    agent = factory.create(AgentConfig(components={"transform": str.upper}))
    assert agent.predict(["hello"]) == ["HELLO"]


def test_base_agent_factory_create_can_be_called_multiple_times() -> None:
    factory = UpperCaseAgentFactory()
    config = AgentConfig(components={})
    agent1 = factory.create(config)
    agent2 = factory.create(config)
    assert isinstance(agent1, Agent)
    assert isinstance(agent2, Agent)
    assert agent1 is not agent2


def test_base_agent_factory_create_ignores_metadata() -> None:
    factory = UpperCaseAgentFactory()
    config = AgentConfig(components={}, metadata={"version": "1.0"})
    agent = factory.create(config)
    assert agent.predict(["test"]) == ["TEST"]
