# research-graph-agents

Research agents implemented with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph).

## Overview

`argos` is an experimental library that provides reusable LangGraph nodes, state definitions,
and utility helpers for building LLM-powered graph agents.
It is designed for research and experimentation with multi-step AI pipelines.

## Quick start

```python
from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from argos.nodes import (
    HaikuJudgeState,
    make_haiku_generator_node,
    make_haiku_judge_node,
)

llm = ChatOllama(model="gemma3:1b")

graph_builder = StateGraph(HaikuJudgeState)
graph_builder.add_node("poet", make_haiku_generator_node(llm))
graph_builder.add_node("judge", make_haiku_judge_node(llm))
graph_builder.add_edge(START, "poet")
graph_builder.add_edge("poet", "judge")
graph_builder.add_edge("judge", END)

graph = graph_builder.compile()
result = graph.invoke({"topic": "the ocean"})
print(result["haiku"])
print(result["evaluation"])
```

## Examples

See the [`examples/`](examples/) directory for runnable scripts:

- [`haiku_llm_judge.py`](examples/haiku_llm_judge.py) – end-to-end haiku generation and evaluation using a local Ollama model.
