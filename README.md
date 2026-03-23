# research-graph-agents

Research agents implemented with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph).

## Overview

`argos` is an experimental library that provides reusable LangGraph nodes, state definitions,
and utility helpers for building LLM-powered graph agents.
It is designed for research and experimentation with multi-step AI pipelines.

## Features

- **Haiku generator node** – prompts an LLM to compose a haiku for a given topic.
- **Haiku judge node** – evaluates a haiku using structured output (syllable counts, topic fidelity, quality score).
- **Logging utilities** – colored console logging via optional `colorlog` integration.
- **Date utilities** – timezone-aware current-date helper.

## Installation

```bash
pip install argos
```

To enable colored logging output, install the optional `colorlog` extra:

```bash
pip install "argos[colorlog]"
```

## Quick start

```python
from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from argos.nodes import HaikuJudgeState, make_haiku_generator_node, make_haiku_judge_node

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

## Requirements

- Python 3.11+
- LangChain / LangGraph
- An Ollama server (for the bundled examples) or any other supported LLM provider
