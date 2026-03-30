r"""Define a script to test the performance of the haiku judge."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.timing import timeblock
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from argos.datasets import generate_haiku_dataset
from argos.metrics import compute_binary_classification_metrics
from argos.nodes import HaikuJudgeState, make_haiku_judge_node
from argos.nodes.haiku_judge import HAIKU_JUDGE_SYSTEM_PROMPT
from argos.utils.batching import batchify
from argos.utils.dataframe import concat_and_merge, summarize_boolean_columns
from argos.utils.logging import configure_logging

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

HAIKU_JUDGE_SYSTEM_PROMPT1 = """
You are a strict, adversarial haiku evaluator.

Your task is to evaluate whether a given haiku satisfies BOTH:
1) the 5-7-5 syllable structure
2) meaningful relevance to the target topic

You must follow the protocol exactly and return a structured result.

---

## Step 1 — Structure Check (`structure_passed`)

A haiku passes ONLY if:
- It has EXACTLY 3 lines
- Line 1 has 5 syllables
- Line 2 has 7 syllables
- Line 3 has 5 syllables

### Syllable Counting Rules (MANDATORY)
- Count syllables phonetically (not visually)
- Treat contractions as spoken (e.g., "don't" = 1, "fire" = 1 or 2 → choose most common pronunciation)
- Hyphenated words count as their spoken parts
- Proper nouns: use best-known pronunciation
- If uncertain, choose the MOST LIKELY spoken form — do NOT give benefit of the doubt

### Enforcement
- Count syllables for EACH line explicitly
- If ANY line deviates → `structure_passed = False`

---

## Step 2 — Topic Relevance (`topic_passed`)

A haiku passes ONLY if:
- The topic is clearly identifiable in the poem, AND
- The connection is specific, not vague or generic

### Strict Criteria
Mark `False` if:
- The topic is only implied weakly or ambiguously
- The poem could apply equally well to many unrelated topics
- The connection relies purely on abstract interpretation without textual evidence

Mark `True` only if:
- There is clear textual evidence (words, imagery, or concrete metaphor) tied to the topic
- A reasonable reader would confidently identify the topic without guessing

---

## Step 3 — Quality Score (`score`)

Rate from 1 to 10:

- 1-3: Broken, incoherent, or purely literal
- 4-6: Basic, predictable imagery
- 7-8: Clear imagery + some depth or juxtaposition
- 9-10: Precise, evocative, layered meaning

---

## Step 4 — Output Format (`reasoning`)

Return:

- `structure_passed`: boolean
- `topic_passed`: boolean
- `score`: integer (1-10)
- `reasoning`: MUST include:
  1. Syllable counts for each line (e.g., 5 / 6 / 5 → FAIL)
  2. Explicit justification for topic relevance (cite words or imagery)
  3. Brief quality assessment

Keep reasoning concise (2-4 sentences). No extra text.

---

## Important Constraints

- Be strict. Do NOT infer correctness.
- Do NOT reward “close enough” syllable counts.
- Do NOT assume topic relevance without explicit evidence.
- When in doubt → FAIL."""

HAIKU_JUDGE_SYSTEM_PROMPT2 = """You are an expert haiku judge. Given a haiku and a target topic, evaluate it and populate the structured output fields.

## structure_passed
Count syllables phonetically for each line.
Set True ONLY if the haiku has exactly 3 lines with counts of 5, 7, 5 — any deviation is False.

## topic_passed
Set True if the haiku clearly and meaningfully engages with the target topic, False otherwise.

## score (1-10)
1-3: Incoherent, purely literal, or no imagery
4-6: Functional but weak — generic word choice, flat imagery
7-8: Vivid, purposeful imagery with effective contrast or juxtaposition
9-10: Precise, resonant, and evocative — each word earns its place

## reasoning
2-3 sentences. Cover: (1) syllable count per line, (2) topic relevance, (3) justification for the score.
Do not restate the haiku."""

HAIKU_JUDGE_SYSTEM_PROMPT3 = """You are an expert haiku judge. Given a haiku and a target topic, evaluate it and populate the structured output fields.

## structure_passed
Verify structure using these steps — do NOT include this work in your output:

1. Count the lines. A haiku has exactly 3 lines.
2. For each line, count syllables phonetically:
   - Sound out each word aloud mentally
   - Count vowel sounds (not letters): "haiku" = hai-ku = 2, "silence" = si-lence = 2, "over" = o-ver = 2
   - Treat contractions as spoken: "don't" = 1, "I'm" = 1
   - Treat "-ed" endings by how they're spoken: "walked" = 1 syllable, "want-ed" = 2
3. Check the syllable pattern is exactly 5 / 7 / 5.

Set True ONLY if all three checks pass. Any deviation — wrong line count, wrong syllable count on any line — is False.

## topic_passed
Set True if the haiku clearly and meaningfully engages with the target topic, False otherwise.

## score (1-10)
1-3: Incoherent, purely literal, or no imagery
4-6: Functional but weak — generic word choice, flat imagery
7-8: Vivid, purposeful imagery with effective contrast or juxtaposition
9-10: Precise, resonant, and evocative — each word earns its place

## reasoning
2-3 sentences. Cover: (1) syllable count per line, (2) topic relevance, (3) justification for the score.
Do not restate the haiku."""

HAIKU_JUDGE_SYSTEM_PROMPT4 = """You are an expert haiku judge. Given a haiku and a target topic, evaluate it and populate the structured output fields.

## structure_passed
Verify structure using these steps — do NOT include this work in your output:

1. Count the lines. A haiku has exactly 3 lines.
2. For each line, count syllables phonetically:
   - Sound out each word aloud mentally
   - Count vowel sounds (not letters): "haiku" = hai-ku = 2, "silence" = si-lence = 2, "over" = o-ver = 2
   - Treat contractions as spoken: "don't" = 1, "I'm" = 1
   - Treat "-ed" endings by how they're spoken: "walked" = 1 syllable, "want-ed" = 2
3. Check the syllable pattern is exactly 5 / 7 / 5.

Set True ONLY if all three checks pass. Any deviation — wrong line count, wrong syllable count on any line — is False.

## topic_passed
Set True if the haiku clearly and meaningfully engages with the target topic, False otherwise.

## score (1-10)
1-3: Incoherent, purely literal, or no imagery
4-6: Functional but weak — generic word choice, flat imagery
7-8: Vivid, purposeful imagery with effective contrast or juxtaposition
9-10: Precise, resonant, and evocative — each word earns its place

## reasoning
2-3 sentences. Cover: (1) syllable count per line, (2) topic relevance, (3) justification for the score.
Do not restate the haiku.

---

## Example

**Topic:** autumn

**Haiku:**
  An old silent pond
  A frog jumps into the pond
  Splash! Silence again

**Verification (internal only — do not include in output):**
  Line 1 — "An old si-lent pond" → 1+1+2+1 = 5 ✓
  Line 2 — "A frog jumps in-to the pond" → 1+1+1+2+1+1 = 7 ✓
  Line 3 — "Splash! Si-lence a-gain" → 1+2+2 = 5 ✓
  Pattern: 5 / 7 / 5 ✓, Lines: 3 ✓

**Expected output:**
  structure_passed: True
  topic_passed: True        — the pond, frog, and silence evoke an autumn stillness
  score: 9                  — sparse, precise imagery; the juxtaposition of splash and silence is resonant
  reasoning: "All three lines hold their syllable counts (5/7/5). The imagery of the pond and frog
              implicitly captures autumn's quiet and transience without naming it directly. The
              contrast between the sudden splash and returning silence earns a 9 — every word
              is purposeful and the moment lands with emotional precision."
  passed: True              — derived: structure_passed ✓ AND topic_passed ✓ AND score ≥ 7 ✓"""


class State(HaikuJudgeState):
    r"""Define the state of the haiku generator-judge system."""


def create_graph(
    model: str = "ollama:gemma3:4b", judge_system_prompt: str = HAIKU_JUDGE_SYSTEM_PROMPT
) -> CompiledStateGraph:
    r"""Create the graph of the haiku generator-judge.

    Args:
        model: The model of the haiku generator-judge.
        judge_system_prompt: The prompt of the judge-system-prompt.

    Returns:
        The graph of the haiku generator-judge.
    """
    llm: BaseChatModel = init_chat_model(model=model, temperature=0, max_retries=999)
    model_version = getattr(llm, "model", getattr(llm, "model_name", "Unknown"))
    logger.info(
        f"class: {type(llm).__name__} | model: {model_version} | temperature: {llm.temperature}"
    )

    graph_builder = StateGraph(State)

    graph_builder.add_node("judge", make_haiku_judge_node(llm, system_prompt=judge_system_prompt))

    graph_builder.add_edge(START, "judge")
    graph_builder.add_edge("judge", END)

    # Compile the graph into a runnable app
    return graph_builder.compile()


def evaluate_metrics(results: pl.DataFrame) -> None:
    r"""Evaluate the metrics of the haiku generator-judge.

    Args:
        results: The results of the haiku generator-judge.
    """
    logger.info(
        f"\n{summarize_boolean_columns(results.select(['target', 'structure_target', 'topic_target']))}"
    )

    overall = compute_binary_classification_metrics(
        results, target_col="target", predict_col="passed"
    )
    logger.info(f"overall\n{overall.to_str()}")

    structure = compute_binary_classification_metrics(
        results, target_col="structure_target", predict_col="structure_passed"
    )
    logger.info(f"structure\n{structure.to_str()}")

    topic = compute_binary_classification_metrics(
        results, target_col="topic_target", predict_col="topic_passed"
    )
    logger.info(f"topic\n{topic.to_str()}")


def prepare_dataset() -> pl.DataFrame:
    r"""Prepare a dataset of haiku examples.

    Returns:
        A DataFrame containing haiku examples.
    """
    with timeblock(message="Dataset generation time: {time}"):
        dataset = generate_haiku_dataset()

    # uncomment this line to sample a smaller version of the dataset.
    # dataset = dataset.sample(n=5, seed=42)
    with pl.Config(tbl_cols=-1, tbl_rows=10):
        logger.info(f"\n{dataset}")

    stats = summarize_boolean_columns(
        dataset.select(["target", "structure_target", "topic_target"])
    )
    logger.info(f"statistics about the dataset\n{stats}")
    return dataset


def prepare_results(dataset: pl.DataFrame, outputs: list[dict[Any, Any]]) -> pl.DataFrame:
    r"""Prepare results of haiku generator-judge.

    Args:
        dataset: The dataset of haiku examples.
        outputs: The results of the haiku generator-judge.

    Returns:
        The results of the haiku generator-judge in a DataFrame.
    """
    cols = [
        "topic",
        "haiku",
        "score",
        "passed",
        "target",
        "structure_passed",
        "structure_target",
        "topic_passed",
        "topic_target",
        "reasoning",
    ]
    flat_data = [
        {**{k: v for k, v in row.items() if k != "evaluation"}, **row["evaluation"].model_dump()}
        for row in outputs
    ]
    return concat_and_merge(pl.DataFrame(flat_data), dataset).select(cols)


def run_inference(model: str, judge_system_prompt: str, path_results: Path) -> pl.DataFrame:
    r"""Run inference and store the results in a parquet file.

    Args:
        model: The name of the model to run inference.
        judge_system_prompt: The prompt of the judge-system-prompt.
        path_results: The path of the parquet file to store the results.

    Returns:
        The DataFrame containing the results of the inference.
    """
    graph = create_graph(model=model, judge_system_prompt=judge_system_prompt)
    logger.info(f"\n{graph.get_graph().draw_ascii()}")

    dataset = prepare_dataset()

    outputs = []
    examples = list(dataset.iter_rows(named=True))
    batch_size = 20
    with timeblock(message="LLM inference time: {time}"):
        for index, batch in enumerate(batchify(examples, size=batch_size)):
            logger.info(f"--- Processing Batch {index + 1} ---")
            outputs.extend(graph.batch(batch, config={"max_concurrency": batch_size}))

    logger.info("Preparing results...")
    results = prepare_results(dataset, outputs)
    logger.info(f"Writing results ({results.shape}) in {path_results}")
    results.write_parquet(path_results)
    return results


def run_evaluation(model: str, judge_system_prompt: str) -> None:
    r"""Run haiku generator-judge evaluation.

    Args:
        model: The name of the model to run inference.
        judge_system_prompt: The prompt of the judge-system-prompt.
    """
    path_results = (
        Path(__file__)
        .resolve()
        .parent.parent.joinpath("results")
        .joinpath("haiku_judge")
        .joinpath(hashlib.sha256(bytes(str(judge_system_prompt), "utf-8")).hexdigest())
        .joinpath(model.replace(":", "_"))
        .joinpath("results.parquet")
    )
    path_results.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"model: {model}")
    logger.info(f"Judge system prompt:\n{judge_system_prompt}")

    if not path_results.is_file():
        logger.info(f"No results found at {path_results}")
        run_inference(
            model=model, judge_system_prompt=judge_system_prompt, path_results=path_results
        )

    logger.info(f"Reading results from {path_results}")
    results = pl.read_parquet(path_results)
    with pl.Config(tbl_cols=-1, tbl_rows=10):
        logger.info(f"\n{results}")

    # for row in results.iter_rows(named=True):
    #     if row["score"] < 7:
    #         logger.info(f"\n{row}")

    evaluate_metrics(results)


def main() -> None:
    r"""Define the main function to test the haiku judge system."""
    # model = "ollama:smollm:135m"
    # model = "ollama:gemma3:1b"
    # model = "anthropic:claude-haiku-4-5-20251001"
    # model = "anthropic:claude-sonnet-4-6"
    # model = "anthropic:claude-opus-4-6"

    models = [
        # "ollama:smollm:135m",
        # "ollama:gemma3:1b",
        # "ollama:gemma3:4b",
        # "ollama:gemma3:12b",
        # "anthropic:claude-haiku-4-5-20251001",
        # "anthropic:claude-sonnet-4-6",
        # "anthropic:claude-opus-4-6",
        # "google_genai:gemini-3.1-flash-lite-preview",
        "openai:gpt-5.4-nano",
    ]
    judge_system_prompts = [
        HAIKU_JUDGE_SYSTEM_PROMPT,
        # HAIKU_JUDGE_SYSTEM_PROMPT1,
        # HAIKU_JUDGE_SYSTEM_PROMPT2,
        # HAIKU_JUDGE_SYSTEM_PROMPT3,
        # HAIKU_JUDGE_SYSTEM_PROMPT4,
    ]

    for model in models:
        for judge_system_prompt in judge_system_prompts:
            run_evaluation(model=model, judge_system_prompt=judge_system_prompt)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    load_dotenv()

    main()
