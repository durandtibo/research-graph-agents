r"""Contain the autoprompt task for haiku judge optimization.

This sub-package implements an iterative prompt-optimization pipeline
that evaluates a haiku judge LLM against a labeled dataset, analyzes
prediction errors, and generates improved system prompts.

Key modules:

- :mod:`~argos.tasks.autoprompt.config` — dataclasses for experiment
  and LLM configuration.
- :mod:`~argos.tasks.autoprompt.dataset` — dataset preparation helpers.
- :mod:`~argos.tasks.autoprompt.judge` — graph construction for the
  haiku judge inference pipeline.
- :mod:`~argos.tasks.autoprompt.predictor` — batch-inference predictor.
- :mod:`~argos.tasks.autoprompt.evaluation` — metric computation.
- :mod:`~argos.tasks.autoprompt.evaluator` — high-level evaluator class.
- :mod:`~argos.tasks.autoprompt.analysis` — prediction error analysis.
- :mod:`~argos.tasks.autoprompt.history` — persistent experiment history.
- :mod:`~argos.tasks.autoprompt.inference` — end-to-end inference pipeline.
"""
