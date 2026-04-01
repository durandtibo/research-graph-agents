r"""Provide utility functions for the argos package.

This sub-package includes a collection of helpers organised by concern:

- :mod:`~argos.utils.batching`: split sequences into fixed-size
  batches with :func:`~argos.utils.batching.batchify`.
- :mod:`~argos.utils.dataframe`: Polars DataFrame helpers such as
  :func:`~argos.utils.dataframe.concat_and_merge` and
  :func:`~argos.utils.dataframe.summarize_boolean_columns`.
- :mod:`~argos.utils.imports`: check for optional dependencies
  (e.g. ``colorlog``) and raise informative errors when they are
  missing.
- :mod:`~argos.utils.io`: read and write JSONL files with
  :func:`~argos.utils.io.read_jsonl_in_batches` and
  :func:`~argos.utils.io.write_jsonl`.
- :mod:`~argos.utils.logging`: configure the standard Python logging
  library, optionally with colored output via ``colorlog``.
- :mod:`~argos.utils.text`: text processing helpers including
  :func:`~argos.utils.text.count_lines`,
  :func:`~argos.utils.text.count_syllables`, and
  :func:`~argos.utils.text.remove_empty_lines`.
- :mod:`~argos.utils.today`: timezone-aware date helpers such as
  :func:`~argos.utils.today.get_today_date`.
"""
