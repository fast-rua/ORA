"""Read-only evaluation of a finished run, its seal and export, and the evidence checks behind them.

``evaluate.evaluate`` reads one run directory and writes ``evaluation.json``;
``evaluate.seal`` and ``evaluate.export`` seal and archive it. The checks it
runs live beside it (``checks``, ``learning``, ``lossless``, ``play``,
``receipts``, ``verdict_guard``), as do the derived records a launch writes
(``metrics``, ``context_audit``, ``context_manifest``, ``ownership_audit``,
``visual_usage``) and ``revalidation`` (``ora revalidate``).
"""
