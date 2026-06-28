# component_fab.runner

The runner package keeps CLI entry points thin and splits orchestration by responsibility:

- `cycle.py`: one autonomous propose/select/grade/promote cycle and human-readable cycle summaries.
- `selection.py`: terminal filtering, measured/NAS screening, quality ordering, and surrogate acquisition selection.
- `grading.py`: autonomous per-spec grading, grade metadata assembly, survivor buffering, niche metadata, and ledger writes.
- `promotion.py`: ARIA registration side effects for newly promoted autonomous components.
- `invention.py`: invention-track grading, optional TinyLM hard-binding comparison, invention ledger metadata, and invention promotion policy.

`component_fab.tools.run_autonomous` should stay CLI-only: argument parsing, signal handling, rotation, loop driving, and report emission.

`component_fab.tools.run_invention` should stay CLI-only: argument parsing, gate enumeration, report output, and result printing.
