"""Standalone visual explainer for component_fab lanes.

A small FastAPI + Plotly single-page app that *shows* what a lane does:
token-mixing influence maps, the surprise-memory state filling token by token,
the learnable-semiring read sliding on the mean<->max axis, plus ledger replay
and live grading over SSE. Launch with ``python -m component_fab.viz``.
"""
