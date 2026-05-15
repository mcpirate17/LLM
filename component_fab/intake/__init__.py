"""Intake — scope existing primitives and templates into the fab catalog."""

from .scope_existing import (
    CATEGORY_COMPRESSION,
    CATEGORY_LANE,
    CATEGORY_ROUTING,
    ComponentRecord,
    classify_op_row,
    classify_template_row,
    load_op_rows,
    load_template_rows,
    scope_all,
    select_underperforming_novel,
)

__all__ = [
    "CATEGORY_COMPRESSION",
    "CATEGORY_LANE",
    "CATEGORY_ROUTING",
    "ComponentRecord",
    "classify_op_row",
    "classify_template_row",
    "load_op_rows",
    "load_template_rows",
    "scope_all",
    "select_underperforming_novel",
]
