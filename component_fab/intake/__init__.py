"""Intake — scope existing primitives and templates into the fab catalog."""

from .model_components import (
    ComponentSite,
    find_component_sites,
    replaceable_component_paths,
)
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
    "ComponentSite",
    "ComponentRecord",
    "classify_op_row",
    "classify_template_row",
    "find_component_sites",
    "load_op_rows",
    "load_template_rows",
    "replaceable_component_paths",
    "scope_all",
    "select_underperforming_novel",
]
