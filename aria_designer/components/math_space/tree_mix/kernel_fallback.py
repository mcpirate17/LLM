"""Kernel handler for tree_mix — delegates to research.mathspaces.tree_mix."""

from aria_designer.runtime.fallback_templates import make_mathspace_handler

ComponentHandler = make_mathspace_handler(
    "tree_mix",
    "research.mathspaces.tree_mix.execute_tree_mix",
    arity=2,
)
