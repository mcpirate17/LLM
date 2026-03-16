# Aria Audit Prompt 01 — Alias Collapse and Canonical Taxonomy

You are auditing the Aria component catalog to collapse aliases, wrappers, stubs, and taxonomy inflation into a canonical component set.

Use the current `component_catalog.csv` as ground truth input, especially these fields when present:
- `name`
- `category`
- `component_type`
- `description`
- `aria_designer_aliases`
- `maps_to_primitive`
- `template_weight`
- `byte_safe`

## Goal
Determine the real primitive count and produce a canonical component taxonomy that Aria should use going forward.

## Required questions
1. Which components are exact aliases?
2. Which map to the same primitive?
3. Which are thin wrappers around the same operation?
4. Which are designer stubs or non-functional placeholders?
5. Which names oversell the implementation?
6. Which components should be collapsed into a single canonical primitive?
7. Which catalog entries should be removed entirely from search?

## Required method
- Use `maps_to_primitive` and `aria_designer_aliases` as hints, not proof.
- Inspect code and manifests for the suspect families.
- Separate:
  - true primitives
  - aliases
  - wrappers
  - templates / blocks
  - stubs / dead entries
- Identify where multiple names point to the same runtime behavior.

## Deliverables
1. Canonical primitive list
2. Alias-to-canonical mapping table
3. Wrapper/stub/dead-entry list
4. Components to delete from search space
5. Components to rename for honesty
6. Recommended updated taxonomy by category
7. Estimated real search-space size after cleanup

## Rules
- Do not count a renamed copy as a distinct component.
- Do not count a manifest-only stub as a real component.
- Do not give credit for catalog breadth if the implementations collapse to the same primitive.
- Be blunt about fake richness.

## Output format
1. Executive summary
2. Canonical taxonomy
3. Alias and duplicate map
4. Stubs / wrappers / dead entries
5. Search-space cleanup actions
6. Final primitive count
