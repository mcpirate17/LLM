# Canonical Component Unification Multi-Agent Plan

## Goal

Eliminate mixed component identity schemes between research, leaderboard/import, designer API, Ask Aria, and saved workflows.

After this work:
- every persisted workflow uses one canonical component ID format
- every import/export path normalizes to that format
- aliases exist in exactly one place
- validation rejects unresolved IDs early with precise errors
- dead compatibility code is removed after cutover

This plan follows [GLOBAL_DEV_PROMPT.md](/home/tim/Projects/LLM/GLOBAL_DEV_PROMPT.md):
- no duplicate mapping logic
- no dead compatibility paths left behind
- no speculative abstraction layers
- small modules with clear ownership
- explicit deletion tasks after cutover

## Canonical Rule

Canonical component identity must be the live designer registry ID from the component DB.

Rules:
- `workflow.nodes[*].component_type` must always be canonical
- research-only names may exist only in metadata fields such as `metadata.research_op`
- aliases and legacy names must never be persisted as `component_type`
- validator must operate only on canonical IDs

## Current Failure Modes

- leaderboard/import emits mixed IDs like `io/input`, `math_space/ultrametric_attention`, `difficulty_scorer`
- Ask Aria and import normalization use different ad hoc maps
- some research/result graphs refer to ops not present in the current designer registry
- save/load/validate operate on exact registry lookups, so partial normalization breaks workflows
- proxy/lifecycle failures hide the real workflow issues behind `502`

## Required Output

Deliver these artifacts:
- one canonical mapping module
- one normalization pass used by all boundaries
- importer cutover to canonical output only
- Ask Aria/chat cutover to canonical output only
- validation hard gate for unresolved IDs
- round-trip tests across research -> import -> validate -> save -> load
- deletion of superseded maps and compatibility branches

## Agent Split

### Agent 1: Canonical Mapping Core

Ownership:
- `aria_designer/api/app/component_identity.py` new
- `aria_designer/api/app/database.py` read-only integration points
- `aria_designer/api/app/routers/components.py` if needed for lookup helpers

Responsibilities:
- define canonical ID contract
- implement one alias table only
- implement:
  - `canonicalize_component_id(raw_id)`
  - `canonicalize_workflow_ids(workflow)`
  - `collect_unresolved_component_ids(workflow)`
- support namespaced forms like `io/input` only through normalization, not persistence
- preserve original raw IDs in metadata when useful for debugging

Constraints:
- pure logic only
- no DB writes
- no duplicate mapping copies elsewhere

Deletion targets after cutover:
- scattered alias maps outside this module
- any fallback resolver duplicated in chat/import/validation

### Agent 2: Import and Leaderboard Normalization

Ownership:
- `aria_designer/api/app/routers/import.py`
- `aria_designer/api/app/research_signals.py`
- any importer helper modules directly used for survivor/leaderboard import

Responsibilities:
- normalize imported research/leaderboard workflows through `component_identity.py`
- store original research op names only in metadata
- fail import with explicit unresolved-ID diagnostics when mapping is impossible
- ensure imported workflows are designer-valid before returning success

Constraints:
- no UI logic
- no second mapping table
- no silent partial success

Deletion targets after cutover:
- importer-local alias handling
- legacy mixed-ID import branches

### Agent 3: Ask Aria / Chat Patch Normalization

Ownership:
- `aria_designer/api/app/conversation.py`
- `aria_designer/api/app/routers/aria.py`
- `aria_designer/api/app/intent_parser.py` only if needed for canonical references

Responsibilities:
- emit canonical IDs only in patch proposals
- normalize all resolved concepts and pattern templates through the canonical mapper
- reject or rewrite legacy aliases before patch response leaves the backend
- remove stale template IDs and duplicated concept maps where superseded

Constraints:
- no mixed IDs in patch payloads
- no duplicate resolver logic beyond calls into `component_identity.py`

Deletion targets after cutover:
- old alias-heavy concept maps
- stale pattern templates using non-canonical IDs

### Agent 4: Validation and Persistence Gate

Ownership:
- `aria_designer/api/app/routers/workflows.py`
- `aria_designer/ui/src/utils/workflow.js`
- `aria_designer/ui/src/App.main.jsx` only for workflow save/apply integration points

Responsibilities:
- normalize before validate/save where appropriate
- reject unresolved IDs with exact node-level messages
- ensure UI serialization writes canonical IDs only
- add a strict persistence gate: no workflow save if unresolved or mixed IDs remain

Constraints:
- keep validator simple
- do not add speculative repair logic in multiple places
- one normalization call path

Deletion targets after cutover:
- UI-side ad hoc component-type fallback assembly
- duplicate pre-save normalization code

### Agent 5: Proxy and Lifecycle Reliability

Ownership:
- `research/scientist/api_routes/_designer.py`
- `aria_designer/tools/dev_up.sh`
- `aria_designer/tools/dev_down.sh`

Responsibilities:
- make designer API lifecycle robust so proxy does not flap into `502`
- ensure backend health is explicit and logged
- make boot path leave a stable API worker on `8091`
- improve timeout/error surfacing so users see upstream-down vs invalid-workflow separately

Constraints:
- keep lifecycle code small
- do not bury failures behind generic `502`
- no duplicate launch paths if one can be deleted

Deletion targets after cutover:
- redundant manual/dev boot variants if one canonical path is enough
- dead watchdog branches if unused

### Agent 6: Test and Cleanup Pass

Ownership:
- `aria_designer/tests/`
- `research/tests/test_designer_proxy*.py`

Responsibilities:
- add round-trip tests:
  - research result -> import -> canonicalize -> validate
  - Ask Aria patch -> apply -> save -> reload -> validate
  - mixed-ID workflow -> normalize -> no unresolved IDs
  - unresolved workflow -> explicit failure
- remove tests that pin old mixed-ID behavior
- run dead-code cleanup after all cutovers land

Constraints:
- tests must target final architecture, not transitional duplication

## Execution Order

1. Agent 1 builds the canonical mapping core.
2. Agent 2 and Agent 3 switch import/chat to that core in parallel.
3. Agent 4 adds persistence and validation gates once the core is stable.
4. Agent 5 fixes proxy/lifecycle so failures are diagnosable and stable during rollout.
5. Agent 6 lands round-trip tests and removes dead compatibility tests.
6. Final cleanup deletes superseded maps and fallback branches.

## File-Level Split Boundaries

Create or keep responsibilities as:
- `component_identity.py`: canonical ID rules and normalization only
- importer modules: external payload ingestion only
- chat modules: proposal generation only
- workflow router: validate/save gate only
- proxy/lifecycle modules: service boot and upstream proxy only

Do not put:
- alias maps in UI files
- import normalization in validator
- validator lookup rules in chat
- research-specific names in persisted workflow core fields

## Non-Negotiable Deletions

These must be removed once cutover is complete:
- duplicate alias maps in `conversation.py` and importer code
- any save-time fallback that writes non-canonical IDs
- dead compatibility branches for mixed-ID persistence
- tests expecting partially normalized workflows to pass

No compatibility code stays "just in case" unless it has a named owner, a removal date, and a live caller.

## Verification Gates

### Functional

- importing a leaderboard result yields a fully canonical workflow
- Ask Aria patch payloads contain only canonical IDs
- validate/save/load succeed on canonical workflows
- unresolved IDs produce exact actionable errors

### Structural

- one mapping module only
- no file-level duplicate normalization logic
- no new god files
- no new functions over 100 lines without split
- no unused imports/functions after cutover

### Performance

Classify hotspots before optimizing:
- mapping core: serialization-bound / algorithmic, likely cheap
- import normalization: parsing-bound / serialization-bound
- validation: DB-bound if repeated lookups remain

Expected optimizations:
- request-scoped component lookup cache
- one-pass workflow normalization
- zero repeated JSON reparsing in import path

Do not introduce native/Cython work unless profiling shows normalization/validation is a real hotspot.

## Recommended Agent Assignments

- `codex`: Agent 1 or Agent 4
  - best fit for shared-core cleanup and validation gates
- `claude-opus`: Agent 5
  - large-file and split discipline for lifecycle/proxy cleanup
- `gemini`: Agent 2 or Agent 6
  - importer/test matrix work with broad file scanning

## Deliverable Checklist

- [ ] `component_identity.py` exists and is the only alias authority
- [ ] import path canonicalizes workflows
- [ ] chat path emits canonical IDs only
- [ ] workflow save rejects unresolved IDs
- [ ] proxy/lifecycle no longer silently flaps into generic `502`
- [ ] round-trip tests pass
- [ ] dead code and duplicate maps deleted

## Cutover Definition of Done

The system is unified only when:
- a top leaderboard architecture can be imported
- the imported graph validates without manual JSON edits
- Ask Aria can modify that graph
- the result saves and reloads
- all persisted `component_type` values are canonical
- there is exactly one mapping source of truth
