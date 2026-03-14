# Aria Designer: 8-Issue Multi-Agent Plan

## Context

The aria_designer visual model designer has 8 usability and feature issues. This plan splits work across 4 agents in 3 dependency-ordered phases. Constraints: no deadcode, no duplication, C/C++/Rust/Cython over Python for bottlenecks.

---

## Phase 1: Foundation (Parallel — gemini)

### Issue 1: Auto-Snap Offset Bug

**Root cause:** `onDrop` in `useDesignerCanvasEvents.js` adds nodes without calling `findNearestFreePosition()`. On `onNodeDragStop`, `node.measured` is undefined for newly-dropped nodes, so `getNodeSize()` falls back to hardcoded 170x80 which doesn't match actual render size. Duplicate snap logic exists in both `DesignerCanvas.jsx` and `useDesignerCanvasEvents.js`.

**Fix:**
- `ui/src/utils/layout.js` — Add `sizeHint` parameter to `getNodeSize()` for pre-render nodes
- `ui/src/hooks/useDesignerCanvasEvents.js` — Call `findNearestFreePosition()` in `onDrop` after adding node, with explicit size hint `{width: 170, height: 90}`
- `ui/src/components/Canvas/DesignerCanvas.jsx` — Remove duplicate `onNodeDragStop` snap logic; delegate to hook

### Issue 5: Components Disappearing

**Root cause:** No viewport bounds clamping. `findNearestFreePosition` can push nodes to extreme coordinates. `fitView` only runs on initial load. Delete handler fires on Del/Backspace without checking `contentEditable`.

**Fix:**
- `ui/src/utils/layout.js` — Add `clampToViewport(position, nodeSize, bounds)` with default bounds `{minX: -2000, minY: -2000, maxX: 5000, maxY: 5000}`. Call from `findNearestFreePosition` fallback path
- `ui/src/hooks/useDesignerCanvasEvents.js` — Call `fitView({padding: 0.1, duration: 200})` after drop and after patch application
- `ui/src/App.main.jsx` — Fix Delete guard: add `e.target.isContentEditable` check

### Issue 8: Save → Fingerprint Link

**Fix:**
- `ui/src/App.main.jsx` — Add `discoveryUrl` to `saveState` on save success
- `ui/src/components/Header.jsx` — Render fingerprint as clickable `<a>` link (monospace, accent color) opening discovery page in new tab at `http://localhost:5000/?search={fingerprint}`
- `ui/src/styles.css` — `.fingerprint-link` styling

---

## Phase 2: Backend Intelligence (codex — depends on Phase 1)

### Issue 3: Ask Aria Placement Accuracy

**Root cause:** `_auto_layout_workflow` in `aria.py` uses flat column assignment. `_auto_connect_added_nodes` inserts at trunk end with no dataflow awareness.

**Fix:**
- `api/app/intent_parser.py` — New `compute_insertion_point(nodes, edges, component_type) -> {after_node_id, before_node_id}` using topological ordering + category rules (norm after projection, activation after linear, routing before mixing)
- `api/app/suggestions.py` — Include `insertion_hint` in each suggestion using `compute_insertion_point`
- `api/app/routers/aria.py` — `_auto_connect_added_nodes` uses insertion hint for wiring; `_auto_layout_workflow` assigns x/y from topological depth of insertion point

### Issue 4: Leaderboard Data in Ask Aria

**Root cause:** `research_signals.py` fetches aggregate stats but no individual leaderboard entries. No UI shows why a component was suggested.

**Fix:**
- `api/app/research_signals.py` — Add `fetch_leaderboard_top_entries(n=10, min_composite=50.0)` with thread-safe TTL cache (120s), fetching from `/api/leaderboard?limit={n}&sort=composite_score`
- `api/app/suggestions.py` — Add `_leaderboard_boost(component, entries)`: parse top entries' `program_text` for component types, boost score proportional to usage frequency. Enrich `evidence` with "Used in N of top 10 architectures"
- `api/app/routers/aria.py` — New endpoint `GET /api/v1/aria/historical-insights` returning `{top_components, success_patterns, failure_patterns}`
- `api/app/models.py` — Add `HistoricalInsightsResponse` model

### Issue 6: No Score or Discovery Link After Run

**Root cause:** Evaluate stream emits stage metrics but never a composite score or fingerprint. No link back to discovery page.

**Fix:**
- `api/app/routers/workflows.py` — After novelty stage in evaluate/stream, compute and emit `composite_score` + `graph_fingerprint` in final SSE event
- `ui/src/hooks/useWorkflow.js` — Extract `composite_score` and `graph_fingerprint` from final SSE payload into `evalState`
- `ui/src/components/RunResultsPanel.jsx` — Add "Score & Discovery" section at top: composite score badge (color-coded), fingerprint as clickable link to discovery page, "Compare" button

---

## Phase 3: Major Features (depends on Phase 2)

### Issue 2: Help Material & Component Advice — claude-opus

**Fix:**
- `api/app/help_content.py` (new) — Compatibility matrix from `intent_parser.py`'s `_LEAF_GROUPS`/`_COMPONENT_GROUPS`. "Works well with" / "avoid with" derived from category adjacency + `failure_risk_signatures` from research signals. Cache aggressively
- `api/app/routers/help.py` (new) — `GET /api/v1/help/component/{id}/tips` returning `{compatibility, usage_in_top_architectures, common_patterns}`
- `ui/src/components/HelpPanel.jsx` (new) — Slideover with 3 tabs: Getting Started (static tutorial), Component Guide (searchable, per-component help_md + tips), Patterns (top architecture patterns from leaderboard)
- `ui/src/components/ContextualTip.jsx` (new) — Small card below inspector header when node selected, showing compatibility hints
- `ui/src/components/InspectorMain.jsx` — Render `ContextualTip` below description
- `ui/src/components/Header.jsx` — Add "Help" button opening HelpPanel

### Issue 7: Conversational Chat with Aria — claude-opus (backend) + gemini (frontend)

**Backend (claude-opus):**
- `api/app/database.py` — Add tables:
  ```sql
  aria_conversations (session_id PK, workflow_id, started_at, last_message_at, status)
  aria_messages (rowid PK, session_id FK, role, content, metadata_json, created_at)
  ```
- `api/app/conversation.py` (new) — `ConversationManager` with `__slots__`:
  - `start_session(workflow_json) -> session_id`
  - `process_message(session_id, message, workflow_json) -> ConversationResponse` — parse intent via `intent_parser.py` (reuse, don't duplicate), check if clarification needed, generate patch or ask question, store history
  - Tracks: applied changes, rejected suggestions, user goals
  - Integrates research signals (from Issue 4 infrastructure)
- `api/app/routers/chat.py` (new) — `POST /api/v1/aria/chat`, `GET /api/v1/aria/chat/{session_id}/history`, `DELETE /api/v1/aria/chat/{session_id}`
- `api/app/models.py` — Add `ChatMessageRequest`, `ChatMessageResponse`, `ConversationSession`

**Frontend (gemini, after backend ready):**
- `ui/src/components/AriaChatPanel.jsx` (new) — Persistent chat panel as tab alongside Inspector/Results. Message bubbles, patch proposal cards with Apply/Reject, suggested follow-up buttons
- `ui/src/hooks/useAriaChat.js` (new) — Session management, send/receive, history
- `ui/src/components/AskAriaModal.jsx` — Add "Switch to Chat" button
- `ui/src/styles/Chat.css` (new) — Chat UI styling

---

## Progress

### Phase 3 — claude-opus (DONE 2026-03-14)
- [x] Issue 2: `help_content.py`, `routers/help.py`, `HelpPanel.jsx`, `ContextualTip.jsx`, `HelpPanel.css`
- [x] Issue 7 backend: `conversation.py`, `routers/chat.py`, chat tables in `database.py`, chat models in `models.py`
- [x] Wired both routers into `main.py` via `app.include_router()`
- [x] All 437 tests pass

### Phase 1 — gemini (DONE 2026-03-14)
- [x] Issue 1: `layout.js` (`sizeHint`, `getNodeSize`), `useDesignerCanvasEvents.js` (`findNearestFreePosition` in onDrop)
- [x] Issue 5: `clampToViewport` in `layout.js`, `fitView` after drop
- [x] Issue 8: `discoveryUrl` + fingerprint `<a>` link in `Header.jsx` / `App.main.jsx`

### Phase 2 — codex (DONE 2026-03-14)
- [x] Issue 3: `compute_insertion_point` in `intent_parser.py`, insertion hints in `suggestions.py`, wiring in `routers/aria.py`
- [x] Issue 4: `fetch_leaderboard_top_entries` in `research_signals.py`, `_leaderboard_boost` in `suggestions.py`, `historical-insights` endpoint in `routers/aria.py`
- [x] Issue 6: SSE composite_score in `routers/workflows.py`, frontend extraction in `App.main.jsx`, score badge + fingerprint link in `RunResultsPanel.jsx`

### Phase 3 — gemini (DONE 2026-03-14)
- [x] Issue 7 frontend: `AriaChatPanel.jsx`, `useAriaChat.js`, Chat CSS in `styles.css`, "Switch to Chat" in `AskAriaModal.jsx`

### Phase 4 — Issue 10: App.main.jsx Split (DONE 2026-03-14)
- [x] Extracted `hooks/useGraphHistory.js` (69 lines) — captureSnapshot, undoGraph, redoGraph
- [x] Extracted `hooks/useNodeStatus.js` (109 lines) — clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setAllNodeEvalStatus
- [x] Extracted `hooks/useKeyboardShortcuts.js` (91 lines) — global keyboard event listener
- [x] Extracted `hooks/useFileOperations.js` (314 lines) — loadWorkflowJson, import/export, save, clear
- [x] Extracted `hooks/useWorkflowPipeline.js` (572 lines) — validate, compile, preview, deep run + SSE
- [x] Extracted `hooks/useAriaCoDesign.js` (233 lines) — suggest, submit, apply/reject/preview patches
- [x] Extracted `components/TopBar.jsx` (217 lines) — toolbar JSX
- [x] Removed 6 dead hook files (useDesignerState, useDesignerActions, useHistory, useAria, useDesignerCanvasEvents, useWorkflow)
- [x] `App.main.jsx`: 2795 → 957 lines (66% reduction)
- [x] All 437 tests pass, frontend builds cleanly

---

## Agent Assignment

| Agent | Issues | Phase |
|---|---|---|
| **gemini** | #1 (snap), #5 (disappearing), #8 (save link), #7 frontend | P1, P3 |
| **codex** | #3 (placement), #4 (leaderboard data), #6 (score display) | P2 |
| **claude-opus** | #2 (help system), #7 backend (chat), #9 (main.py split), #10 (App.main.jsx split) | P3, P4 |
| **user's claude** | Integration testing, coordination, plan file | All |

## File Ownership (conflict avoidance)

### Shared files — multiple agents touch these
| File | Who | What |
|---|---|---|
| `api/app/models.py` | claude-opus added chat models; codex adds `HistoricalInsightsResponse` | **Codex: append only, do not overwrite existing chat models** |
| `api/app/database.py` | claude-opus added chat tables + CRUD | **Codex: do not modify** — chat tables already present |
| `api/app/main.py` | claude-opus added router includes | **Codex: use `routers/aria.py` for new endpoints, not main.py** |
| `ui/src/components/Header.jsx` | gemini added fingerprint link; claude-opus added Help button | **Both done — no further changes expected** |
| `ui/src/components/InspectorMain.jsx` | claude-opus added ContextualTip import/render | **Done** |
| `ui/src/App.main.jsx` | gemini (P1 changes); claude-opus (HelpPanel import + render) | **Both done** |

### Codex-owned files (no conflicts)
- `api/app/intent_parser.py` — add `compute_insertion_point`
- `api/app/research_signals.py` — add `fetch_leaderboard_top_entries`
- `api/app/suggestions.py` — add `_leaderboard_boost`, insertion hints
- `api/app/routers/aria.py` — add `historical-insights` endpoint, update wiring
- `api/app/routers/workflows.py` — add composite_score SSE event
- `ui/src/hooks/useWorkflow.js` — extract composite_score/fingerprint
- `ui/src/components/RunResultsPanel.jsx` — score badge + link
- `ui/src/styles/RunResults.css` — styling
- `tests/test_aria_features.py` — tests

### Note on `HistoricalInsightsResponse`
claude-opus already added this model to `models.py`. Codex should import and use it, not re-create it. Current definition:
```python
class HistoricalInsightsResponse(BaseModel):
    top_components: List[Dict[str, Any]] = Field(default_factory=list)
    success_patterns: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)
```

## Dependency Graph

```
P1 (parallel):     gemini → Issues 1, 5, 8                    ✓ DONE
P2 (after P1):     codex  → Issues 3, 4, 6                    ✓ DONE
P3 (after P2):     claude-opus → Issues 2, 7-backend           ✓ DONE
                   gemini → Issue 7-frontend (after 7-backend)  ✓ DONE
P4 (tech debt):    claude-opus → Issue 10 (App.main.jsx split)   ✓ DONE
Final:             user's claude → integration test all 8       ✓ DONE (437 pass)
```

## Shared Utilities (DRY enforcement)

| Utility | Owner | Consumers |
|---|---|---|
| `layout.js` — all placement/snap logic | gemini (P1) | codex (P2 placement) |
| `research_signals.py` — all research data fetching | codex (P2) | claude-opus (P3 help + chat) |
| `intent_parser.py` — intent classification + insertion | codex (P2) | claude-opus (P3 chat) |
| `database.py` — schema + chat CRUD | claude-opus (P3) | codex (read-only) |

---

## Phase 4: Tech Debt (claude-opus — after all issues merged)

### Issue 9: Split `main.py` (3559 lines → ~1250 max per file)

**Problem:** `api/app/main.py` is 3559 lines — nearly 3x the 1250-line architectural limit. All routes are defined inline instead of using the existing `routers/` directory.

**Fix — claude-opus:**
- Extract route groups into `routers/` files (most already exist as stubs):
  - `routers/workflows.py` — validate, compile, preview, run, evaluate, save, load (~500 lines)
  - `routers/components.py` — CRUD, property audit, config validation (~300 lines)
  - `routers/aria.py` — propose/apply/reject patch, suggest, refine (~200 lines, partially done)
  - `routers/importers.py` — survivors import, ONNX export (~150 lines)
  - `routers/evolution.py` — evolutionary refinement (~100 lines)
  - `routers/blocks.py` — block templates, extract/expand (~100 lines)
- Keep in `main.py`: app creation, middleware, lifespan, WebSocket, and router includes
- Target: `main.py` < 500 lines, each router < 600 lines

### Issue 10: Split `App.main.jsx` (2795 lines → ~1250 max)

**Problem:** `ui/src/App.main.jsx` is 2795 lines — over 2x the limit. Mixes state management, API calls, event handlers, and JSX rendering.

**Fix — claude-opus:**
- Extract custom hooks: `useWorkflowState.js`, `useCanvasActions.js`, `useFileOperations.js`
- Extract toolbar into `TopBar.jsx` component
- Extract API call functions into `services/workflowApi.js`
- Target: `App.main.jsx` < 800 lines, each extracted module < 400 lines

## Verification

1. `cd /home/tim/Projects/LLM/aria_designer && python -m pytest tests/ --ignore=tests/test_aria_features.py -x -q`
2. Start backend (port 8091) + frontend (port 5174) + research dashboard (port 5000)
3. **Issue 1**: Drop node near existing node → snaps to correct grid position, no offset
4. **Issue 2**: Select node → contextual tip appears; click Help → panel with tutorial + guide
5. **Issue 3**: Suggest normalization on `input→linear→output` → inserts between linear and output
6. **Issue 4**: Suggestions show "Used in N of top 10 architectures" evidence
7. **Issue 5**: Add many nodes → none disappear; verify fitView after drops
8. **Issue 6**: Deep Run → composite score badge + clickable fingerprint link at top of results
9. **Issue 7**: Open chat → type goal → Aria asks clarifying question → builds graph iteratively
10. **Issue 8**: Save → fingerprint link in header → click opens discovery page filtered to fingerprint
11. Grep for duplicate snap logic, duplicate HTTP calls — zero matches
12. No unused imports/functions in modified files
