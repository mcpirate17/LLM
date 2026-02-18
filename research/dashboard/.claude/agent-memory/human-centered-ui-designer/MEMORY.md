# Aria Dashboard — UI/UX Designer Memory

## Project: HYDRA / Aria AI Scientist Dashboard
React dashboard for autonomous neural architecture search. Located at `research/dashboard/src/`.

## Stack
- React (hooks, no UI library — all custom)
- Plain CSS in App.css (one file)
- No TypeScript, no component library (Radix/Shadcn/MUI not used)
- Server-Sent Events for live updates (EventSource)

## Design System (App.css tokens)
```
--bg-primary:   #0d1117   (page background)
--bg-secondary: #161b22   (card background)
--bg-tertiary:  #21262d   (input/nested background)
--border:       #30363d
--text-primary: #e6edf3
--text-secondary: #8b949e
--text-muted:   #484f58
--accent-blue:    #58a6ff
--accent-green:   #3fb950
--accent-yellow:  #d29922
--accent-red:     #f85149
--accent-purple:  #bc8cff
--accent-orange:  #f0883e
--radius:       8px
```

## Layout Pattern
- Fixed header → tab nav → main content area (max-width 1400px, margin auto)
- Overview tab: 2-col grid (1fr 2fr), bottom row (1fr 1fr)
- Cards: `.card` class, `--bg-secondary` background, 8px radius
- Tables: `.data-table`, custom th/td styles, hover highlight
- Modal: `.modal-overlay` (fixed, z-index 1000) for ProgramDetail

## Navigation
- 13 tabs grouped in 4 sections via inline `<span>` separators in the nav
- Tab labels stored in TAB_LABELS object in App.js
- Drill-down uses tab switching + state (no URLs/routing)
- ProgramDetail opens as a global modal overlay

## Component File Sizes (complexity signals)
- ProgramDetail.js: 1649 lines (very large)
- CampaignView.js: 1130 lines (large)
- StrategyAdvisor.js: 826 lines
- AriaChatPanel.js: 877 lines
- ControlPanel.js: 1486 lines (very large)
- Leaderboard.js: 1048 lines

## Recurring Issues Found (2026-02-18 Review)
See detailed-review.md for full TODO list.
Key patterns to avoid going forward:
1. Inline styles everywhere — no design token consistency for spacing
2. DOM manipulation via querySelector in StrategyAdvisor (line 752)
3. window.confirm() / window.alert() used in ExperimentList
4. var(--border-color) used inconsistently alongside var(--border)
5. Tab nav overflows on medium screens — no responsive handling
6. AriaChatPanel and LiveFeed both open independent EventSource connections
7. Leaderboard header has 7 consecutive <p> explanation paragraphs
8. ControlPanel has 7 mode buttons in a single flex row — wraps badly
9. No focus-visible styles for keyboard navigation
10. `font-family` in body includes `monospace` as a fallback (wrong)
