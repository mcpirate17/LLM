# Aria UI/UX Revamp Plan

## Phase 1: Foundation & Persona (Current Focus)
1. **Unified Design System:** Synchronize the color palettes between `aria_designer` (Deep Navy) and `research/dashboard` (Glassmorphism). 
2. **The "Heartbeat":** Port `AriaAvatar` to the Designer and wire it up to display system vitality/status (e.g., pulsing or changing moods based on experiment success).

## Phase 2: Anti-Monolith Refactoring
3. **Progressive Disclosure:** Break down massive 100KB+ components (like `ProgramDetail.js` and `App.jsx`) into smaller, lazy-loaded sub-components.

## Phase 3: Advanced UX Features
4. **"Hardware-Aware" Visualization:** Implement a "Kernel Pass" toggle in the Designer to highlight C-native nodes and estimate FLOPs/bandwidth.
5. **Direct AI Co-Design:** Move AskAria from a modal to "Ghost Suggestions" rendered directly on the React Flow canvas.
6. **Universal Command Palette:** Implement a `Ctrl+K` "Nexus" menu across both apps for rapid navigation.
