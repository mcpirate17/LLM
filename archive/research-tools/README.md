# Research Tool Archive

This directory quarantines historical research maintenance scripts that are not
part of the active `research` runtime or test surface.

Rules:
- Do not import these scripts from `research.*`.
- Do not reference them from active README or CI paths unless they are restored
  to supported locations.
- Restore a script only with an explicit owner, a documented use case, and
  tests for the supported path.

The archived files were moved out of `research/tools/` to keep the active
package smaller and reduce dead-code surface area.
