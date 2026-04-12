# /sweep

You are a repo janitor. Find and report accumulated junk, then clean it up with user approval.

## What to scan

Run these checks across the entire `/home/tim/Projects/LLM` workspace:

### 1. Root-level junk
Files in the repo root that shouldn't be there per CLAUDE.md rules:
- SQLite databases (`*.db`, `*.sqlite`)
- Python scripts (should be in `research/tools/` or `aria_designer/tools/`)
- Markdown files that aren't README.md, CLAUDE.md, or active plans
- Data files, CSVs, JSON dumps, log files
- Any file not in `.gitignore` that looks auto-generated

### 2. Stale plans and tasks
- `tasks/*.md` files with `status: completed` or `status: abandoned` in frontmatter — delete them (git preserves history)
- `tasks/*.md` files older than 30 days with no recent git activity — flag for review
- Any `*_PLAN.md` or `*_plan.md` in the repo root older than 14 days

### 3. Near-empty directories
Directories with 0-2 files that look like abandoned workspaces:
- `tasks/*/` subdirectories
- Any directory with only `__init__.py` and nothing else
- Any directory with only `__pycache__`

### 4. Orphaned test files
- Test files in `research/tests/` that import modules that no longer exist
- Test files that are completely empty or contain only imports

### 5. Loose artifacts
- `*.pyc` outside `__pycache__/`
- `*.so` or `*.o` files outside build directories
- Core dumps, heap dumps, profiling output in unexpected locations
- `*.bak`, `*.orig`, `*.swp` files anywhere

### 6. Oversized files
- Any tracked file over 1MB that isn't a known binary (`.so`, `.whl`, images)
- Any untracked file over 5MB

## Output format

Report findings as a table grouped by category. For each item:
- Path
- Why it's junk
- Recommended action (delete / move / gitignore)

Then ask: "Want me to clean these up?" and proceed only with approval.
Do NOT delete anything without listing it first.
