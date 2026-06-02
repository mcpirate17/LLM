"""Launch the visual explainer:  ``python -m component_fab.viz``."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="component_fab visual explainer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    print(f"component_fab visual explainer → http://{args.host}:{args.port}")
    uvicorn.run(
        "component_fab.viz.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
