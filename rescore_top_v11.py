import sqlite3
import sys

# Add the project root to sys.path so we can import research modules
sys.path.append("/home/tim/Projects/LLM")

from research.scientist.leaderboard_scoring import (
    build_score_kwargs,
    compute_composite_v11,
)


def main():
    db_path = "/home/tim/Projects/LLM/research/lab_notebook.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get top 50 by old score to see the shift
    rows = conn.execute(
        "SELECT * FROM leaderboard ORDER BY composite_score DESC LIMIT 50"
    ).fetchall()

    rescored = []
    for row in rows:
        d = dict(row)
        kwargs = build_score_kwargs(
            conn, None, d["result_id"], d, d.get("is_reference", 0)
        )
        new_score = compute_composite_v11(decompose=False, **kwargs)
        rescored.append(
            {
                "id": d["entry_id"],
                "old_score": d["composite_score"],
                "new_score": new_score,
                "desc": d["architecture_desc"],
                "tok": kwargs.get("tokenizer_mode"),
            }
        )

    # Sort by new score
    rescored.sort(key=lambda x: x["new_score"], reverse=True)

    print(
        f"{'Rank':<5} | {'New Score':<10} | {'Old Score':<10} | {'Tokenizer':<10} | {'Description'}"
    )
    print("-" * 80)
    for i, item in enumerate(rescored[:15]):
        print(
            f"{i + 1:<5} | {item['new_score']:<10.2f} | {item['old_score']:<10.2f} | {str(item['tok']):<10} | {item['desc']}"
        )


if __name__ == "__main__":
    main()
