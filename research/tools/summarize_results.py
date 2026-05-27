import json
import numpy as np

with open("analysis_results.json") as f:
    data = json.load(f)


def is_valid(r):
    c = r["corr"]
    if c is None or np.isnan(c):
        return False
    # Significant if CI doesn't cross zero
    return r["ci_low"] > 0 or r["ci_high"] < 0


for metric, results in data["partial_primitive"].items():
    print(f"\n### {metric}")
    valid_results = [r for r in results if is_valid(r)]
    for r in valid_results[:10]:
        print(
            f"- **{r['primitive']}**: {r['corr']:.3f} CI[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        )

print("\n### Top Combinations (Residual Diff)")
for c in data["combinations"]:
    print(f"- {c['pair']}: {c['residual_diff']:.3f} (n={c['n']})")
