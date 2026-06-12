import json
import statistics
from pathlib import Path
from collections import defaultdict


def aggregate_results():
    reports_dir = Path("research/reports")
    # Only aggregate results from today's session (2026-06-07) and relevant regrades
    patterns = [
        "hardened_corrected_hard_params_*.json",
        "hardened_corrected_hard_flops_*.json",
        "matched_corrected_regrade_params.json",
        "matched_corrected_regrade_flops.json",
    ]

    # model_name -> task_name -> list of accuracies
    aggregated = defaultdict(lambda: defaultdict(list))

    for pattern in patterns:
        for file_path in reports_dir.glob(pattern):
            try:
                data = json.loads(file_path.read_text())
                rows = data.get("rows", {})
                for model_name, model_data in rows.items():
                    per_task = model_data.get("per_task", {})
                    for task_name, task_result in per_task.items():
                        # Handle different JSON structures (some have dicts with means, some have float direct)
                        if isinstance(task_result, dict):
                            val = task_result.get("accuracy_mean")
                            if val is not None:
                                aggregated[model_name][task_name].append(val)
                            # Also grab seed accuracies if available
                            seeds = task_result.get("seed_accuracies", [])
                            if seeds:
                                aggregated[model_name][task_name].extend(seeds)
                        elif isinstance(task_result, (int, float)):
                            aggregated[model_name][task_name].append(task_result)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    # Compute Stats
    final_stats = {}
    for model, tasks in aggregated.items():
        final_stats[model] = {}
        all_accs = []
        for task, accs in tasks.items():
            if not accs:
                continue
            mean = statistics.fmean(accs)
            stdev = statistics.stdev(accs) if len(accs) > 1 else 0.0
            final_stats[model][task] = {"mean": mean, "stdev": stdev, "n": len(accs)}
            all_accs.extend(accs)

        if all_accs:
            final_stats[model]["total_avg"] = statistics.fmean(all_accs)
            final_stats[model]["total_stdev"] = (
                statistics.stdev(all_accs) if len(all_accs) > 1 else 0.0
            )

    return final_stats


def generate_markdown(stats):
    tasks = [
        "episodic_unique_multi_query",
        "episodic_distinct_key_interference",
        "episodic_compositional",
    ]

    # Sort models by total average
    sorted_models = sorted(
        stats.keys(),
        key=lambda m: stats[model].get("total_avg", 0) if (model := m) in stats else 0,
        reverse=True,
    )

    lines = []
    lines.append(
        "| Model | Total Avg (±σ) | Unique (128) | Interference (256) | Compositional (128) |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- |")

    for model in sorted_models:
        row = stats[model]
        avg_str = (
            f"**{row.get('total_avg', 0):.3f}** (±{row.get('total_stdev', 0):.3f})"
        )

        task_cells = []
        for t in tasks:
            # Task names might vary slightly in reports (e.g. episodic_ vs hard_)
            # We match by substring
            matching_task = next((tn for tn in row.keys() if t in tn or tn in t), None)
            if matching_task and isinstance(row[matching_task], dict):
                t_stats = row[matching_task]
                task_cells.append(f"{t_stats['mean']:.3f} (±{t_stats['stdev']:.3f})")
            else:
                task_cells.append("-")

        lines.append(f"| {model} | {avg_str} | {' | '.join(task_cells)} |")

    return "\n".join(lines)


if __name__ == "__main__":
    stats = aggregate_results()
    md = generate_markdown(stats)
    print(md)
    with open("research/reports/statistical_consolidation.md", "w") as f:
        f.write(md)
