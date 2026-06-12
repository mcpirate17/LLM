from research.tools._recall_probe_common import UniversalRecallLane, run_comparisons


def run_eval() -> None:
    STEPS = 1000
    DIM = 64
    comparisons = [
        ("universal_recall", lambda d: UniversalRecallLane(d), "distractor_kv_recall"),
        ("universal_recall", lambda d: UniversalRecallLane(d), "long_gap_recall"),
        ("universal_recall", lambda d: UniversalRecallLane(d), "compositional_binding"),
    ]
    run_comparisons(
        comparisons,
        steps=STEPS,
        dim=DIM,
        device="cuda",
        out_path="research/reports/universal_recall_results.json",
    )


if __name__ == "__main__":
    run_eval()
