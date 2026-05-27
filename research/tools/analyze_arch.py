import sqlite3
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Variance Budget from Reruns
VARIANCE_BUDGET = {
    "induction_intermediate_auc": 0.2087,
    "ar_validation_rank_score": 0.8259,
    "binding_multislot_held_entity_slot_acc": 0.0217,
    "binding_intermediate_auc": 0.0020,  # using binding_range std from rerun
    "ar_curriculum_auc_pair_final": 0.015,  # estimated from similar metrics
}


def load_data(db_path):
    conn = sqlite3.connect(db_path)
    # Using columns that exist and are relevant
    query = """
    SELECT 
        result_id, 
        graph_fingerprint,
        graph_json,
        param_count,
        n_train_steps,
        ar_curriculum_auc_pair_final,
        binding_intermediate_auc,
        binding_curriculum_auc,
        binding_multislot_held_entity_slot_acc,
        induction_intermediate_auc,
        ar_validation_rank_score
    -- program_results_compat (= graph_runs LEFT JOIN graphs) is the canonical
    -- read path post-Phase-5b; the raw program_results table is legacy and is NOT
    -- updated by probe backfills, so it returns stale (pre-leak-fix) binding.
    -- See research/notes/adjacent_token_merge_leak_2026-05-23.md.
    FROM program_results_compat
    WHERE graph_json IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def parse_graph(graph_json_str):
    try:
        g = json.loads(graph_json_str)
        nodes = g.get("nodes", {})
        if isinstance(nodes, list):
            primitives = [n.get("op_name") for n in nodes if n.get("op_name")]
        else:
            primitives = [n.get("op_name") for n in nodes.values() if n.get("op_name")]

        # Filter out trivial ops
        ignore = {"input", "output", "add", "residual", "identity"}
        primitives = [p for p in primitives if p not in ignore]

        model_dim = g.get("model_dim", 0)
        n_blocks = g.get("n_blocks", 1)  # default to 1 if not specified
        return primitives, model_dim, n_blocks
    except Exception:
        return [], 0, 0


def bootstrap_corr(x, y, noise_std, n_resamples=1000):
    corrs = []
    n = len(x)
    for _ in range(n_resamples):
        idx = np.random.choice(n, n, replace=True)
        xi = x[idx]
        yi = y[idx]
        # Add noise according to variance budget
        yi_noisy = yi + np.random.normal(0, noise_std, size=n)

        # Handle NaNs
        mask = ~np.isnan(xi) & ~np.isnan(yi_noisy)
        if mask.sum() > 5:
            r, _ = stats.pearsonr(xi[mask], yi_noisy[mask])
            corrs.append(r)

    if not corrs:
        return 0, (0, 0)
    return np.median(corrs), (np.percentile(corrs, 5), np.percentile(corrs, 95))


from sklearn.linear_model import LinearRegression


def get_residuals(df, metric):
    valid = df[
        ~df[metric].isna() & ~df["model_dim"].isna() & ~df["param_count"].isna()
    ].copy()
    if len(valid) < 10:
        return None, None

    # Use log param_count as it usually scales better
    X = valid[["model_dim", "param_count"]].copy()
    X["log_params"] = np.log10(X["param_count"] + 1)
    X = X[["model_dim", "log_params"]]

    model = LinearRegression()
    model.fit(X, valid[metric])
    residuals = valid[metric] - model.predict(X)
    return residuals, valid.index


def main():
    df = load_data("research/runs.db")
    print(f"Loaded {len(df)} runs.")

    # Process graphs
    parsed = df["graph_json"].apply(parse_graph)
    df["primitives"] = [p for p, d, b in parsed]
    df["model_dim"] = [d for p, d, b in parsed]
    df["n_blocks"] = [b for p, d, b in parsed]

    # One-hot encode primitives
    mlb = MultiLabelBinarizer()
    prim_matrix = mlb.fit_transform(df["primitives"])
    prim_df = pd.DataFrame(prim_matrix, columns=mlb.classes_)

    # Filter for primitives with enough occurrences
    min_count = 5
    popular_prims = prim_df.columns[prim_df.sum() > min_count]
    prim_df = prim_df[popular_prims]

    metrics = [
        "ar_curriculum_auc_pair_final",
        "binding_intermediate_auc",
        "binding_multislot_held_entity_slot_acc",
        "induction_intermediate_auc",
        "ar_validation_rank_score",
    ]

    results = {}
    partial_results = {}

    for metric in metrics:
        print(f"Analyzing {metric}...")
        y_all = df[metric].values
        noise_std = VARIANCE_BUDGET.get(metric, 0.05)

        # Naive bootstrap
        metric_results = []
        for prim in popular_prims:
            x = prim_df[prim].values
            med, ci = bootstrap_corr(x, y_all, noise_std)
            metric_results.append(
                {"primitive": prim, "corr": med, "ci_low": ci[0], "ci_high": ci[1]}
            )
        results[metric] = sorted(
            metric_results, key=lambda x: abs(x["corr"]), reverse=True
        )

        # Partial bootstrap
        residuals, idx = get_residuals(df, metric)
        if residuals is not None:
            print(f"Analyzing partials for {metric}...")
            p_metric_results = []
            prim_df_valid = prim_df.loc[idx]
            for prim in popular_prims:
                x = prim_df_valid[prim].values
                med, ci = bootstrap_corr(x, residuals.values, noise_std)
                p_metric_results.append(
                    {"primitive": prim, "corr": med, "ci_low": ci[0], "ci_high": ci[1]}
                )
            partial_results[metric] = sorted(
                p_metric_results, key=lambda x: abs(x["corr"]), reverse=True
            )

    # Co-occurrence analysis
    print("Analyzing combinations...")
    combinations = []
    # Use residuals for AR if available
    ar_metric = "ar_curriculum_auc_pair_final"
    res, idx = get_residuals(df, ar_metric)
    if res is not None:
        target = res
        df.loc[idx]
        prim_df_valid = prim_df.loc[idx]
    else:
        target = df[ar_metric]
        prim_df_valid = prim_df

    for i, p1 in enumerate(popular_prims):
        for p2 in popular_prims[i + 1 :]:
            mask = (prim_df_valid[p1] == 1) & (prim_df_valid[p2] == 1)
            if mask.sum() > 5:
                combo_score = target[mask].mean()
                others_score = target[~mask].mean()
                combinations.append(
                    {
                        "pair": f"{p1} + {p2}",
                        "n": int(mask.sum()),
                        "residual_diff": float(combo_score - others_score)
                        if not np.isnan(combo_score) and not np.isnan(others_score)
                        else 0,
                    }
                )

    combinations = sorted(
        combinations, key=lambda x: abs(x["residual_diff"]), reverse=True
    )[:15]

    # Save results
    with open("analysis_results.json", "w") as f:
        json.dump(
            {
                "per_primitive": results,
                "partial_primitive": partial_results,
                "combinations": combinations,
                "variance_budget": VARIANCE_BUDGET,
            },
            f,
            indent=2,
        )

    print("Done. Results saved to analysis_results.json")


if __name__ == "__main__":
    main()
