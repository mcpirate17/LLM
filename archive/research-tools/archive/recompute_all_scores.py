#!/usr/bin/env python3
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from research.scientist.notebook import LabNotebook

DB_PATH = "research/lab_notebook.db"


def main():
    db = LabNotebook(DB_PATH)
    cursor = db.conn.cursor()

    # Fetch all entries from leaderboard with the metrics needed for scoring
    cursor.execute("""
        SELECT l.entry_id, l.result_id, 
               l.screening_loss_ratio, l.screening_novelty,
               l.investigation_loss_ratio, l.investigation_robustness,
               l.validation_loss_ratio, l.validation_baseline_ratio, l.validation_multi_seed_std,
               pr.novelty_confidence, l.scaling_param_efficiency, l.is_reference,
               l.routing_savings_ratio, l.compression_ratio, l.routing_collapse_score,
               pr.discovery_loss_ratio, l.fp_jacobian_spectral_norm, l.robustness_noise_score,
               l.quant_int8_retention, l.robustness_long_ctx_score, l.init_sensitivity_std,
               pr.loss_improvement_rate, l.quant_quality_per_byte, l.wikitext_perplexity
        FROM leaderboard l
        JOIN program_results pr ON l.result_id = pr.result_id
    """)
    rows = cursor.fetchall()
    print(f"Recomputing scores for {len(rows)} entries...")

    for row in tqdm(rows):
        (
            eid,
            rid,
            s_lr,
            s_nov,
            i_lr,
            i_rob,
            v_lr,
            v_base,
            v_std,
            nov_conf,
            scal_eff,
            is_ref,
            rout_sav,
            comp_ratio,
            rout_coll,
            disc_lr,
            spec_norm,
            rob_noise,
            q_ret,
            long_ctx,
            init_std,
            loss_imp,
            q_qual,
            w_perp,
        ) = row

        # Call the updated scoring function
        new_score = LabNotebook.compute_composite_score(
            screening_lr=s_lr,
            screening_nov=s_nov,
            inv_lr=i_lr,
            inv_robust=i_rob,
            val_lr=v_lr,
            val_baseline=v_base,
            val_std=v_std,
            novelty_confidence=nov_conf,
            scaling_param_efficiency=scal_eff,
            is_reference=bool(is_ref),
            routing_savings=rout_sav,
            compression_ratio=comp_ratio,
            entropy=rout_coll,  # Using collapse score as proxy for entropy penalty if high
            discovery_lr=disc_lr,
            spectral_norm=spec_norm,
            robustness_noise=rob_noise,
            quant_retention=q_ret,
            long_ctx_score=long_ctx,
            init_std=init_std,
            loss_improvement_rate=loss_imp,
            quant_quality_per_byte=q_qual,
            wikitext_perplexity=w_perp,
        )

        cursor.execute(
            "UPDATE leaderboard SET composite_score = ? WHERE entry_id = ?",
            (new_score, eid),
        )

    db.conn.commit()
    db.conn.close()
    print("Scores recomputed and database updated.")


if __name__ == "__main__":
    main()
