
import sqlite3
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.scientist.notebook import LabNotebook

def backfill():
    db_path = "research/lab_notebook.db"
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    nb = LabNotebook(db_path)
    cursor = nb.conn.cursor()

    # 1. For already rescored models:
    # We know loss_ratio currently holds the corpus result.
    # discovery_loss_ratio is likely NULL.
    # We can't easily recover the OLD discovery loss ratio if it was overwritten and not backed up.
    # Wait, did we overwrite it? Yes, rescore_db.py does: UPDATE program_results SET loss_ratio = ?
    
    # 2. For models NOT yet rescored:
    # loss_ratio holds the discovery result.
    print("Backfilling discovery_loss_ratio from loss_ratio for non-rescored models...")
    cursor.execute("""
        UPDATE program_results
        SET discovery_loss_ratio = loss_ratio,
            discovery_loss = final_loss
        WHERE discovery_loss_ratio IS NULL
          AND (error_type IS NULL OR error_type != 'corpus_rescored')
          AND stage1_passed = 1
    """)
    print(f"Rows updated: {cursor.rowcount}")

    # 3. For already rescored models, set validation_loss_ratio = loss_ratio
    print("Backfilling validation_loss_ratio from loss_ratio for rescored models...")
    cursor.execute("""
        UPDATE program_results
        SET validation_loss_ratio = loss_ratio,
            validation_loss = final_loss
        WHERE validation_loss_ratio IS NULL
          AND error_type = 'corpus_rescored'
    """)
    print(f"Rows updated: {cursor.rowcount}")

    # 4. Sync leaderboard table with new columns
    print("Syncing discovery_loss_ratio to leaderboard table...")
    cursor.execute("""
        UPDATE leaderboard
        SET discovery_loss_ratio = (
            SELECT pr.discovery_loss_ratio 
            FROM program_results pr 
            WHERE pr.result_id = leaderboard.result_id
        )
        WHERE discovery_loss_ratio IS NULL
          AND EXISTS (
              SELECT 1 FROM program_results pr 
              WHERE pr.result_id = leaderboard.result_id 
                AND pr.discovery_loss_ratio IS NOT NULL
          )
    """)
    print(f"Rows updated: {cursor.rowcount}")

    # Also update screening_loss_ratio if it was discovery but we now want validation by default?
    print("Updating screening_loss_ratio from validation_loss_ratio where possible...")
    cursor.execute("""
        UPDATE leaderboard
        SET screening_loss_ratio = (
            SELECT pr.validation_loss_ratio 
            FROM program_results pr 
            WHERE pr.result_id = leaderboard.result_id
        )
        WHERE (tier = 'screening' OR tier = 'validation')
          AND EXISTS (
              SELECT 1 FROM program_results pr 
              WHERE pr.result_id = leaderboard.result_id 
                AND pr.validation_loss_ratio IS NOT NULL
          )
    """)
    print(f"Leaderboard screening rows updated: {cursor.rowcount}")

    # Also update composite_score because the formula changed
    print("Recomputing Scientific Utility scores for all leaderboard entries...")
    # Fetch all data from leaderboard joined with program_results to get all metrics
    cursor.execute("""
        SELECT l.result_id as entry_result_id, l.entry_id, l.tier, l.model_source, l.architecture_desc, 
               l.is_reference, l.reference_name, l.tags, l.notes,
               pr.*
        FROM leaderboard l
        JOIN program_results pr ON l.result_id = pr.result_id
    """)
    rows = cursor.fetchall()
    col_names = [d[0] for d in cursor.description]
    
    for row_tuple in rows:
        d = dict(zip(col_names, row_tuple))
        
        # Call upsert_leaderboard - it will recompute score using all kwargs
        nb.upsert_leaderboard(
            result_id=d["entry_result_id"],
            model_source=d.get("model_source") or "unknown",
            architecture_desc=d.get("architecture_desc"),
            tier=d.get("tier") or "screening",
            is_reference=bool(d.get("is_reference")),
            reference_name=d.get("reference_name"),
            tags=d.get("tags"),
            notes=d.get("notes"),
            # Metrics from program_results
            screening_loss_ratio=d.get("loss_ratio"),
            screening_novelty=d.get("novelty_score"),
            investigation_loss_ratio=d.get("loss_ratio"),
            validation_loss_ratio=d.get("validation_loss_ratio"),
            validation_baseline_ratio=d.get("validation_baseline_loss_ratio") or d.get("baseline_loss_ratio"),
            validation_multi_seed_std=d.get("init_sensitivity_std"),
            routing_savings_ratio=d.get("routing_savings_ratio"),
            compression_ratio=d.get("compression_ratio"),
            discovery_loss_ratio=d.get("discovery_loss_ratio"),
            fp_jacobian_spectral_norm=d.get("fp_jacobian_spectral_norm"),
            robustness_noise_score=d.get("robustness_noise_score"),
            quant_int8_retention=d.get("quant_int8_retention"),
            robustness_long_ctx_score=d.get("long_context_score"),
            init_sensitivity_std=d.get("init_sensitivity_std"),
            loss_improvement_rate=d.get("loss_improvement_rate"),
            quant_quality_per_byte=d.get("quant_quality_per_byte"),
            novelty_confidence=d.get("novelty_confidence")
        )

    print(f"Recomputed and updated {len(rows)} entries.")
    nb.close()
    print("Backfill complete.")

if __name__ == "__main__":
    backfill()
