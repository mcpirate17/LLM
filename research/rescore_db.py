import sqlite3
import json
import torch
import os
import sys
from typing import Dict, Any

# Add the project root to the path so we can import research modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig
from research.eval.sandbox import safe_eval
from research.synthesis.serializer import graph_from_json
from research.scientist.native_runner import compile_model_native_first as compile_model

def main():
    db_path = "/home/tim/Projects/LLM/research/lab_notebook.db"
    db = LabNotebook(db_path)
    
    # Get all models that passed Stage 1 and haven't been rescored yet.
    # We explicitly exclude 'reference_arch' error_type to avoid overwriting 
    # gold-standard reference baselines with micro-corpus rescores.
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT p.experiment_id, p.result_id, p.graph_json, e.config_json 
        FROM program_results p
        JOIN experiments e ON p.experiment_id = e.experiment_id
        WHERE p.stage1_passed = 1
          AND (p.error_type IS NULL OR p.error_type NOT IN ('corpus_rescored', 'reference_arch'))
    """)
    rows = cursor.fetchall()
        
    print(f"Found {len(rows)} models that passed Stage 1.")
    
    runner = ExperimentRunner(db_path)
    
    for row in rows:
        exp_id, prog_id, code, config_json = row
        print(f"Processing {prog_id}...")
        
        try:
            config_dict = json.loads(config_json)
            
            # Filter config_dict to only include valid RunConfig fields
            import dataclasses
            valid_fields = {f.name for f in dataclasses.fields(RunConfig)}
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
            
            config = RunConfig(**filtered_config)
            
            # Force corpus mode for rescoring
            config.data_mode = "corpus"
            config.corpus_path = "/home/tim/Projects/LLM/research/micro_corpus.txt"
            
            # Build the model
            try:
                graph = graph_from_json(code)
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
            except Exception as e:
                print(f"  -> FAILED Compilation: {e}")
                cursor = db.conn.cursor()
                # If it's an unknown primitive like rfft_seq, it's likely a causality violation or restricted op
                err_type = "compilation_error"
                if "Unknown primitive" in str(e):
                    err_type = "restricted_op_violation"
                
                cursor.execute("""
                    UPDATE program_results 
                    SET stage1_passed = 0, error_type = ?
                    WHERE result_id = ?
                """, (err_type, prog_id))
                db.conn.commit()
                continue
            
            # Run the causality gate (Stage 0.5)
            sandbox_result = safe_eval(
                model=model,
                batch_size=2,
                seq_len=min(128, config.max_seq_len),
                vocab_size=config.vocab_size,
                device=config.device,
                run_stability_probe=True
            )
            
            if not sandbox_result.causality_passed:
                print(f"  -> FAILED Causality Gate. Marking as cheater.")
                cursor = db.conn.cursor()
                cursor.execute("""
                    UPDATE program_results 
                    SET stage1_passed = 0, error_type = 'causality_violation'
                    WHERE result_id = ?
                """, (prog_id,))
                db.conn.commit()
                continue
                
            print(f"  -> PASSED Causality Gate. Rescoring Stage 1...")
            
            # Re-run Stage 1
            dev = torch.device(config.device if torch.cuda.is_available() else "cpu")
            stage1_result = runner._micro_train(model, config, dev, seed=42)
            
            if not stage1_result.get("passed", False):
                print(f"  -> FAILED Stage 1 Rescoring: {stage1_result.get('error')}")
                cursor = db.conn.cursor()
                cursor.execute("""
                    UPDATE program_results 
                    SET stage1_passed = 0, error_type = ?
                    WHERE result_id = ?
                """, (stage1_result.get("error", "unknown"), prog_id))
                db.conn.commit()
                continue
                
            print(f"  -> PASSED Stage 1 Rescoring. New loss: {stage1_result.get('final_loss')}")
            
            # Compute baseline loss ratio
            baseline_ratio = None
            try:
                baseline = runner._get_baseline()
                baseline_steps = int(stage1_result.get("n_train_steps") or config.stage1_steps)
                baseline_recipe = runner._resolve_baseline_recipe(stage1_result, default_lr=config.stage1_lr)
                bl_data_fn, bl_data_tag, bl_cache = runner._make_baseline_data_fn(config)
                baseline_ratio = baseline.compare(
                    stage1_result.get("final_loss"),
                    d_model=config.model_dim,
                    seq_len=min(128, config.max_seq_len),
                    n_steps=max(1, baseline_steps),
                    vocab_size=config.vocab_size,
                    batch_size=config.stage1_batch_size,
                    lr=baseline_recipe["lr"],
                    device=config.device,
                    n_layers=config.n_layers,
                    optimizer_name=baseline_recipe["optimizer_name"],
                    weight_decay=baseline_recipe["weight_decay"],
                    momentum=baseline_recipe["momentum"],
                    betas=baseline_recipe["betas"],
                    data_fn=bl_data_fn,
                    data_tag=bl_data_tag,
                    cache_data_fn=bl_cache,
                )
            except Exception as e:
                print(f"  -> Failed to compute baseline ratio: {e}")
            
            # Update DB with new scores
            cursor = db.conn.cursor()
            cursor.execute("""
                UPDATE program_results 
                SET discovery_loss_ratio = loss_ratio,
                    discovery_loss = final_loss,
                    validation_loss = ?, 
                    validation_loss_ratio = ?, 
                    final_loss = ?,
                    loss_ratio = ?,
                    baseline_loss_ratio = ?, 
                    error_type = 'corpus_rescored'
                WHERE result_id = ?
            """, (
                stage1_result.get("final_loss"),
                stage1_result.get("loss_ratio", 1.0),
                stage1_result.get("final_loss"),
                stage1_result.get("loss_ratio", 1.0),
                baseline_ratio,
                prog_id
            ))
            db.conn.commit()
                
        except Exception as e:
            print(f"  -> ERROR processing {prog_id}: {e}")

if __name__ == "__main__":
    main()
