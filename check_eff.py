from research.scientist.notebook import LabNotebook
import sqlite3

nb = LabNotebook()
rows = nb.conn.execute(
    "SELECT result_id, loss_ratio, param_count, flops_forward, "
    "throughput_tok_s, peak_memory_mb, forward_time_ms "
    "FROM program_results WHERE experiment_id = '0d03add4-686' "
    "ORDER BY loss_ratio ASC LIMIT 5"
).fetchall()

print(f"{'ID':<12} | {'Loss':<8} | {'Params':<8} | {'FLOPs':<8} | {'Thrput':<8} | {'Mem':<8} | {'Eta':<8}")
print("-" * 80)

for row in rows:
    d = dict(row)
    eff = LabNotebook.compute_efficiency_multiple(
        loss_ratio=d["loss_ratio"],
        param_count=d["param_count"],
        flops_forward=d["flops_forward"],
        throughput_tok_s=d["throughput_tok_s"],
        peak_memory_mb=d["peak_memory_mb"],
        forward_time_ms=d["forward_time_ms"]
    )
    
    eta = eff["geomean"] if eff else 0.0
    print(f"{d['result_id'][:12]} | {d['loss_ratio']:.4f} | {d['param_count']/1e6:5.1f}M | {d['flops_forward']/1e6:5.1f}M | {d['throughput_tok_s']/1e3:5.1f}k | {d['peak_memory_mb']:5.0f}MB | {eta:.4f}")
    if eff:
        for k, v in eff.items():
            if k != 'geomean' and k != 'n_dimensions':
                print(f"  - {k}: {v:.4f}")
