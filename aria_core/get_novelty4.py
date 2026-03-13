import sqlite3

conn = sqlite3.connect('/home/tim/Projects/LLM/research/lab_notebook.db')
c = conn.cursor()

c.execute("SELECT result_id, novelty_score, stage1_passed FROM program_results WHERE novelty_score IS NOT NULL AND stage1_passed = 1")
s1_rows = c.fetchall()

print("Total S1 survivors with novelty:", len(s1_rows))
if len(s1_rows) > 0:
    vals = [r[1] for r in s1_rows]
    print("S1 Min novelty:", min(vals))
    print("S1 Max novelty:", max(vals))
    print("S1 Mean novelty:", sum(vals)/len(vals))
    above_04 = sum(1 for n in vals if float(n) >= 0.4)
    print(f"S1 Num above 0.4: {above_04} ({above_04/len(vals)*100:.1f}%)")

