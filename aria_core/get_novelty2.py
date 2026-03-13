import sqlite3

conn = sqlite3.connect('/home/tim/Projects/LLM/research/lab_notebook.db')
c = conn.cursor()

c.execute("SELECT result_id, novelty_score, stage1_passed FROM program_results WHERE novelty_score IS NOT NULL")
rows = c.fetchall()

print("Total with novelty:", len(rows))
if len(rows) > 0:
    novelties = [r[1] for r in rows]
    print("Min novelty:", min(novelties))
    print("Max novelty:", max(novelties))
    print("Mean novelty:", sum(novelties)/len(novelties))
    above_04 = sum(1 for n in novelties if float(n) >= 0.4)
    print(f"Num above 0.4: {above_04} ({above_04/len(novelties)*100:.1f}%)")

