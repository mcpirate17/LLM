import sqlite3

conn = sqlite3.connect('/home/tim/Projects/LLM/research/lab_notebook.db')
c = conn.cursor()

c.execute("SELECT result_id, novelty_score, novelty_confidence FROM program_results WHERE novelty_confidence IS NOT NULL")
rows = c.fetchall()

print("Total with novelty_confidence:", len(rows))
if len(rows) > 0:
    vals = [r[2] for r in rows]
    print("Min novelty_conf:", min(vals))
    print("Max novelty_conf:", max(vals))
    print("Mean novelty_conf:", sum(vals)/len(vals))
