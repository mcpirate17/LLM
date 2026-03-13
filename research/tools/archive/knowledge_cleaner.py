import sqlite3
import requests
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cleaner")

DB_PATH = "research/lab_notebook.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"

def query_ollama(prompt: str) -> str:
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        })
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return ""

def clean_knowledge_base():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Pass 1: Exact String Deduplication (Fast)
    cursor.execute("SELECT insight_id, content FROM insights WHERE status IN ('active', 'superseded')")
    rows = cursor.fetchall()
    
    seen_content = {}
    for r in rows:
        content = r['content'].strip()
        if content not in seen_content:
            seen_content[content] = []
        seen_content[content].append(r['insight_id'])
    
    exact_duplicates = []
    for content, ids in seen_content.items():
        if len(ids) > 1:
            exact_duplicates.extend(ids[:-1])
    
    if exact_duplicates:
        logger.info(f"Found {len(exact_duplicates)} exact duplicates. Archiving...")
        with conn:
            for i in range(0, len(exact_duplicates), 500):
                batch = exact_duplicates[i:i+500]
                placeholders = ",".join(["?"] * len(batch))
                cursor.execute(f"UPDATE insights SET status = 'archived_duplicate' WHERE insight_id IN ({placeholders})", batch)

    # Pass 2: Fuzzy Deduplication (Gemma)
    cursor.execute("SELECT insight_id, content FROM insights WHERE status IN ('active', 'superseded')")
    remaining_rows = cursor.fetchall()
    
    if len(remaining_rows) < 2:
        logger.info("Not enough unique insights for fuzzy analysis.")
        conn.close()
        return

    logger.info(f"Analyzing {len(remaining_rows)} unique insights for fuzzy redundancy (sampling top 60)...")
    # Take top 60 to keep time reasonable
    rows_to_check = remaining_rows[:60]
    
    batch_size = 20
    to_delete = []
    
    for i in range(0, len(rows_to_check), batch_size):
        batch = rows_to_check[i:i+batch_size]
        contents = [f"[{r['insight_id']}] {r['content']}" for r in batch]
        
        prompt = f"""
Identify redundant or nearly identical research insights from the list below.
An insight is redundant if it conveys the same meaning as another, even if numbers vary slightly.

Insights:
{chr(10).join(contents)}

Return a JSON list of insight_ids that should be DELETED because they are duplicates of others in this list. 
Keep the most recent or descriptive one.
Output ONLY a JSON array of strings. Example: ["id1", "id2"]
"""
        result = query_ollama(prompt)
        try:
            # Cleanup markdown
            res_text = result.strip()
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                res_text = res_text.split("```")[1].split("```")[0]
            
            ids = json.loads(res_text.strip())
            if isinstance(ids, list):
                to_delete.extend(ids)
        except Exception as e:
            logger.debug(f"Failed to parse Gemma output: {e}")

    if to_delete:
        logger.info(f"Fuzzy deduplication identified {len(to_delete)} candidates. Archiving...")
        with conn:
            placeholders = ",".join(["?"] * len(to_delete))
            cursor.execute(f"UPDATE insights SET status = 'archived_duplicate' WHERE insight_id IN ({placeholders})", to_delete)

    conn.close()
    logger.info("Knowledge base cleaning complete.")

if __name__ == "__main__":
    clean_knowledge_base()
