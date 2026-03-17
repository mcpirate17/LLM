import json
import sqlite3
import requests
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("distiller")

DB_PATH = "research/lab_notebook.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"


class LocalDistiller:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def query_ollama(self, prompt: str) -> str:
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
            )
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""

    def cluster_failures(self):
        """Analyze failure signatures and cluster them into thematic anti-patterns."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch failure signatures with at least 2 occurrences
        cursor.execute(
            "SELECT signature, error_types, n_failures FROM failure_signatures WHERE n_failures >= 2"
        )
        signatures = cursor.fetchall()

        if not signatures:
            logger.info("No significant failure signatures to cluster.")
            return

        logger.info(f"Clustering {len(signatures)} failure signatures...")

        # Prepare data for Gemma
        sig_data = []
        for s in signatures:
            sig_data.append(
                f"Types: {s['error_types']} | Count: {s['n_failures']} | Sig: {s['signature'][:200]}"
            )

        prompt = f"""
You are an expert deep learning systems engineer. Below is a list of failure signatures from an architecture search experiment.
Your task is to cluster these signatures into 5-8 thematic "Structural Anti-Patterns".

Signatures:
{chr(10).join(sig_data[:50])}

Format your output as a valid JSON object:
{{
  "clusters": [
    {{
      "name": "Category Name",
      "description": "Why this happens and what structural choices cause it.",
      "representative_signatures": ["sig1", "sig2"]
    }}
  ]
}}
"""
        result = self.query_ollama(prompt)
        # We'll save this to a file for Sonnet to read as 'Negative Learning' context
        try:
            # Basic cleanup of markdown markers if Gemma adds them
            clean_json = (
                result.strip().replace("```json", "").replace("```", "").strip()
            )
            with open("research/artifacts/failure_clusters.json", "w") as f:
                f.write(clean_json)
            logger.info(
                "Saved failure clusters to research/artifacts/failure_clusters.json"
            )
        except Exception as e:
            logger.error(f"Failed to save clusters: {e}")
            logger.debug(f"Raw output: {result}")

    def generate_narratives(self, limit: int = 50):
        """Generate architectural intent descriptions for top models."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch top models missing narratives (using a hypothetical column or just printing)
        cursor.execute(
            """
            SELECT result_id, graph_json, loss_ratio, novelty_score 
            FROM program_results 
            WHERE stage1_passed = 1
            ORDER BY loss_ratio DESC 
            LIMIT ?
        """,
            (limit,),
        )
        rows = cursor.fetchall()

        logger.info(f"Generating narratives for {len(rows)} models...")

        narratives = []
        for row in rows:
            graph = json.loads(row["graph_json"])
            ops = [
                node["op_name"]
                for node in graph["nodes"].values()
                if not node.get("is_input")
            ]

            prompt = f"""
Convert this computation graph JSON description into a one-sentence high-level architectural intent.
Focus on the combination of operations and what it suggests about the model's design (e.g., "Hybrid spectral-linear mixer", "Gated convolutional bottleneck").

Operations: {", ".join(ops)}
Metrics: Loss Ratio {row["loss_ratio"]}, Novelty {row["novelty_score"]}

Output only the one-sentence description.
"""
            desc = self.query_ollama(prompt)
            narratives.append({"result_id": row["result_id"], "narrative": desc})
            logger.info(f"Generated narrative for {row['result_id']}: {desc}")

        with open("research/artifacts/architecture_narratives.json", "w") as f:
            json.dump(narratives, f, indent=2)
        logger.info(
            "Saved narratives to research/artifacts/architecture_narratives.json"
        )


if __name__ == "__main__":
    distiller = LocalDistiller()
    distiller.cluster_failures()
    distiller.generate_narratives(limit=20)
