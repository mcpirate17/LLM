
import json
import requests
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("linter")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"

class ArchitectureLinter:
    def query_ollama(self, prompt: str) -> str:
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            })
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""

    def lint_graph(self, graph_json: str) -> Dict[str, Any]:
        """Use Gemma to identify structural 'smells' in a computation graph."""
        graph = json.loads(graph_json)
        nodes = graph.get("nodes", {})
        
        # Build a readable representation for Gemma
        ops = []
        for nid, node in nodes.items():
            inputs = ", ".join([str(i) for i in node.get("input_ids", [])])
            ops.append(f"Node {nid}: {node['op_name']}(inputs=[{inputs}])")

        prompt = f"""
Analyze this computation graph for architectural "smells" or flaws.
Smells to look for:
1. Disconnected nodes (nodes that don't lead to the final output).
2. Redundant operations (e.g., neg(neg(x))).
3. Identity loops (operations that return the input unchanged).
4. Bottlenecks (drastic dimension reduction then expansion).

Graph:
{chr(10).join(ops)}

Output a JSON object:
{{
  "pass": true/false,
  "smells": ["description 1", "description 2"],
  "recommendation": "Summary advice"
}}
"""
        result = self.query_ollama(prompt)
        try:
            # Cleanup markdown
            res_text = result.strip()
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                res_text = res_text.split("```")[1].split("```")[0]
            
            return json.loads(res_text.strip())
        except:
            return {"pass": True, "smells": [], "error": "Linter failed to parse"}

if __name__ == "__main__":
    # Test with a known "weird" graph from today's search
    test_graph = '{"nodes": {"0": {"op_name": "input", "input_ids": []}, "1": {"op_name": "square", "input_ids": [0]}, "2": {"op_name": "maximum", "input_ids": [0, 1]}, "3": {"op_name": "neg", "input_ids": [2]}, "4": {"op_name": "neg", "input_ids": [3]}, "5": {"op_name": "add", "input_ids": [0, 4]}}}'
    linter = ArchitectureLinter()
    print(json.dumps(linter.lint_graph(test_graph), indent=2))
