import sqlite3
import json
from collections import Counter
from typing import List
import logging

from research.synthesis.graph import ComputationGraph


logger = logging.getLogger(__name__)

def _graph_edge_bigrams(g: ComputationGraph) -> Counter:
    edges = Counter()
    for nid, node in g.nodes.items():
        if node.is_input:
            continue
        for in_id in node.input_ids:
            if in_id in g.nodes:
                edges[(g.nodes[in_id].op_name, node.op_name)] += 1
    return edges

def graph_structural_similarity(g1: ComputationGraph, g2: ComputationGraph) -> float:
    """Compute structural similarity scalar [0.0, 1.0]."""
    ops1 = Counter([n.op_name for n in g1.nodes.values() if not n.is_input])
    ops2 = Counter([n.op_name for n in g2.nodes.values() if not n.is_input])
    
    op_int = sum((ops1 & ops2).values())
    op_un = sum((ops1 | ops2).values())
    op_sim = op_int / max(1, op_un)
    
    e1 = _graph_edge_bigrams(g1)
    e2 = _graph_edge_bigrams(g2)
    e_int = sum((e1 & e2).values())
    e_un = sum((e1 | e2).values())
    edge_sim = e_int / max(1, e_un)
    
    return (0.4 * op_sim) + (0.6 * edge_sim)

class PhenotypicPruner:
    """Tracks known dead branches and filters new graphs that are too similar."""
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.dead_branches: List[ComputationGraph] = []

    def load_dead_branches_from_db(self, db_path: str):
        """Load architectures failing the scaling gate from program_results."""
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT graph_json FROM program_results WHERE scaling_gate_passed = 0 AND validation_loss_ratio > 1.5"
            ).fetchall()
            
            for r in rows:
                try:
                    data = json.loads(r["graph_json"])
                    if "nodes" in data:
                        g = ComputationGraph.from_dict(data)
                        self.dead_branches.append(g)
                except Exception:
                    pass
            logger.info(f"Loaded {len(self.dead_branches)} phenotypic dead branches from DB.")
        except Exception as e:
            logger.warning(f"Failed to load dead branches: {e}")

    def add_dead_branch(self, graph: ComputationGraph):
        self.dead_branches.append(graph)

    def is_pruned(self, graph: ComputationGraph) -> bool:
        """Returns True if graph should be pruned because it matches a dead branch."""
        if not self.dead_branches:
            return False
            
        for db in self.dead_branches:
            sim = graph_structural_similarity(graph, db)
            if sim >= self.similarity_threshold:
                return True
        return False
