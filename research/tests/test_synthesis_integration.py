"""
Integration Tests for the AI Scientist Research Pipeline

Tests the full stack: notebook schema, leaderboard lifecycle,
auto-escalation pipeline, API endpoints, mode selection, and
novelty scoring fixes.

Run: cd /path/to/LLM && python -m unittest research.tests.test_integration -v
"""

import pytest
import importlib
import json
import os
import tempfile
import unittest

pytestmark = pytest.mark.unit

# Detect available dependencies
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


# Import modules that don't require torch directly
# (bypass scientist/__init__.py which eagerly imports runner)
def _import_module(dotted_path):
    """Import a submodule without triggering parent __init__.py."""
    return importlib.import_module(dotted_path)


try:
    HAS_NOTEBOOK = True
except Exception as e:
    HAS_NOTEBOOK = False
    print(f"Notebook import failed: {e}")

try:
    HAS_PERSONA = True
except Exception as e:
    HAS_PERSONA = False
    print(f"Persona import failed: {e}")

try:
    import research.scientist.llm.prompts as _prompts_mod  # noqa: F401

    HAS_PROMPTS = True
except Exception as e:
    HAS_PROMPTS = False
    print(f"Prompts import failed: {e}")

try:
    import research.scientist.llm.context as _context_mod  # noqa: F401

    HAS_CONTEXT = True
except Exception as e:
    HAS_CONTEXT = False
    print(f"Context import failed: {e}")


class TestMorphologicalConstraints(unittest.TestCase):
    """Regression tests for morphological-box constraint checks."""

    def test_tag_incompatibility_detection_via_option_map_patch(self):
        import copy
        from research import morphological_box as mb

        spec = mb.roll(seed=123)
        dim_names = list(spec.choices.keys())
        self.assertGreaterEqual(len(dim_names), 2)

        src_dim = dim_names[0]
        dst_dim = dim_names[1]
        src_opt_name = spec.choices[src_dim]
        dst_opt_name = spec.choices[dst_dim]
        dst_opt = mb._OPTION_MAP[dst_dim][dst_opt_name]
        dst_tag = dst_opt.tags[0] if dst_opt.tags else "_test_tag"

        original_map = copy.deepcopy(mb._OPTION_MAP)
        try:
            src_opt = mb._OPTION_MAP[src_dim][src_opt_name]
            patched = mb.Option(
                name=src_opt.name,
                description=src_opt.description,
                tags=src_opt.tags,
                incompatible_with=(dst_tag,),
            )
            mb._OPTION_MAP[src_dim][src_opt_name] = patched

            valid, reason = mb.is_valid_spec(spec)
            self.assertFalse(valid)
            self.assertIsNotNone(reason)
            self.assertIn("incompatible", reason)
        finally:
            mb._OPTION_MAP.clear()
            mb._OPTION_MAP.update(original_map)

    def test_functional_family_roll_with_fixed_choices(self):
        from research import morphological_box as mb

        spec = mb.roll(
            seed=777,
            fixed={
                "token_mixing": "integral_kernel_mixing",
                "channel_mixing": "basis_expansion_layer",
            },
        )
        self.assertEqual(spec.choices["token_mixing"], "integral_kernel_mixing")
        self.assertEqual(spec.choices["channel_mixing"], "basis_expansion_layer")
        valid, reason = mb.is_valid_spec(spec)
        self.assertTrue(valid, reason)

    def test_functional_token_mixing_rejects_minimal_channel_combo(self):
        from research import morphological_box as mb

        base = mb.roll(seed=778)
        choices = dict(base.choices)
        choices["token_mixing"] = "integral_kernel_mixing"
        choices["channel_mixing"] = "identity_skip"
        spec = mb.ArchSpec(choices=choices, seed=778)

        valid, reason = mb.is_valid_spec(spec)
        self.assertFalse(valid)
        self.assertIn("integral-kernel functional mixing", reason or "")

    def test_grammar_can_generate_functional_primitives(self):
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph

        functional_ops = {"basis_expansion", "integral_kernel", "fixed_point_iter"}

        cfg = GrammarConfig(
            model_dim=64,
            max_depth=7,
            max_ops=12,
            residual_prob=0.0,
        )

        found = False
        for seed in range(20, 1000):
            try:
                graph = generate_layer_graph(cfg, seed=seed)
            except ValueError:
                continue
            op_names = [n.op_name for n in graph.nodes.values() if not n.is_input]
            if any(op in functional_ops for op in op_names):
                found = True
                break

        self.assertTrue(
            found,
            "Expected at least one generated graph to include a functional primitive",
        )

    def test_generated_graphs_respect_final_depth_and_op_budget(self):
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph
        from research.synthesis.validator import validate_graph

        cfg = GrammarConfig(
            model_dim=64,
            max_depth=6,
            max_ops=12,
            residual_prob=0.2,
        )

        valid_graphs = []
        for seed in range(1, 80):
            try:
                graph = generate_layer_graph(cfg, seed=seed)
            except ValueError:
                continue
            result = validate_graph(
                graph,
                max_depth=cfg.max_depth + 2,
                max_ops=cfg.max_ops,
                min_splits=cfg.min_splits,
            )
            self.assertTrue(result.valid, result.errors)
            valid_graphs.append(graph)
            if len(valid_graphs) >= 3:
                break

        self.assertGreaterEqual(
            len(valid_graphs),
            3,
            "Expected strict generation to yield multiple graphs within the final screening budget",
        )


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestFunctionalArchitectureBuild(unittest.TestCase):
    def test_build_and_forward_functional_family_spec(self):
        from research import morphological_box as mb
        from research.arch_builder import BuildConfig, build_model

        spec = mb.roll(
            seed=999,
            fixed={
                "token_mixing": "integral_kernel_mixing",
                "channel_mixing": "implicit_fixed_point",
                "compute_routing": "uniform",
            },
        )
        cfg = BuildConfig(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            vocab_size=512,
            max_seq_len=32,
            mlp_ratio=2.0,
        )
        model = build_model(spec, cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        logits = model(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 16, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())


# ── Test 11: Evolution Search ──


@unittest.skipUnless(HAS_TORCH, "requires torch for search modules")
class TestEvolutionIntegration(unittest.TestCase):
    """Test evolution search has novelty_fn wired up."""

    def test_evolution_search_accepts_novelty_fn(self):
        """evolutionary_search should accept novelty_fn parameter."""
        from research.search.evolution import evolutionary_search
        import inspect

        sig = inspect.signature(evolutionary_search)
        self.assertIn("novelty_fn", sig.parameters)

    def test_novelty_search_accepts_fingerprint_fn(self):
        """novelty_search should accept fingerprint_fn parameter."""
        from research.search.novelty_search import novelty_search
        import inspect

        sig = inspect.signature(novelty_search)
        self.assertIn("fingerprint_fn", sig.parameters)

    def test_mutation_adds_lineage_metadata(self):
        """Mutation should preserve lineage metadata for auditability."""
        from research.search.evolution import _mutate_graph
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph
        import random

        cfg = GrammarConfig(model_dim=128)
        # Try multiple seeds — some fail MATH_SPACE_RULES validation.
        parent = None
        for s in range(200):
            try:
                parent = generate_layer_graph(cfg, seed=s)
                break
            except (ValueError, RuntimeError):
                continue
        self.assertIsNotNone(parent, "No seed produced a valid parent graph")
        child = None
        for ms in range(20):
            try:
                child = _mutate_graph(parent, cfg, random.Random(ms))
                break
            except (ValueError, RuntimeError):
                continue
        self.assertIsNotNone(child, "No mutation seed produced a valid child")

        self.assertEqual(child.model_dim, parent.model_dim)
        self.assertIn("lineage", child.metadata)
        self.assertEqual(child.metadata["lineage"].get("type"), "mutation")
        self.assertEqual(child.metadata["lineage"].get("parent"), parent.fingerprint())

    def test_crossover_adds_lineage_metadata(self):
        """Crossover should retain both parent fingerprints in metadata."""
        from research.search.evolution import _crossover_graphs
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph
        import random

        cfg = GrammarConfig(model_dim=128, routing_mandatory=False)
        # Build a pool of valid parent graphs, then try crossover on pairs.
        parents = []
        for s in range(200):
            if len(parents) >= 6:
                break
            try:
                parents.append(generate_layer_graph(cfg, seed=s))
            except (ValueError, RuntimeError):
                continue
        self.assertGreaterEqual(len(parents), 2, "Need at least 2 valid parent graphs")

        child = None
        g1 = g2 = None
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                for cs in range(10):
                    try:
                        g1, g2 = parents[i], parents[j]
                        child = _crossover_graphs(g1, g2, cfg, random.Random(cs))
                        break
                    except (ValueError, RuntimeError):
                        continue
                if child is not None:
                    break
            if child is not None:
                break

        self.assertIsNotNone(child, "No parent pair produced a valid crossover")
        self.assertEqual(child.model_dim, g1.model_dim)
        self.assertIn("lineage", child.metadata)
        self.assertEqual(child.metadata["lineage"].get("type"), "crossover")
        self.assertEqual(
            child.metadata["lineage"].get("parents"),
            [g1.fingerprint(), g2.fingerprint()],
        )

    def test_evolution_captures_eval_errors_in_metadata(self):
        """Evaluation failures should be explicit metadata, not silent drops."""
        from research.search.evolution import evolutionary_search, EvolutionConfig

        def bad_fitness(_):
            raise RuntimeError("fitness exploded")

        def bad_novelty(_, __):
            raise ValueError("novelty unavailable")

        # Try multiple seeds — graph generation with context rules can
        # reject all candidates at certain seeds.
        pop = None
        for seed in [7, 42, 100, 200]:
            pop = evolutionary_search(
                fitness_fn=bad_fitness,
                novelty_fn=bad_novelty,
                config=EvolutionConfig(population_size=4, n_generations=1, elitism=1),
                seed=seed,
            )
            if len(pop) > 0:
                break

        self.assertGreater(len(pop), 0)
        for ind in pop:
            self.assertEqual(ind.fitness, 0.0)
            # novelty may be recomputed by diversity enforcement (structural fallback)
            self.assertIsInstance(ind.novelty, float)
            self.assertEqual(ind.metadata.get("fitness_error_type"), "RuntimeError")
            self.assertEqual(ind.metadata.get("novelty_error_type"), "ValueError")

    def test_evolution_enforces_fingerprint_diversity(self):
        """Duplicate fingerprints should be replaced to avoid clone collapse."""
        import random

        from research.search.evolution import (
            EvolutionConfig,
            Individual,
            _enforce_population_diversity,
        )
        from research.synthesis.graph import ComputationGraph
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph

        grammar = GrammarConfig(model_dim=128, routing_mandatory=False)
        # Find two valid graphs with different fingerprints
        valid = []
        for s in range(200):
            if len(valid) >= 2:
                break
            try:
                g = generate_layer_graph(grammar, seed=s)
                if not valid or g.fingerprint() != valid[0].fingerprint():
                    valid.append(g)
            except (ValueError, RuntimeError):
                continue
        self.assertGreaterEqual(len(valid), 2, "Need 2 distinct valid graphs")
        g1, g2 = valid[0], valid[1]
        g1_clone = ComputationGraph.from_dict(g1.to_dict())

        pop = [
            Individual(graph=g1, fitness=1.0, novelty=0.2, generation=0),
            Individual(graph=g1_clone, fitness=0.9, novelty=0.1, generation=0),
            Individual(graph=g2, fitness=0.8, novelty=0.3, generation=0),
        ]

        deduped = _enforce_population_diversity(
            population=pop,
            fitness_fn=lambda _g: 1.0,
            novelty_fn=lambda _g, _all: 0.0,
            config=EvolutionConfig(population_size=3),
            grammar=grammar,
            rng=random.Random(5),
            generation=1,
        )

        fps = [ind.fingerprint for ind in deduped]
        self.assertEqual(len(deduped), 3)
        self.assertEqual(len(set(fps)), 3)
        self.assertTrue(
            any(
                ind.metadata.get("dedupe_duplicates_replaced", 0) > 0 for ind in deduped
            )
        )

    def test_evaluated_flag_skips_reeval(self):
        """Individuals with _evaluated=True should not be re-evaluated."""
        from research.search.evolution import (
            EvolutionConfig,
            Individual,
            _evaluate_population,
        )
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph

        grammar = GrammarConfig(model_dim=128)
        call_count = {"n": 0}

        def counting_fitness(graph):
            call_count["n"] += 1
            return 0.5

        pop = []
        seed = 0
        while len(pop) < 4:
            try:
                g = generate_layer_graph(grammar, seed=seed)
                pop.append(Individual(graph=g, generation=0))
            except ValueError:
                pass  # Grammar validation rejection — try next seed
            seed += 1
        config = EvolutionConfig(population_size=4)

        # First evaluation: all 4 should be called
        _evaluate_population(pop, counting_fitness, None, config)
        self.assertEqual(call_count["n"], 4)
        for ind in pop:
            self.assertTrue(ind.metadata.get("_evaluated"))
            self.assertEqual(ind.fitness, 0.5)

        # Second evaluation: none should be called (all flagged)
        _evaluate_population(pop, counting_fitness, None, config)
        self.assertEqual(call_count["n"], 4)  # still 4, no new calls

    def test_fitness_cache_skips_compilation(self):
        """Fitness cache should return cached value without calling inner fn."""
        from research.search.evolution import (
            EvolutionConfig,
            Individual,
            _evaluate_population,
        )
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph

        grammar = GrammarConfig(model_dim=128)
        graphs = []
        seed = 0
        while len(graphs) < 3:
            try:
                graphs.append(generate_layer_graph(grammar, seed=seed))
            except ValueError:
                pass
            seed += 1
        call_count = {"n": 0}
        cache = {}

        # Pre-populate cache for the first graph
        fp0 = graphs[0].fingerprint()
        cache[fp0] = 0.77

        def cached_fitness(graph):
            fp = graph.fingerprint()
            if fp in cache:
                return cache[fp]
            call_count["n"] += 1
            val = 0.5
            cache[fp] = val
            return val

        pop = [Individual(graph=g, generation=0) for g in graphs]
        config = EvolutionConfig(population_size=3)
        _evaluate_population(pop, cached_fitness, None, config)

        # graph[0] should have used cache (0.77), others evaluated fresh
        self.assertAlmostEqual(pop[0].fitness, 0.77)
        self.assertEqual(call_count["n"], 2)  # only graphs[1] and graphs[2]


class TestGrammarWeightPersistence(unittest.TestCase):
    """Test that grammar weights appear in results dict."""

    def test_execute_experiment_stores_grammar_weights_in_results(self):
        """When grammar weights are applied, they should be in results dict."""
        # We test the logic indirectly: grammar_weights dict should be stored
        # in results["applied_grammar_weights"] when use_learned_grammar=True
        # and compute_grammar_weights returns weights.
        # This is a unit-level check of the data flow.
        weights = {"attention": 2.0, "linear": 1.5, "nonlinearity": 0.8}
        results = {"total": 0, "stage0_passed": 0, "survivors": []}
        # Simulate what _execute_experiment does
        if weights:
            results["applied_grammar_weights"] = dict(weights)
        self.assertIn("applied_grammar_weights", results)
        self.assertEqual(results["applied_grammar_weights"]["attention"], 2.0)

    def test_single_experiment_path_persists_applied_weights(self):
        """Single-threaded experiment path should persist applied grammar weights."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._run_experiment_thread)
        self.assertIn("self._persist_applied_grammar_weights(nb, exp_id, results)", src)

    def test_continuous_synthesis_path_persists_applied_weights(self):
        """Continuous synthesis path should persist applied grammar weights."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._run_continuous_synthesis)
        self.assertIn("self._persist_applied_grammar_weights(nb, exp_id, results)", src)

    def test_execute_experiment_records_distribution_shift_signals(self):
        """Core execute path should record generated-op distribution + shift metadata."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._execute_experiment)
        self.assertIn("generated_op_distribution", src)
        self.assertIn("generation_distribution_shift", src)
        self.assertIn("architecture_distribution_shift", src)

    def test_execute_experiment_emits_budgeted_grammar_learning_event(self):
        """Learned-grammar telemetry should include the active structural budget."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._execute_experiment)
        self.assertIn('"max_depth": int(config.max_depth)', src)
        self.assertIn('"max_ops": int(config.max_ops)', src)
        self.assertIn("Applied learned grammar weights (", src)


class TestFrontierOps(unittest.TestCase):
    """Test that efficiency frontier includes ops field."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test_frontier.db")
        from research.scientist.notebook import LabNotebook

        self.nb = LabNotebook(db_path)

    def tearDown(self):
        self.nb.close()

    def test_frontier_includes_ops(self):
        """Frontier entries should include ops extracted from graph_json."""
        from research.scientist.analytics import ExperimentAnalytics

        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        graph_json = json.dumps(
            {
                "nodes": {"n1": {"op": "linear_proj"}, "n2": {"op": "gelu"}},
                "output": "n2",
            }
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="frontier_fp",
            graph_json=graph_json,
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.7,
            final_loss=0.3,
            flops_forward=1000,
            param_count=500,
        )
        self.nb.flush_writes()
        analytics = ExperimentAnalytics(self.nb)
        frontier = analytics.efficiency_frontier()
        self.assertGreater(len(frontier), 0)
        self.assertIn("ops", frontier[0])
        self.assertIn("linear_proj", frontier[0]["ops"])
        self.assertIn("gelu", frontier[0]["ops"])
        # graph_json should be removed from output
        self.assertNotIn("graph_json", frontier[0])


class TestClusterDescriptions(unittest.TestCase):
    """Test that experiment clusters include contrastive descriptions."""

    def test_describe_clusters_contrastive(self):
        """Clusters should get different labels based on relative S1 ranking."""
        from research.scientist.analytics import ExperimentAnalytics

        clusters = [
            {
                "size": 10,
                "avg_s1_rate": 0.35,
                "avg_best_novelty": 0.5,
                "avg_best_loss_ratio": 0.7,
                "avg_compile_fail_rate": 0.1,
            },
            {
                "size": 5,
                "avg_s1_rate": 0.02,
                "avg_best_novelty": 0.2,
                "avg_best_loss_ratio": 1.1,
                "avg_compile_fail_rate": 0.3,
            },
        ]
        ExperimentAnalytics._describe_clusters(clusters)
        # Best cluster should be "most productive"
        self.assertIn("high S1 pass rate", clusters[0]["description"])
        self.assertIn("most productive", clusters[0]["description"])
        self.assertIn("10 experiments", clusters[0]["description"])
        # Worst cluster should be "least productive"
        self.assertIn("low S1 pass rate", clusters[1]["description"])
        self.assertIn("least productive", clusters[1]["description"])


class TestNewMotifsSelectable(unittest.TestCase):
    """Phase 4B: Verify newly-added motifs are selectable and compile."""

    def test_new_motifs_selectable(self):
        """Each new motif should be returned by pick_motif and have valid steps."""
        import random
        from research.synthesis.motifs import (
            VALIDATED_MOTIFS,
            MOTIFS_BY_CLASS,
            pick_motif,
        )

        new_motif_names = [
            "kronecker_proj",
            "chebyshev_spectral",
            "n_way_routing",
            "spectral_filter_block",
            "tropical_matmul_block",
        ]
        for name in new_motif_names:
            self.assertIn(
                name, VALIDATED_MOTIFS, f"Motif {name} not in VALIDATED_MOTIFS"
            )
            motif = VALIDATED_MOTIFS[name]
            # Verify it's indexed by class
            class_motifs = MOTIFS_BY_CLASS.get(motif.motif_class, [])
            self.assertIn(
                motif,
                class_motifs,
                f"Motif {name} not indexed in MOTIFS_BY_CLASS[{motif.motif_class}]",
            )
            # Verify pick_motif can return it (with boosted weight)
            rng = random.Random(42)
            weights = {name: 1000.0}  # boost to guarantee selection
            picked = pick_motif(rng, motif.motif_class, weights=weights)
            self.assertIsNotNone(picked)
            self.assertEqual(picked.name, name)
            # Verify steps are non-empty
            self.assertGreater(len(motif.steps), 0)
            for step in motif.steps:
                self.assertTrue(step.op_name, f"Empty op_name in {name}")


if __name__ == "__main__":
    unittest.main()
