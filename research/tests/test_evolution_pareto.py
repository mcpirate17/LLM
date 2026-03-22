"""Tests for vectorized NSGA-II fast_non_dominated_sort."""

from unittest.mock import MagicMock

from research.search.evolution import Individual, fast_non_dominated_sort


def _make_individual(fitness: float, novelty: float) -> Individual:
    graph = MagicMock()
    graph.fingerprint.return_value = f"fp_{fitness}_{novelty}"
    return Individual(graph=graph, fitness=fitness, novelty=novelty)


def test_two_objective_known_fronts():
    """Verify correct Pareto fronts on a known 2-objective case.

    Layout (fitness, novelty):
        A(3, 1)  B(1, 3)  — Pareto front 1 (non-dominated)
        C(2, 1)  D(1, 2)  — Pareto front 2
        E(1, 1)            — Pareto front 3
    """
    a = _make_individual(3.0, 1.0)
    b = _make_individual(1.0, 3.0)
    c = _make_individual(2.0, 1.0)
    d = _make_individual(1.0, 2.0)
    e = _make_individual(1.0, 1.0)

    fronts = fast_non_dominated_sort([a, b, c, d, e])

    assert len(fronts) == 3

    # Front 0 (rank 1): A and B are non-dominated
    front0_set = set(id(ind) for ind in fronts[0])
    assert id(a) in front0_set
    assert id(b) in front0_set
    assert len(fronts[0]) == 2

    # Front 1 (rank 2): C and D
    front1_set = set(id(ind) for ind in fronts[1])
    assert id(c) in front1_set
    assert id(d) in front1_set
    assert len(fronts[1]) == 2

    # Front 2 (rank 3): E
    assert fronts[2] == [e]

    # Verify pareto_rank assignments
    assert a.pareto_rank == 1
    assert b.pareto_rank == 1
    assert c.pareto_rank == 2
    assert d.pareto_rank == 2
    assert e.pareto_rank == 3


def test_empty_population():
    assert fast_non_dominated_sort([]) == []


def test_single_individual():
    ind = _make_individual(1.0, 1.0)
    fronts = fast_non_dominated_sort([ind])
    assert len(fronts) == 1
    assert fronts[0] == [ind]
    assert ind.pareto_rank == 1


def test_all_identical():
    """All identical individuals form a single non-dominated front."""
    inds = [_make_individual(1.0, 1.0) for _ in range(5)]
    fronts = fast_non_dominated_sort(inds)
    assert len(fronts) == 1
    assert len(fronts[0]) == 5


def test_minimization_objective():
    """Test with a min objective (lower = better)."""
    a = _make_individual(3.0, 1.0)  # high fitness, low novelty (good for min)
    b = _make_individual(
        1.0, 3.0
    )  # low fitness (good for max), high novelty (bad for min)

    objectives = [("fitness", "max"), ("novelty", "min")]
    fronts = fast_non_dominated_sort([a, b], objectives=objectives)

    # Both are non-dominated: a is better on both (higher fitness, lower novelty)
    # Actually a dominates b: fitness 3>1 (max), novelty 1<3 (min, so 1 is better)
    assert len(fronts) == 2
    assert fronts[0] == [a]
    assert fronts[1] == [b]
