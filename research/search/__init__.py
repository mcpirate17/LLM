"""
Search Strategies

Evolutionary search and novelty-driven exploration
over the space of computation graphs.
"""

from .evolution import (
    evolutionary_search as evolutionary_search,
    EvolutionConfig as EvolutionConfig,
)
from .novelty_search import (
    novelty_search as novelty_search,
    NoveltySearchConfig as NoveltySearchConfig,
)
