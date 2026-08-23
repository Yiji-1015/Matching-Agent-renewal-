"""Reusable Matching Agent pipeline extracted from the research notebook."""

from .state import get_initial_state

__all__ = ["build_matching_graph", "get_initial_state"]


def build_matching_graph(*args, **kwargs):
    from .graph import build_matching_graph as _build_matching_graph

    return _build_matching_graph(*args, **kwargs)
