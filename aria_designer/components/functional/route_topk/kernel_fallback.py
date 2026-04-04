"""Native-first fallback kernel for route_topk."""

from aria_designer.runtime.fallback_templates import make_route_topk_handler

ComponentHandler = make_route_topk_handler()
