"""Python fallback kernel for loop control flow."""


class ComponentHandler:
    """Iterates over items and passes each one through.

    In fallback mode, the subgraph execution is not available — this simply
    passes the items list through as results.
    """

    __slots__ = ()

    def validate_config(self, config):
        errors = []
        max_iter = config.get("max_iterations", 100)
        if not isinstance(max_iter, int) or max_iter < 1:
            errors.append("max_iterations must be int >= 1")
        return errors

    def build(self, config):
        return None

    def forward(self, inputs, config):
        items = inputs.get("items", [])
        max_iter = config.get("max_iterations", 100)
        if hasattr(items, "__len__"):
            items = list(items)[:max_iter]
        return {"item": items[-1] if items else None, "results": items}
