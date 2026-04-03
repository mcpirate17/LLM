"""Python fallback kernel for dataset_filter."""


class ComponentHandler:
    """Filters a dataset using a Python condition expression."""

    __slots__ = ()

    def validate_config(self, config):
        errors = []
        cond = config.get("condition", "True")
        if not isinstance(cond, str) or not cond.strip():
            errors.append("condition must be a non-empty string")
        return errors

    def build(self, config):
        return None

    def forward(self, inputs, config):
        dataset = inputs.get("dataset", [])
        condition = config.get("condition", "True")
        if condition == "True":
            return {"filtered": dataset}
        # Compile once for the batch
        code = compile(condition, "<filter_condition>", "eval")
        filtered = [x for x in dataset if eval(code, {"__builtins__": {}}, {"x": x})]  # noqa: S307
        return {"filtered": filtered}
