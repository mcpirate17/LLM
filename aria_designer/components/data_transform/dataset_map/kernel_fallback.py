"""Python fallback kernel for dataset_map."""


class ComponentHandler:
    """Transforms a dataset by applying a Python expression to each row."""

    __slots__ = ()

    def validate_config(self, config):
        errors = []
        body = config.get("function_body", "x")
        if not isinstance(body, str) or not body.strip():
            errors.append("function_body must be a non-empty string")
        return errors

    def build(self, config):
        return None

    def forward(self, inputs, config):
        dataset = inputs.get("dataset", [])
        function_body = config.get("function_body", "x")
        if function_body == "x":
            return {"mapped": dataset}
        # Compile once for the batch
        code = compile(function_body, "<map_function>", "eval")
        mapped = [eval(code, {"__builtins__": {}}, {"x": x}) for x in dataset]  # noqa: S307
        return {"mapped": mapped}
