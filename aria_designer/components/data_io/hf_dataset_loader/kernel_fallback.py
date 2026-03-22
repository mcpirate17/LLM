"""Python handler for HF Dataset Loader."""

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class ComponentHandler:
    def validate_config(self, config):
        if load_dataset is None:
            return ["'datasets' library not installed. Run 'pip install datasets'."]
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        if load_dataset is None:
            raise RuntimeError("'datasets' library not installed")

        path = config.get("path", "wikitext")
        name = config.get("name")
        split = config.get("split", "train")
        streaming = config.get("streaming", True)

        dataset = load_dataset(path, name=name, split=split, streaming=streaming)
        return {"dataset": dataset}
