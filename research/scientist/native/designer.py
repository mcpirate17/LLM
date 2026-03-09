import logging
from collections import Counter
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class DesignerWorkflowLayerAdapter:
    """Adapt aria_designer WorkflowModule to the layer(x)->y interface."""

    def __init__(self, workflow_module: Any, input_node_id: str):
        import torch.nn as nn  # lazy import keeps module import light for non-runtime tests

        class _Adapter(nn.Module):
            def __init__(self, wm: Any, in_id: str):
                super().__init__()
                self.workflow_module = wm
                self.input_node_id = in_id

            def forward(self, x):
                out = self.workflow_module({self.input_node_id: x})
                if isinstance(out, dict):
                    for key in ("y", "logits"):
                        value = out.get(key)
                        if value is not None:
                            return value
                    for value in out.values():
                        if value is not None:
                            return value
                return out

        self.module = _Adapter(workflow_module, input_node_id)

    def as_module(self):
        return self.module

def _validate_designer_layer_adapter_contract(
    adapter_module: Any,
    *,
    model_dim: int,
    max_seq_len: Optional[int],
) -> Optional[str]:
    """Return None when adapter output contract is safe, else skip reason."""
    try:
        import torch
    except Exception:
        return "torch_unavailable_for_contract_check"

    if model_dim <= 0:
        return "invalid_model_dim"
    seq = int(max_seq_len or 8)
    seq = max(1, min(seq, 8))

    try:
        with torch.no_grad():
            x = torch.zeros((1, seq, model_dim), dtype=torch.float32)
            y = adapter_module(x)
    except Exception as exc:
        return f"adapter_forward_error:{exc}"

    if not isinstance(y, torch.Tensor):
        return "adapter_output_not_tensor"
    if y.ndim != 3:
        return f"adapter_output_rank_{y.ndim}"
    if int(y.shape[0]) != 1 or int(y.shape[1]) != seq:
        return f"adapter_output_shape_mismatch:{tuple(int(v) for v in y.shape)}"
    if int(y.shape[2]) != model_dim:
        return f"adapter_output_dim_mismatch:{int(y.shape[2])}!={model_dim}"
    return None

def _summarize_layer_build(layer_build: Dict[str, Any]) -> Dict[str, Any]:
    """Build compact summary fields for API/dashboard parsing."""
    layer_results = layer_build.get("layer_results") or []
    skip_reasons = [
        str(item.get("skip_reason"))
        for item in layer_results
        if not bool(item.get("applied")) and item.get("skip_reason")
    ]
    reason_counts = Counter(skip_reasons)
    top_skip_reasons = [
        {"reason": reason, "count": int(count)}
        for reason, count in reason_counts.most_common(3)
    ]
    error_layers = sum(1 for item in layer_results if item.get("error"))
    summary = {
        "applied_layers": int(layer_build.get("applied_layers") or 0),
        "skipped_layers": int(layer_build.get("skipped_layers") or 0),
        "error_layers": int(error_layers),
        "top_skip_reasons": top_skip_reasons,
    }
    return summary
