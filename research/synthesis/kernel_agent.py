"""LLM-driven Triton kernel generation and verification agent.

Extracts high-performing subgraphs, prompts an LLM to produce fused
Triton kernels, verifies numerical equivalence against PyTorch reference
implementations, and profiles for speedup.

Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §4
"""

import logging
import time
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class KernelAgent:
    """Autonomous agent for generating and verifying fused Triton kernels.

    Workflow:
    1. Extract PyTorch code for a subgraph.
    2. Prompt an LLM to generate a fused Triton kernel.
    3. Compile, verify numerical equivalence, and profile.
    4. If verification fails, feed errors back to the LLM for refinement.

    Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §4
    """

    def __init__(self, llm_client: Any, max_retries: int = 3, atol: float = 1e-3):
        """
        Args:
            llm_client: Object with a .generate(prompt) -> str method.
            max_retries: Max LLM refinement rounds on failure.
            atol: Absolute tolerance for numerical verification.
        """
        self.llm = llm_client
        self.max_retries = max_retries
        self.atol = atol

    def generate_triton_kernel(self, pytorch_code: str, math_description: str) -> str:
        """Ask the LLM to produce a fused Triton kernel from PyTorch code.

        Args:
            pytorch_code: Source code of the PyTorch module.
            math_description: Human-readable mathematical context.

        Returns:
            Generated Python code containing @triton.jit kernel + wrapper.
        """
        prompt = (
            "Convert the following PyTorch module into a highly optimized, "
            "fused Triton kernel. Maximize SRAM usage and minimize global "
            "memory reads.\n\n"
            f"PyTorch Code:\n```python\n{pytorch_code}\n```\n\n"
            f"Mathematical Context:\n{math_description}\n\n"
            "Output ONLY valid Python code containing the Triton @triton.jit "
            "kernel and a PyTorch wrapper function named `triton_wrapper`."
        )
        return self.llm.generate(prompt)

    def _compile_kernel(self, triton_code: str) -> Tuple[bool, Any]:
        """Dynamically compile generated Triton code.

        Returns:
            (success, wrapper_fn_or_error_msg)
        """
        local_env: dict = {}
        try:
            exec(triton_code, {"torch": torch, "__builtins__": __builtins__}, local_env)  # noqa: S102
        except Exception as e:
            return False, f"Compilation error: {e}"

        wrapper = local_env.get('triton_wrapper')
        if wrapper is None:
            return False, "No `triton_wrapper` function found in generated code."
        return True, wrapper

    def verify_and_profile(
        self,
        triton_code: str,
        pytorch_module: nn.Module,
        input_shape: tuple,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[bool, Any]:
        """Compile, verify, and profile a generated Triton kernel.

        Args:
            triton_code: Generated Python/Triton source.
            pytorch_module: Reference PyTorch module.
            input_shape: Shape of the test input tensor.
            device: Device to run on.
            dtype: Data type for verification.

        Returns:
            (success, info_dict_or_error_string)
        """
        ok, result = self._compile_kernel(triton_code)
        if not ok:
            return False, result
        triton_wrapper = result

        x = torch.randn(input_shape, device=device, dtype=dtype)
        pytorch_module = pytorch_module.to(device=device, dtype=dtype)

        # Numerical verification
        try:
            with torch.no_grad():
                y_ref = pytorch_module(x)
                y_triton = triton_wrapper(x)
            if not torch.allclose(y_ref, y_triton, atol=self.atol, rtol=1e-3):
                max_diff = (y_ref - y_triton).abs().max().item()
                return False, f"Numerical mismatch: max_diff={max_diff:.6f}"
        except Exception as e:
            return False, f"Runtime error: {e}"

        # Profiling
        try:
            import triton as _triton  # noqa: F811

            ms_pytorch = _triton.testing.do_bench(lambda: pytorch_module(x))
            ms_triton = _triton.testing.do_bench(lambda: triton_wrapper(x))
            speedup = ms_pytorch / max(ms_triton, 1e-6)
        except ImportError:
            # Triton not available — fall back to basic timing
            def _time_fn(fn, n=50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n * 1000  # ms

            ms_pytorch = _time_fn(lambda: pytorch_module(x))
            ms_triton = _time_fn(lambda: triton_wrapper(x))
            speedup = ms_pytorch / max(ms_triton, 1e-6)

        return True, {
            "speedup": round(speedup, 3),
            "ms_pytorch": round(ms_pytorch, 4),
            "ms_triton": round(ms_triton, 4),
        }

    def run(
        self,
        pytorch_code: str,
        math_description: str,
        pytorch_module: nn.Module,
        input_shape: tuple,
        device: str = "cuda",
    ) -> dict:
        """Full agentic loop: generate → verify → refine.

        Returns:
            Dict with success, code, profile, and attempt count.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            logger.info("KernelAgent attempt %d/%d", attempt, self.max_retries)

            if attempt == 1:
                code = self.generate_triton_kernel(pytorch_code, math_description)
            else:
                # Feed error back for refinement
                refinement_prompt = (
                    f"The previous Triton kernel failed verification.\n"
                    f"Error: {last_error}\n\n"
                    f"Original PyTorch code:\n```python\n{pytorch_code}\n```\n\n"
                    f"Previous attempt:\n```python\n{code}\n```\n\n"
                    "Fix the issue and output ONLY the corrected Python code "
                    "with @triton.jit kernel and `triton_wrapper` function."
                )
                code = self.llm.generate(refinement_prompt)

            ok, result = self.verify_and_profile(
                code, pytorch_module, input_shape, device=device
            )
            if ok:
                return {
                    "success": True,
                    "code": code,
                    "profile": result,
                    "attempts": attempt,
                }
            last_error = result
            logger.warning("Attempt %d failed: %s", attempt, last_error)

        return {
            "success": False,
            "error": last_error,
            "attempts": self.max_retries,
        }
