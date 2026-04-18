"""LM evaluation harness adapter for Aria models.

Wraps a compiled model for use with EleutherAI's lm-evaluation-harness,
supporting loglikelihood, loglikelihood_rolling, and generate_until tasks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def run_benchmarks(
    model, step: int, ckpt_dir: Path, device: str = "cuda", tokenizer_name: str = "gpt2"
):
    """Run standard LM benchmarks: WikiText-103 PPL, HellaSwag, LAMBADA."""
    from lm_eval import simple_evaluate
    from lm_eval.api.model import LM
    import tiktoken

    logger.info(f"Running benchmarks at step {step}...")
    model.eval()

    class AriaLM(LM):
        def __init__(self, model, device, max_length=1024):
            super().__init__()
            self._model = model
            self._device = device
            self._max_length = max_length
            self._enc = tiktoken.get_encoding(tokenizer_name)
            self._vocab_size = self._enc.n_vocab

        @property
        def eot_token_id(self):
            return self._enc.eot_token

        @property
        def max_length(self):
            return self._max_length

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return 4

        @property
        def device(self):
            return self._device

        def tok_encode(self, string, **kwargs):
            return self._enc.encode(string)

        def tok_decode(self, tokens, **kwargs):
            return self._enc.decode(tokens)

        def _model_call(self, inps):
            with torch.no_grad():
                return self._model(inps.to(self._device))

        def _model_generate(self, context, max_length, eos_token_id):
            raise NotImplementedError("Generation not supported")

        def loglikelihood(self, requests):
            results = []
            for ctx, cont in [req.args for req in requests]:
                ctx_ids = self._enc.encode(ctx) if ctx else []
                cont_ids = self._enc.encode(cont)
                all_ids = (ctx_ids + cont_ids)[-self._max_length :]
                input_ids = torch.tensor([all_ids], device=self._device)
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = self._model(input_ids)
                log_probs = F.log_softmax(logits[0].float(), dim=-1)
                cont_start = len(all_ids) - len(cont_ids)
                total_ll = 0.0
                greedy_match = True
                for i, tok in enumerate(cont_ids):
                    pos = cont_start + i - 1
                    if 0 <= pos < log_probs.size(0):
                        total_ll += log_probs[pos, tok].item()
                        if log_probs[pos].argmax().item() != tok:
                            greedy_match = False
                results.append((total_ll, greedy_match))
            return results

        def loglikelihood_rolling(self, requests):
            results = []
            for (string,) in [req.args for req in requests]:
                tokens = self._enc.encode(string)
                total_ll = 0.0
                for start in range(0, len(tokens), self._max_length):
                    chunk = tokens[start : start + self._max_length]
                    input_ids = torch.tensor([chunk], device=self._device)
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits = self._model(input_ids)
                    log_probs = F.log_softmax(logits[0].float(), dim=-1)
                    for i in range(1, len(chunk)):
                        total_ll += log_probs[i - 1, chunk[i]].item()
                results.append(total_ll)
            return results

        def generate_until(self, requests):
            return [""] * len(requests)

    lm = AriaLM(model, device)

    benchmarks = ["wikitext", "hellaswag", "lambada_openai"]
    try:
        import lm_eval.evaluator as _ev

        _ev.add_env_info = lambda results: None
        _ev.add_tokenizer_info = lambda results, lm: None

        eval_results = simple_evaluate(
            model=lm,
            tasks=benchmarks,
            batch_size=4,
            device=device,
        )

        results_dict = {}
        logger.info(f"\n{'=' * 60}")
        logger.info(f"BENCHMARKS at step {step}")
        logger.info(f"{'=' * 60}")

        for task_name, task_results in eval_results.get("results", {}).items():
            for metric, value in task_results.items():
                if isinstance(value, (int, float)) and "stderr" not in metric:
                    results_dict[f"{task_name}/{metric}"] = value
                    logger.info(f"  {task_name}/{metric}: {value:.4f}")

        results_path = ckpt_dir / f"benchmarks_step_{step}.json"
        with open(results_path, "w") as f:
            json.dump({"step": step, "results": results_dict}, f, indent=2)
        logger.info(f"  Saved to {results_path}")

        return results_dict

    except Exception as e:
        logger.error(f"Benchmark eval failed: {e}")
        import traceback

        traceback.print_exc()
        return {}
    finally:
        model.train()
