"""Synthetic datasets for Gemini trajectory metrics.

Two corpora that test specific reasoning capabilities at small budgets where
real-text capability probes are at the noise floor:

* ``dyck_sequences`` — balanced-bracket sequences. Every token's correct
  next-token distribution depends purely on the bracket-stack history.
  Architectures with working in-context attention show steeply decreasing
  per-position loss; bag-of-tokens models stay flat.

* ``transitive_triples`` — synthetic relational facts ``A is X. X is Y.``
  with the held-out target ``Y`` for query ``A``. Architectures capable of
  relational composition produce a monotonically growing logit margin on
  the target token during training; bottlenecked architectures plateau.

Both datasets use a small dedicated vocabulary so the difficulty stays
fixed across architectures and we don't need a real tokenizer.
"""

from __future__ import annotations

import torch

# Vocabulary layout for Dyck:
#   0          : padding/unused
#   1, 2       : '(' bracket types — paired open/close brackets
#   3, 4       : ')' bracket types
#   5..MAX-1   : neutral filler tokens to hold model_dim padding constant
# We use 2 bracket pairs to make the language non-trivial (a stack with
# two distinct symbols).
DYCK_VOCAB_SIZE = 16
DYCK_OPEN_TOKENS = (1, 2)
DYCK_CLOSE_TOKENS = (3, 4)
DYCK_OPEN_TO_CLOSE = dict(zip(DYCK_OPEN_TOKENS, DYCK_CLOSE_TOKENS))
DYCK_CLOSE_TO_OPEN = dict(zip(DYCK_CLOSE_TOKENS, DYCK_OPEN_TOKENS))


def dyck_sequences(
    *,
    batch_size: int,
    seq_len: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate batched balanced Dyck sequences.

    Each sequence is a valid balanced bracketing using the two pairs
    declared above. The returned tensor shape is ``(batch_size, seq_len)``
    with dtype ``int64``. Sequences are filled deterministically given
    the supplied generator.

    Algorithm: at each position, randomly choose between opening a new
    bracket (if depth allows and we have room to close) or closing the
    top of the stack. Token IDs are one of ``DYCK_OPEN_TOKENS`` /
    ``DYCK_CLOSE_TOKENS``. The stack tracks which open bracket needs
    closing so the closing token type is forced — that's where
    in-context attention earns its keep.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2 for Dyck generation")

    out = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=dev)

    rand_buffer = torch.rand(
        (batch_size, seq_len),
        generator=generator,
        device=dev,
    )
    open_choice_buffer = torch.randint(
        0,
        len(DYCK_OPEN_TOKENS),
        (batch_size, seq_len),
        generator=generator,
        device=dev,
    )

    # We can't vectorize the stack push/pop cleanly across positions, so
    # do the sequence loop on-CPU for correctness, then ship to device.
    rand_cpu = rand_buffer.cpu()
    open_cpu = open_choice_buffer.cpu()
    out_cpu = torch.zeros_like(out, device="cpu")

    for b in range(batch_size):
        stack: list[int] = []
        for t in range(seq_len):
            remaining = seq_len - t
            depth = len(stack)
            # Need to leave room to close everything still on the stack.
            must_close = depth >= remaining
            must_open = depth == 0
            if must_open or (not must_close and rand_cpu[b, t].item() < 0.5):
                tok = DYCK_OPEN_TOKENS[int(open_cpu[b, t].item())]
                stack.append(tok)
                out_cpu[b, t] = tok
            else:
                top = stack.pop()
                close_tok = DYCK_OPEN_TO_CLOSE[top]
                out_cpu[b, t] = close_tok

    out.copy_(out_cpu.to(dev))
    return out


# Vocabulary layout for transitive triples:
#   0          : padding/separator '.'
#   1..N_ENT   : entity tokens (A, X, Y, etc.)
# We pre-allocate a small entity pool. Triples take the form
#   ``A is X . X is Y . A is ?`` (ground truth Y).
# We pack triples as ``[A, sep, X, sep, X, sep, Y, sep, A, sep]`` with the
# 11th token being the target Y — a fixed query-position structure so the
# probe is uniform across batches.
TRANSITIVE_VOCAB_SIZE = 64
TRANSITIVE_SEP_TOKEN = 0
TRANSITIVE_ENTITY_RANGE = (1, TRANSITIVE_VOCAB_SIZE)
TRANSITIVE_QUERY_LEN = 10  # length up to and including the position right
# before the target token
TRANSITIVE_TARGET_INDEX = TRANSITIVE_QUERY_LEN  # index of target Y in seq


def transitive_triples(
    *,
    batch_size: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate transitive-relation triples for the logit-margin probe.

    Returns:
        inputs: shape ``(batch_size, TRANSITIVE_QUERY_LEN + 1)`` int64.
            Layout: ``[A, sep, X, sep, X, sep, Y, sep, A, sep, Y]``.
            Position ``TRANSITIVE_TARGET_INDEX`` carries Y and is the
            position the model must predict.
        targets: shape ``(batch_size,)`` int64 — Y for each row.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    lo, hi = TRANSITIVE_ENTITY_RANGE
    n_entities = hi - lo

    if n_entities < 3:
        raise ValueError("entity pool too small for transitive triples")

    base = torch.randint(
        lo,
        hi,
        (batch_size, 3),
        generator=generator,
        device=dev,
    )
    # Ensure distinctness: re-roll any row with collisions.
    for _ in range(8):  # bounded retries; n_entities >> 3 makes collisions rare
        a = base[:, 0]
        x = base[:, 1]
        y = base[:, 2]
        bad = (a == x) | (a == y) | (x == y)
        if not bad.any():
            break
        n_bad = int(bad.sum().item())
        replacement = torch.randint(lo, hi, (n_bad, 3), generator=generator, device=dev)
        base[bad] = replacement
    a = base[:, 0]
    x = base[:, 1]
    y = base[:, 2]

    sep = TRANSITIVE_SEP_TOKEN
    seq_len = TRANSITIVE_QUERY_LEN + 1

    # Build sequences columnwise so each row gets the same template.
    inputs = torch.full(
        (batch_size, seq_len),
        sep,
        dtype=torch.int64,
        device=dev,
    )
    inputs[:, 0] = a
    inputs[:, 1] = sep
    inputs[:, 2] = x
    inputs[:, 3] = sep
    inputs[:, 4] = x
    inputs[:, 5] = sep
    inputs[:, 6] = y
    inputs[:, 7] = sep
    inputs[:, 8] = a
    inputs[:, 9] = sep
    inputs[:, 10] = y  # target token at TRANSITIVE_TARGET_INDEX

    return inputs, y
