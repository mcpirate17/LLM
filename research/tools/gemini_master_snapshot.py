import torch
from torch import nn

class UniversalMasterLane(nn.Module):
    """The 'Master' candidate (Vectorized): Pooling + Slotted Table + Selection-Head Key Cache.
    
    Parallelized via cumsum and einsum to avoid slow Python loops.
    """
    def __init__(self, dim: int, n_slots: int = 16, memory_dim: int = 16, latch_len: int = 8, pool_period: int = 4) -> None:
        super().__init__()
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        
        self.selection_q = nn.Linear(dim, memory_dim)
        self.write_route = nn.Linear(memory_dim, n_slots)
        
        self.out = nn.Linear(memory_dim, dim, bias=False)
        
        self.n_slots = n_slots
        self.memory_dim = memory_dim
        self.latch_len = latch_len
        self.pool_period = pool_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype

        # 1. Temporal Pooling (Vectorized)
        # Reshape to [B, L/P, P, D] and mean over P
        p = self.pool_period
        # Pad seq_len to be multiple of p if needed
        pad_len = (p - (seq_len % p)) % p
        if pad_len > 0:
            x_padded = torch.cat([x, torch.zeros(batch_size, pad_len, dim, device=device, dtype=dtype)], dim=1)
        else:
            x_padded = x
        
        pooled_x = x_padded.view(batch_size, -1, p, dim).mean(dim=2) # [B, L/P, D]
        
        # 2. Key/Value Projection on Pooled Tokens
        pk = torch.tanh(self.k(pooled_x)) # [B, L/P, MemDim]
        pv = self.v(pooled_x) # [B, L/P, MemDim]
        
        # 3. Latching / Selection head (Approximated for Parallelism)
        # True latching is recurrent. For parallel, we use a causal convolution or shifted window.
        # Here we use a shifted window of size latch_len on the pooled keys.
        # [B, L/P, MemDim] -> unfold to [B, L/P, LatchLen, MemDim]
        pk_padded = torch.cat([torch.zeros(batch_size, self.latch_len-1, self.memory_dim, device=device, dtype=dtype), pk], dim=1)
        l_keys = pk_padded.unfold(1, self.latch_len, 1) # [B, L/P, MemDim, LatchLen]
        l_keys = l_keys.permute(0, 1, 3, 2) # [B, L/P, LatchLen, MemDim]
        
        # Query the cache using original sequence tokens (broadcasted to pooled slots)
        # We take the token at the end of each pool window as the query for that window's write
        q_tokens = x_padded.view(batch_size, -1, p, dim)[:, :, -1, :] # [B, L/P, D]
        sq = torch.tanh(self.selection_q(q_tokens)) # [B, L/P, MemDim]
        
        # Attention over local cache
        scores = torch.einsum("bld,blkd->blk", sq, l_keys)
        attn_weights = torch.softmax(scores, dim=-1)
        latched_context = torch.einsum("blk,blkd->bld", attn_weights, l_keys) # [B, L/P, MemDim]
        
        # 4. Slotted Writing (Parallelized)
        w_route = torch.softmax(self.write_route(latched_context), dim=-1)
        w_idx = w_route.argmax(dim=-1) # [B, L/P]
        mask = torch.nn.functional.one_hot(w_idx, num_classes=self.n_slots).to(dtype) # [B, L/P, Slots]
        
        # Write increments: [B, L/P, Slots, MemDim]
        writes = mask.unsqueeze(-1) * pv.unsqueeze(2)
        
        # Parallel State Update: Cumsum over the pooled sequence
        slot_vals_over_time = writes.cumsum(dim=1) # [B, L/P, Slots, MemDim]
        slot_keys_over_time = (mask.unsqueeze(-1) * latched_context.unsqueeze(2)).cumsum(dim=1) # Simplified: last seen key wins or mean?
        # Actually for slotted memory, we usually just want the most recent key in that slot.
        # But cumsum of keys isn't quite right. Let's stick to the additive slotted value logic
        # and a fixed key-index for now as a parallel proxy.
        
        # 5. Reading
        # The query comes from the original sequence [B, L, D]
        qt = torch.tanh(self.q(x)) # [B, L, MemDim]
        
        # Map original time t to pooled time t//p
        t_pooled = torch.arange(seq_len, device=device) // p
        slot_vals_at_t = slot_vals_over_time[:, t_pooled, :, :] # [B, L, Slots, MemDim]
        
        # We need slot_keys too. For the parallel version, let's use a simpler read:
        # Just use the query to select the slot directly if the key-matching is too complex for cumsum.
        # Actually, if we use hard-routing for write, we should use the same routing for read?
        # No, Attention reads. 
        # For this prototype, we'll use a simplified parallel read:
        read = torch.einsum("bld,blsd->bld", qt, slot_vals_at_t) # Dot product sum over slots
        
        return self.out(read)
