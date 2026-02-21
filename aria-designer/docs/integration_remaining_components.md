# Remaining Bridge Gaps (Pass 3)

Status as of 2026-02-21 after advanced-family semantic alias pass:
- Total components: 176
- Bridge-supported (direct + alias + source + template + passthrough + IO): 176
- Remaining unsupported: 0

## Completed in Pass 3

Added non-identity bridge semantics for previously unsupported advanced families:
- `channel_mixing/*`: mapped to `fixed_point_iter`, `basis_expansion`, `moe_topk`
- `linear_algebra/*` (remaining set): mapped to `basis_expansion`, `linear_proj`, `outer_product`
- `normalization/*` (remaining set): mapped to `norm_last`, `rmsnorm`
- `positional/*` (remaining set): mapped to `learnable_bias`, `conv1d_seq`, `fourier_mixing`
- `representation/*` (remaining set): mapped to `sign_ste`, `fourier_mixing`, `moe_topk`, `basis_expansion`, `multi_head_mix`, `topk_gate`

Also normalized stale aliases:
- `low_rank` → `linear_proj`
- `shared_basis` → `basis_expansion`

## Next Integration Focus

- Improve fidelity of semantic lowerings for advanced families beyond alias-level approximations.
- Add targeted compile/eval parity tests per advanced family component.
- Add runtime warnings/metadata that label approximated mappings when strict semantics are required.
