"""Compiler handlers for the NM-F forced-structure operator families.

Each handler dispatches to the self-contained NM-F mixer instantiated by the
matching ``_init_*`` in ``compiled_op_params_nmf.py`` (same convention as the
NM-C seam, ``compiler_ops_compaction.py``): the full class is the compiled op,
so search/synthesis and final model generation share exact semantics. Missing
submodules fail loud with ``AttributeError``.

Wired: F2 idempotent oblique memory, F3 nilpotent-Lie signature scan,
F4 integral-control mixer, F5 port-Hamiltonian mixer, F9 CDMA slot binding.
F6 (scale-equivariant wavelet) is deliberately NOT wired until its files are
committed by the owning lane. Lane docs live on the mixer classes and in
``tasks/nm_f_operator_families_2026-07-01.md``.
"""

from __future__ import annotations


def _op_idempotent_oblique_memory(module, inputs, _config):
    return module.oblique_memory_block(inputs[0])


def _op_nilpotent_lie_scan(module, inputs, _config):
    return module.lie_scan_block(inputs[0])


def _op_integral_control_mixer(module, inputs, _config):
    return module.integral_control_block(inputs[0])


def _op_port_hamiltonian_mix(module, inputs, _config):
    return module.port_hamiltonian_block(inputs[0])


def _op_cdma_slot_binding(module, inputs, _config):
    return module.cdma_binding_block(inputs[0])


def _op_scale_equivariant_wavelet(module, inputs, _config):
    return module.wavelet_block(inputs[0])


def _op_nonabelian_group_conv(module, inputs, _config):
    return module.group_conv_block(inputs[0])


OP_IMPLS = {
    "idempotent_oblique_memory": _op_idempotent_oblique_memory,
    "nilpotent_lie_scan": _op_nilpotent_lie_scan,
    "integral_control_mixer": _op_integral_control_mixer,
    "port_hamiltonian_mix": _op_port_hamiltonian_mix,
    "cdma_slot_binding": _op_cdma_slot_binding,
    "scale_equivariant_wavelet": _op_scale_equivariant_wavelet,
    "nonabelian_group_conv": _op_nonabelian_group_conv,
}
