from __future__ import annotations

import importlib

_LAZY_EXPORTS = {
    "ApiRouteContext": ".deps",
    "register_analytics_routes": ".analytics_bp",
    "register_experiments_routes": ".experiments_bp",
    "register_programs_routes": ".programs_bp",
    "register_reporting_routes": ".reporting_bp",
    "register_strategy_bp_routes": ".strategy_bp",
    "register_general_routes": ".general_bp",
    "register_chat_routes": ".chat_bp",
    "register_leaderboard_routes": ".leaderboard_bp",
    "register_native_routes": ".native_bp",
    "register_campaigns_routes": ".campaigns_bp",
    "register_knowledge_routes": ".knowledge_bp",
    "register_actions_routes": ".actions_bp",
    "register_diagnostics_routes": ".diagnostics_bp",
    "register_config_routes": ".config_bp",
    "register_events_routes": ".events_bp",
    "register_system_routes": ".system_bp",
    "register_designer_routes": ".designer_bp",
    "register_misc_routes": ".misc_bp",
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)
