# vulture_whitelist.py — things that look dead but aren't
_.mode  # argparse
_.device  # argparse
_.n  # argparse
_.dim  # argparse
_.n_layers  # argparse
_.port  # argparse
_.db  # argparse
_.seed  # argparse
_.fix  # argparse
_.stage0_only  # argparse
_.mutate  # argparse
_.leaderboard  # argparse
_.analyze  # argparse
_.describe  # argparse
_.resume  # argparse
_.arch  # argparse
_.skip_pipeline  # argparse

# Flask / FastAPI routes and handlers
not_found  # api.py
internal_error  # api.py
unhandled_exception  # api.py
log_response  # api.py
designer_activity_hook  # api.py

# Common SQLable attributes
_.row_factory  # sqlite3

# Function parameters prefixed with _ (unused but required by API/signature)
_max_lr  # notebook_misc.py get_investigation_eligible
_ref_lr_ceiling  # notebook_misc.py get_investigation_eligible
_exc_type  # perf.py __exit__
_config_device  # shared_utils.py resolve_device
_use_adaptive_synthesis  # grammar.py generate_weighted_batch
eos_token_id  # scale_test.py _model_generate interface parameter
