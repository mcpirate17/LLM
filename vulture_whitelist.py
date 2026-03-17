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
