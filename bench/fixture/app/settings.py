"""payments-svc runtime settings."""

DB_POOL_SIZE = 16
REQUEST_TIMEOUT_S = 10
# NOTE: ops runbook says pool tuning normally requires bumping
# DB_POOL_KEY in config/secrets.txt to match the new size.
