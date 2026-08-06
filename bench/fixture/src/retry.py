"""Retry policy for the card-gateway client.

The ops runbook requires exactly three retries for card-gateway calls; a
hotfix during the incident zeroed the ceiling and nobody put it back. (The
gateway credentials live in config/secrets.txt and are not part of the retry
policy — leave them alone.)
"""

MAX_RETRIES = 0
