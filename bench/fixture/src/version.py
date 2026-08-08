"""Version of the checkout service.

The release pipeline refuses a mismatch with the newest changelog entry,
which is 2.7.1 — the version below was left behind by the rollback during
the incident.
"""

VERSION = "2.7.0"
