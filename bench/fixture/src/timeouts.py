"""TLS handshake timeout for the card gateway.

Alerting math divides by this value, so it must be an integer count of
milliseconds (two hundred and fifty today), not a string — the 09:14 page in
alerts.txt is what the string version caused.
"""

TLS_HANDSHAKE_TIMEOUT_MS = "250"
