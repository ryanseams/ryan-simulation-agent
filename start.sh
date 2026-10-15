#!/usr/bin/env bash
# Runs the web server and the LiveKit agent worker in one Railway service.
# If either process exits, exit so Railway restarts the container.
set -euo pipefail
python server.py &
python agent.py start &
wait -n
exit $?
