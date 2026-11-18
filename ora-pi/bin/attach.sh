#!/bin/bash
# Spike harness: Pi's terminal UI in this terminal *as* the agent core of a run
# whose Supervisor waits on SOCKET (ORA_PI_ATTACH_SOCKET), under the same
# file-permission sandbox the headless core uses.
#   ora-pi/bin/attach.sh RUN_ROOT SOCKET
# Operators use `ora interactive` / `ora attach` instead: the core stays
# headless under the Supervisor and serves its TUI (ora-pi/src/tui/).
#
# Pi's TUI probes every ancestor directory for .git and .agents/skills; under
# the sandbox those probes read as absent (ora-pi/src/tui/sandbox.ts), so no
# path outside the run needs to be readable.
set -euo pipefail
RUN_ROOT=$(realpath "$1")
SOCKET=$2
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$RUN_ROOT/native/pi-agent"
export PI_OFFLINE=1 HOME="$RUN_ROOT/native/pi-agent"
cd "$RUN_ROOT/native"
exec node --permission \
  --allow-fs-read="$HERE" --allow-fs-read="$RUN_ROOT/native" --allow-fs-read="$RUN_ROOT/agent-events.jsonl" \
  --allow-fs-write="$RUN_ROOT/native" --allow-fs-write="$RUN_ROOT/agent-events.jsonl" \
  "$HERE/src/main.ts" --attach "$SOCKET"
