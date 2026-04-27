#!/bin/bash
# Helper: source SSH agent and run command on gautschi
source /tmp/ssh-agent-env 2>/dev/null
ssh -F ~/.ssh/config -o ConnectTimeout=15 gautschi "$@"
