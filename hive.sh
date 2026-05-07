#!/bin/bash
# FILE: hive.sh
# Hive Orchestrator: Launches the LLM signal bus and agents.

PROJECT_ROOT="/home/tim/Projects/LLM"
VENV_PATH="/home/tim/venvs/llm/bin/activate"

# 1. Environment Setup
if [ -f "$VENV_PATH" ]; then
    # shellcheck source=/home/tim/venvs/llm/bin/activate
    # shellcheck disable=SC1091
    source "$VENV_PATH"
else
    echo "⚠️  Warning: venv not found at $VENV_PATH"
fi

cd "$PROJECT_ROOT" || exit

# 2. Export Bus Path for all sub-processes
export LLM_BUS_SOCKET="/tmp/llm_hive.sock"
export LLM_MODE="YOLO"

echo "🐝 Initializing the Hive..."

# 3. Start Signal Broker (The Brain)
python3 research/tools/hive/signal_broker.py &
BROKER_PID=$!
sleep 1 # Wait for socket to initialize

# 4. Start Ollama Bridge
python3 research/tools/hive/ollama_bridge.py &
OLLAMA_PID=$!

# 5. Define cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down the Hive..."
    kill $BROKER_PID $OLLAMA_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# 6. Launch a simple Bus Monitor in this pane
echo "📊 Hive Monitor Active. Listening for signals..."
echo "------------------------------------------------"

# This part just echoes anything that comes across the bus
# using a tiny python snippet to keep it simple.
python3 -c "
import socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect('$LLM_BUS_SOCKET')
    while True:
        data = s.recv(4096)
        if not data: break
        sys.stdout.write(data.decode())
        sys.stdout.flush()
except KeyboardInterrupt:
    pass
except Exception as e:
    print(f'Monitor error: {e}')
"

wait
