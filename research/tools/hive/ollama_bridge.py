#!/usr/bin/env python3
"""
Ollama Bridge: Connects Ollama to the Hive Signal Bus.
Broadcasts Ollama's status and captures interaction signals.
"""

import socket
import json
import time
import requests
import logging

SOCKET_PATH = "/tmp/llm_hive.sock"
OLLAMA_URL = "http://localhost:11434/api/tags"

logging.basicConfig(level=logging.INFO, format="[HIVE-OLLAMA] %(message)s")
logger = logging.getLogger("ollama_bridge")


def connect_to_bus():
    while True:
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(SOCKET_PATH)
            return client
        except socket.error:
            logger.info("Waiting for Signal Broker...")
            time.sleep(2)


def main():
    bus = connect_to_bus()
    logger.info("Connected to Hive Signal Bus.")

    last_status = None

    while True:
        try:
            # Check Ollama status
            try:
                response = requests.get(OLLAMA_URL, timeout=2)
                status = "online" if response.status_code == 200 else "error"
                models = [m["name"] for m in response.json().get("models", [])]
            except requests.exceptions.RequestException:
                status = "offline"
                models = []

            # Send heartbeats / status changes
            msg = {
                "source": "ollama_bridge",
                "type": "status",
                "status": status,
                "models": models,
                "timestamp": time.time(),
            }

            if status != last_status:
                logger.info(f"Ollama is {status}. Models: {models}")
                bus.sendall(json.dumps(msg).encode() + b"\n")
                last_status = status

            # Idle wait
            time.sleep(5)

        except (socket.error, BrokenPipeError):
            logger.warning("Lost connection to Bus. Reconnecting...")
            bus = connect_to_bus()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
