#!/usr/bin/env python3
"""
Signal Broker: A high-performance Unix Domain Socket server for LLM agents.
Handles message routing between multiple CLI agents (Gemini, Claude, etc.).
"""

import socket
import os
import threading
import logging
from typing import List

SOCKET_PATH = "/tmp/llm_hive.sock"
logging.basicConfig(level=logging.INFO, format="[HIVE-BROKER] %(message)s")
logger = logging.getLogger("broker")


class HiveBroker:
    def __init__(self):
        self.clients: List[socket.socket] = []
        self.lock = threading.Lock()

    def broadcast(self, message: bytes, sender: socket.socket):
        with self.lock:
            disconnected = []
            for client in self.clients:
                if client != sender:
                    try:
                        client.sendall(message + b"\n")
                    except (socket.error, BrokenPipeError):
                        disconnected.append(client)

            for client in disconnected:
                if client in self.clients:
                    self.clients.remove(client)

    def handle_client(self, conn: socket.socket):
        with conn:
            with self.lock:
                self.clients.append(conn)

            logger.info(f"Agent connected. Total: {len(self.clients)}")

            buffer = b""
            while True:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break

                    buffer += data
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            self.broadcast(line, conn)
                except Exception as e:
                    logger.error(f"Client error: {e}")
                    break

        with self.lock:
            if conn in self.clients:
                self.clients.remove(conn)
        logger.info(f"Agent disconnected. Total: {len(self.clients)}")

    def start(self):
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o660)
        server.listen(10)

        logger.info(f"Signal Bus listening on {SOCKET_PATH}")

        try:
            while True:
                conn, _ = server.accept()
                threading.Thread(
                    target=self.handle_client, args=(conn,), daemon=True
                ).start()
        except KeyboardInterrupt:
            logger.info("Broker shutting down.")
        finally:
            server.close()
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)


if __name__ == "__main__":
    HiveBroker().start()
