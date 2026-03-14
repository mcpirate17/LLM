#!/usr/bin/env python3
"""
Hive Bus Client: Easy interface for agents to connect to the Hive Signal Bus.
"""

import socket
import json
import threading
import os
import time

class HiveBusClient:
    def __init__(self, socket_path="/tmp/llm_hive.sock", name="agent"):
        self.socket_path = socket_path
        self.name = name
        self.sock = None
        self.listener_thread = None
        self.on_message_cb = None

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.socket_path)
            return True
        except socket.error:
            return False

    def send(self, msg_type, content):
        if not self.sock:
            return
        
        payload = {
            "source": self.name,
            "type": msg_type,
            "content": content,
            "timestamp": time.time()
        }
        try:
            self.sock.sendall(json.dumps(payload).encode() + b"\n")
        except socket.error:
            self.sock = None

    def _listen(self):
        buffer = b""
        while self.sock:
            try:
                data = self.sock.recv(4096)
                if not data: break
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if self.on_message_cb:
                        try:
                            msg = json.loads(line)
                            self.on_message_cb(msg)
                        except:
                            pass
            except:
                break

    def start_listening(self, callback):
        self.on_message_cb = callback
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()

if __name__ == "__main__":
    # Test Client
    client = HiveBusClient(name="test_agent")
    if client.connect():
        print("Connected! Type a message and hit enter.")
        client.start_listening(lambda m: print(f"\n[FROM BUS] {m}"))
        while True:
            text = input("> ")
            client.send("chat", text)
    else:
        print("Could not connect to bus. Start hive.sh first.")
