"""A loopback listener whose accept queue is full, so new connects hang.

Usage: python full_backlog.py  -> prints the port, then holds it until stdin
closes. Linux drops a SYN while the accept queue is full, so connect() to the
port times out: the connect-timeout scenarios of the recovery differential.
"""
from __future__ import annotations

import socket
import sys
import time

server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(0)
port = server.getsockname()[1]
fillers = []
for _ in range(4):
    filler = socket.socket()
    filler.setblocking(False)
    try:
        filler.connect(("127.0.0.1", port))
    except BlockingIOError:
        pass
    fillers.append(filler)
time.sleep(0.3)
print(port, flush=True)
sys.stdin.read()
