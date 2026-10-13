#!/usr/bin/env python3
"""Block until a TCP port accepts a connection, or give up loudly.

`sleep N` in CI is a bet that the machine is as fast today as it was when you
picked N. Polling is the same three lines and cannot be wrong.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time


def wait(host: str, port: int, timeout: float, interval: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(interval)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if wait(args.host, args.port, args.timeout):
        print(f"{args.host}:{args.port} is up")
        return 0
    print(
        f"{args.host}:{args.port} did not accept a connection within "
        f"{args.timeout:g}s",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
