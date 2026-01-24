#!/usr/bin/env python3
"""
Test manager client for AirVLN simulator server.

This script demonstrates how to:
- connect to the management server (msgpackrpc)
- call `ping`
- call `reopen_scenes` to request an AirSim instance be started
- connect to the returned AirSim instance (using `airsim.MultirotorClient`) and call a simple API

Usage example:
    python airsim_plugin/scripts/test_manager_client.py --host 127.0.0.1 --port 30000 --scene ModularPark --gpu 0

Notes:
- Requires `msgpackrpc` (msgpack-rpc-python) to talk to the manager server.
- To exercise AirSim API calls this machine must have `airsim` Python package installed
  and the manager server must successfully start an AirSim instance and return its port.
"""

import argparse
import time
import sys


def create_msgpack_client(host: str, port: int, timeout: int = 30):
    try:
        import msgpackrpc
    except Exception as e:
        print("ERROR: cannot import msgpackrpc:", e)
        return None, None

    addr = msgpackrpc.Address(host, port)
    client = msgpackrpc.Client(addr, timeout=timeout)
    return client, msgpackrpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="manager host")
    parser.add_argument("--port", type=int, default=30000, help="manager port")
    parser.add_argument("--scene", type=str, default="ModularPark", help="scene name to open")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id for the scene")
    parser.add_argument("--wait", type=int, default=60, help="seconds to wait for AirSim confirmConnection")
    args = parser.parse_args()

    client, msgpackrpc = create_msgpack_client(args.host, args.port)
    if client is None:
        print("Please install msgpack-rpc-python (package name may be 'msgpack-rpc-python' or 'msgpackrpc').")
        sys.exit(2)

    # ping
    try:
        res = client.call("ping")
        print("ping ->", res)
    except Exception as e:
        print("ping failed:", e)
        sys.exit(3)

    # request reopen_scenes -> returns (ok, (ip, [ports...]))
    try:
        print(f"calling reopen_scenes on {args.host}:{args.port} for scene {args.scene} gpu {args.gpu}")
        res = client.call("reopen_scenes", args.host, [(args.scene, args.gpu)])
        print("reopen_scenes ->", res)
    except Exception as e:
        print("reopen_scenes failed:", e)
        sys.exit(4)

    try:
        ok, data = res
    except Exception:
        print("unexpected reopen_scenes return format")
        sys.exit(5)

    if not ok:
        print("server returned failure for reopen_scenes")
        sys.exit(6)

    ip, ports = data
    if not ports:
        print("no ports returned by server")
        sys.exit(7)

    first_port = ports[0]
    print(f"Attempting to connect to AirSim at {ip}:{first_port}")

    # try to import airsim and connect
    try:
        import airsim
    except Exception as e:
        print("airsim package not available:", e)
        print("If you only want to test the manager server, the 'ping' and 'reopen_scenes' calls are enough.")
        sys.exit(0)

    mc = airsim.MultirotorClient(ip=ip, port=first_port, timeout_value=60)

    # confirmConnection with retries
    connected = False
    start = time.time()
    while time.time() - start < args.wait:
        try:
            mc.confirmConnection()
            connected = True
            print("MultirotorClient confirmConnection succeeded")
            break
        except Exception as e:
            print("confirmConnection attempt failed:", e)
            time.sleep(1)

    if not connected:
        print("Could not connect to AirSim instance within wait time")
        sys.exit(8)

    # quick API probe
    try:
        ver = mc.getServerVersion()
        print("AirSim server version ->", ver)
    except Exception as e:
        print("getServerVersion failed:", e)

    try:
        state = mc.getMultirotorState()
        print("getMultirotorState ok, vehicle_name:", getattr(state, 'vehicle_name', None))
    except Exception as e:
        print("getMultirotorState failed:", e)

    try:
        mc.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
