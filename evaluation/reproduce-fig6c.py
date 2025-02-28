import asyncio
import json
import os

from server import start_server, stop_server
from benchmark import run_test, prepare_real_test


cur_dir = os.path.dirname(os.path.realpath(__file__))
with open(f"{cur_dir}/configs/config-t4-7b.json", "r") as f:
    config = json.load(f)


async def one_round(server_name: str):
    start_server(server_name, config)
    try:
        await run_test(*prepare_real_test("osc", config, server_name), 0.5)
    finally:
        stop_server()
    await asyncio.sleep(5)


async def main():
    await one_round("vllm")
    await one_round("ours")


if __name__ == "__main__":
    asyncio.run(main())