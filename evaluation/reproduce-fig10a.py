import asyncio
import json
import os

from server import start_server, stop_server
from benchmark import run_test, prepare_mock_test


cur_dir = os.path.dirname(os.path.realpath(__file__))
with open(f"{cur_dir}/configs/config-a10-8b.json", "r") as f:
    config = json.load(f)


async def one_round(server_name: str):
    start_server(server_name, config)
    try:
        await run_test(*prepare_mock_test(1000, 1000, 50, server_name, config))
        await run_test(*prepare_mock_test(1000, 1000, 100, server_name, config))
        await run_test(*prepare_mock_test(1000, 1000, 200, server_name, config))
        await run_test(*prepare_mock_test(1000, 1000, 300, server_name, config))
        await run_test(*prepare_mock_test(1000, 1000, 400, server_name, config))
    finally:
        stop_server()
    await asyncio.sleep(5)


async def main():
    await one_round("base")
    await one_round("ours")


if __name__ == "__main__":
    asyncio.run(main())
