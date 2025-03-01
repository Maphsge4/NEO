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
        # Change the rate argument (in reqs/s) to other values to see more results
        # Feel free to comment out some of the following lines to reduce running time
        if server_name == "ours":
            await run_test(*prepare_real_test("osc", config, server_name), rate=0.5)
            await run_test(*prepare_real_test("osc", config, server_name), rate=1.5)
            await run_test(*prepare_real_test("osc", config, server_name), rate=2.5)
            await run_test(*prepare_real_test("osc", config, server_name), rate=3.1)
            await run_test(*prepare_real_test("osc", config, server_name), rate=3.5)
            await run_test(*prepare_real_test("osc", config, server_name), rate=3.7)
            await run_test(*prepare_real_test("osc", config, server_name), rate=3.9)
        if server_name == "vllm":
            await run_test(*prepare_real_test("osc", config, server_name), rate=0.2)
            await run_test(*prepare_real_test("osc", config, server_name), rate=0.4)
            await run_test(*prepare_real_test("osc", config, server_name), rate=0.5)
            await run_test(*prepare_real_test("osc", config, server_name), rate=0.6)
    finally:
        stop_server()
    await asyncio.sleep(5)


async def main():
    await one_round("vllm")
    await one_round("ours")


if __name__ == "__main__":
    asyncio.run(main())