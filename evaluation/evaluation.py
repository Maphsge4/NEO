import asyncio

from server import start_server, stop_server
from benchmark import run_test, prepare_real_test


async def one_round(name: str):
    global server_name
    server_name = name
    start_server(server_name)
    if server_name == "ours":
        # await run_test(*prepare_real_test("osc"), 0.5)
        await run_test(*prepare_real_test("osc"), 1.0)
    stop_server()
    await asyncio.sleep(10)


async def main():
    await one_round("ours")
    # await one_round("vllm")


if __name__ == "__main__":
    asyncio.run(main())