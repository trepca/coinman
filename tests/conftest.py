from coinman.simulator import NodeSimulator
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def node():
    return await NodeSimulator.create()
