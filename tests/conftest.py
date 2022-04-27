from coinman.simulator import NodeSimulator
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def node():
    return await NodeSimulator.create()


# @pytest_asyncio.fixture(autouse=True)
# async def coinman():
#     from coinman.core import Coinman

#     return await Coinman.create_instance("tests/config.yaml")
