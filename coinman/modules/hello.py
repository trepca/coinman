import asyncio


async def hi(data):
    await asyncio.sleep(0.01)
    return ["hello world " + str(len(data))]
