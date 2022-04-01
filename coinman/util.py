import asyncio


def syncify(f):
    loop = asyncio.get_event_loop()

    def decorated(*args, **kwargs):
        return loop.run_until_complete(f(*args, **kwargs))

    return decorated
