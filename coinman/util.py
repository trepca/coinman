import asyncio


def syncify(f):
    def decorated(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no event loop running:
            loop = asyncio.get_event_loop()
        return loop.run_until_complete(f(*args, **kwargs))

    return decorated
