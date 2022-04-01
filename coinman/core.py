from chia.util.ints import uint64
from coinman.simulator import NodeSimulator
from contextlib import asynccontextmanager
from inspect import FullArgSpec
import json
from click import disable_unicode_literals_warning
from clvm.SExp import SExp

from yaml.loader import SafeLoader
from aiohttp_rpc import JsonRpcServer, JsonRpcMethod
from aiohttp import web
from chia.types.blockchain_format.program import Program

from coinman.driver import Driver, load_drivers, transpile
from coinman.wallet import Wallet, disassemble
from coinman.node import Node
from typing import Dict, TextIO, Union
import yaml
from contextvars import ContextVar
import logging

log = logging.getLogger(__name__)

context = ContextVar("coinman")


def parse_config(file_or_str: Union[str, TextIO]) -> Dict:
    config = {}
    config = yaml.load(file_or_str, Loader=SafeLoader)
    return config


class Coinman:

    node: Node
    wallets: Dict
    config: Dict

    def __init__(self, node, wallets, config):
        self.wallets = wallets
        self.node = node
        self.config = config

    @staticmethod
    @asynccontextmanager
    async def create(config_file_path: str, simulate=True):
        instance = None
        try:
            config_file = open(config_file_path)
            config = parse_config(config_file)
            wallets = {}
            if not simulate:
                node = await Node.create(config["chia_config"])
            else:
                node = await NodeSimulator.create()
            for wallet_id, private_key in config["wallets"].items():
                wallets[wallet_id] = Wallet(
                    node, private_key["mnemonic"], private_key.get("passphrase", "")
                )
            instance = Coinman(node, wallets, config)
            context.set(instance)
            yield instance
        except Exception as e:
            log.exception("Error creating Coinman object")
            raise
        finally:
            if instance:
                await instance.destroy()

    def get_wallet(self, wallet_id) -> Wallet:
        return self.wallets[wallet_id]

    async def destroy(self):
        await self.node.close()

    async def create_rpc_app(self):
        rpc_server = JsonRpcServer()

        # first add drivers
        for path in self.config["drivers"]:
            drivers = load_drivers(path)
            for driver in drivers:
                method = _make_rpc_method(driver)
                rpc_server.add_method(method)

        # now node methods
        rpc_server.add_method(JsonRpcMethod(self.node.push, name="node.push"))
        if isinstance(self.node, NodeSimulator):
            rpc_server.add_method(
                JsonRpcMethod(self.node.farm_block, name="node.farm_block")
            )
        app = web.Application()
        app.router.add_routes(
            [
                web.post("/rpc/v1", rpc_server.handle_http_request),
            ]
        )
        return app


def get_coinman() -> Coinman:
    return context.get()


def _make_rpc_method(driver: Driver) -> JsonRpcMethod:
    async def driver_func_factory(*args):
        try:
            log.debug("Running driver method: %s with args: %s", driver.name, args)
            result: SExp = driver.run(args)
            data: bytes = result.as_python()
            log.debug("Got back data: %s", data)
            raw_data = await transpile(data)
            data = []
            for chunk in raw_data:
                if isinstance(chunk, bytes):
                    data.append(chunk.decode("utf-8"))
                elif isinstance(chunk, int):
                    data.append(str(chunk))
            data = "".join([x for x in data])
            output: str = data.replace("'", '"')
            log.debug("Got back output: %s", output)
            return json.loads(output)
        except Exception as e:
            log.exception("Error running driver")
            return []

    method = JsonRpcMethod(driver_func_factory, name=f"{driver.module}.{driver.name}")
    method.supported_args = FullArgSpec(driver.args, None, None, None, [], None, {})
    return method
