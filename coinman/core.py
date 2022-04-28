import asyncio
from collections import defaultdict
from functools import partial
import os
from pathlib import Path
import aiohttp_rpc
from chia.consensus.default_constants import DEFAULT_CONSTANTS
from chia.types.coin_record import CoinRecord
from chia.types.coin_spend import CoinSpend

from chia.util.keychain import generate_mnemonic
from chia.util.streamable import Streamable
from coinman.simulator import NodeSimulator
from contextlib import asynccontextmanager
from inspect import FullArgSpec
import json
from clvm.SExp import SExp

from yaml.loader import SafeLoader
from aiohttp_rpc import JsonRpcServer, JsonRpcMethod
from aiohttp import web

from blspy import AugSchemeMPL, G1Element, G2Element, PrivateKey
from coinman.contract import Contract, load_contracts, transpile
from coinman.wallet import ContractWallet
from coinman.node import Node
from typing import Any, Dict, List, TextIO, Union
import yaml
import logging
import time
import logging.config
import sys

import os


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "verbose": {
                "format": "[{levelname}] [{asctime}] [{module}:{funcName}():{lineno}] - {message}",
                "style": "{",
            },
            "simple": {
                "format": "{levelname} {message}",
                "style": "{",
            },
        },
        "disable_existing_loggers": True,
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "verbose"},
        },
        "loggers": {
            "chia": {  # root logger
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False,
            },
            "aiohttp_rpc": {
                "handlers": ["console"],
                "level": os.environ.get("LOG_LEVEL", "DEBUG"),
                "propagate": False,
            },
            "aiohttp": {
                "handlers": ["console"],
                "level": os.environ.get("LOG_LEVEL", "DEBUG"),
                "propagate": False,
            },
            "coinman": {
                "handlers": ["console"],
                "level": os.environ.get("LOG_LEVEL", "DEBUG"),
                "propagate": False,
            },
        },
    }
)
log = logging.getLogger(__name__)

COINMAN = None


def parse_config(file_or_str: Union[str, TextIO]) -> Dict:
    config = {}
    config = yaml.load(file_or_str, Loader=SafeLoader)
    return config


def obj_from_hex(obj: Any):
    """Recursively convert ascii hex values to integers"""
    if isinstance(obj, dict):
        return {k: obj_from_hex(v) for k, v in obj.items()}
    elif all((isinstance(obj, str), obj.startswith("0x"))):
        return bytes.fromhex(obj[2:])
    elif isinstance(obj, list):
        return [obj_from_hex(l) for l in obj]
    return obj


class Coinman:

    node: Node
    wallets: Dict
    config: Dict

    def __init__(self, node, wallets, contracts, config, simulator=False):
        self.wallets = wallets
        self.node = node
        self.contracts = contracts
        self.config = config
        self.simulator = simulator

    @staticmethod
    def init(path_str: str, testnet=False):
        """
        chia_config: "/home/user/.chia/mainnet" # this is optional
        wallets:
          hot_wallet:
            mnemonic: super hot mnemonic # this won't work in non-simulator mode
        drivers:
          # list of directories with driver Chialisp modules in them
          - "./drivers"
        """
        initial_config = {
            "chia_config": "~/.chia/mainnet",
            "network": {"testnet": testnet},
            "wallets": {"main": {"mnemonic": generate_mnemonic()}},
        }
        if testnet:
            # add testnet10 agg sig me by default
            initial_config["network"][
                "agg_sig_me_additional_data"
            ] = "ae83525ba8d1dd3f09b277de18ca3e43fc0af20d20c4b3e92ef2a48bd291ccb2"
        path = Path(path_str) / Path("coinman.yaml")
        if os.path.exists(path):
            print("Project already initialized. Check coinman.yaml.")
            return False
        config_file = open(path, "w")
        yaml.dump(initial_config, config_file)
        return True

    @staticmethod
    async def create_instance(config_file_path: str, simulate=True):
        config_file = open(config_file_path)
        config = parse_config(config_file)
        wallets = {}
        chia_config = os.path.expanduser(config["chia_config"])
        if not simulate:
            node = await Node.create(chia_config)
        else:
            node = await NodeSimulator.create()
            config["network"] = {}
        for wallet_id, private_key in config["wallets"].items():
            wallets[wallet_id] = ContractWallet(
                node,
                private_key["mnemonic"],
                private_key.get("passphrase", ""),
                network=config.get("network"),
            )
        contracts = defaultdict(list)
        contracts_config = config.get("contracts", [])
        for path in contracts_config:
            contract_module = load_contracts(path)
            for contract_method in contract_module:
                contracts[contract_method.full_path].append(contract_method)

        instance = Coinman(node, wallets, contracts, config, simulate)
        global COINMAN
        COINMAN = instance
        return instance

    @staticmethod
    @asynccontextmanager
    async def create(config_file_path: str, simulate=True):
        global context
        instance = None
        try:
            instance = await Coinman.create_instance(config_file_path, simulate)
            yield instance
        except Exception as e:
            log.exception("Error creating Coinman object")
            raise
        finally:
            if instance:
                await instance.destroy()

    def create_wallet(
        self, wallet_id, simple_seed=None, mnemonic=None, passphrase=None
    ) -> ContractWallet:
        if not (mnemonic or simple_seed):
            simple_seed = wallet_id
        w = ContractWallet(
            self.node,
            simple_seed=simple_seed,
            mnemonic=mnemonic,
            passphrase=passphrase,
            network=self.config["network"],
        )
        self.wallets[wallet_id] = w
        return w

    def get_wallet(self, wallet_id=None) -> ContractWallet:
        if not wallet_id:
            return list(self.wallets.values())[0]
        return self.wallets[wallet_id]

    async def invoke(
        self,
        contract_filename: str,
        state: Dict,
        method: bytes,
        args: List,
        amount=1,
        fee=0,
        wallet_id=None,
    ):
        try:
            state: Dict = obj_from_hex(state)
            args = remove_str(args)
            log.debug(
                "Received invoke contract method request: %s"
                % str((state, method, args))
            )
            contract = Contract(contract_filename, state, amount=int(amount))
            w = self.get_wallet(wallet_id)
            result = await w.run(contract, method, *args, fee=fee)
            log.debug("Got back result: %r" % result)
            return result

        except Exception as e:
            log.exception(
                "error invoking contract method with: %s"
                % str((contract_filename, state, method, args, amount))
            )
            return dict(error=type(e).__name__)

    async def get_min_fee_per_cost(self):
        try:
            mempool_items = await self.node.get_all_mempool_items()

            if not mempool_items:
                return 0.0
            fees_and_cost = [
                (x["fee"], x["cost"], x["fee"] / x["cost"] if x["fee"] else 0)
                for x in mempool_items.values()
            ]
            max_cost = int(
                DEFAULT_CONSTANTS.MAX_BLOCK_COST_CLVM
                * DEFAULT_CONSTANTS.MEMPOOL_BLOCK_BUFFER
            )
            total_cost = sum([x[1] for x in fees_and_cost])

            if total_cost * 1.05 > max_cost:
                fees_and_cost = [
                    x for x in fees_and_cost if x[2] > 5
                ]  # 5 is the magic number for now
            fees_and_cost.sort(key=lambda x: x[0])
            min_fee = fees_and_cost[0][2]
            for fees in fees_and_cost:
                if fees[2] > min_fee:
                    return fees[2] * 1.05  # increase a bit to be safe
            return min_fee + 1
        except Exception as e:
            log.exception("Error getting min fee per cost")
            return dict(error=e)

    async def fee_for_invoke(
        self,
        contract_filename: str,
        state: Dict,
        fee_per_cost: int,
        method: bytes,
        *args,
        amount=1,
        wallet_id=None,
    ):
        try:
            state: Dict = obj_from_hex(state)
            args = remove_str(args)
            log.debug(
                "Received invoke contract method request: %s"
                % str((state, method, args))
            )
            w = self.get_wallet(wallet_id)
            contract = Contract(contract_filename, state, amount=int(amount))
            result = await w.calculate_fee(
                int(float(fee_per_cost)),
                contract,
                method,
                args,
            )
            log.debug("Got back fee result: %r" % result)
            return result

        except Exception as e:
            log.exception(
                "error calculating fee for  method with: %s"
                % str((contract_filename, state, method, args, amount))
            )
            return dict(error=str(e))

    async def get_coins(
        self, contract_filename: str, state: Dict, amount=1, wallet_id=None
    ):
        try:
            state = obj_from_hex(state)
            contract = Contract(contract_filename, state, amount=int(amount))
            result = await contract.get_coin_query(self.node)
            return result
        except Exception as e:
            log.exception(
                "error fetching contract coins with: %s"
                % str((contract_filename, state, amount))
            )
            return dict(error=str(e))

    async def mint(
        self, contract_filename: str, state: Dict, amount=1, fee=0, wallet_id=None
    ):
        try:
            state = obj_from_hex(state)
            w = self.get_wallet(wallet_id)
            contract = Contract(contract_filename, state, amount=int(amount))
            result = await w.mint(contract, amount, fee)
            return result
        except Exception as e:
            log.exception(
                "error minting contract coin with: %s"
                % str((contract_filename, state, amount))
            )
            return dict(error=str(e))

    async def inspect(self, contract_filename: str, state: Dict, amount=1) -> Dict:
        try:
            state = obj_from_hex(
                state,
            )
            log.debug(
                "Ok, inspecting a contract... %s"
                % str((contract_filename, state, amount))
            )
            contract = Contract(contract_filename, state, amount=amount)
            return dict(
                name=contract.full_path,
                methods=contract.methods,
                properties=contract.props,
                hints=contract.hints,
            )
        except Exception as e:
            log.exception(
                "error inspecting contract with: %s"
                % str((contract_filename, state, amount))
            )
            return dict(error=str(e))

    async def get_wallet_info(self, wallet_id=None) -> Dict:
        w = self.get_wallet(wallet_id)
        return dict(
            balance=await w.balance(),
            is_simulator=isinstance(self.node, NodeSimulator),
            address=w.address,
            puzzle_hash=w.puzzle_hash,
            public_key=w.pk(),
            blockchain_state=await self.node.get_blockchain_state(),
        )

    async def farm_block(self, wallet_id=None):
        if self.simulator:
            w = self.get_wallet(wallet_id)
            self.node.set_current_time()
            new_block = await self.node.farm_block(w.puzzle_hash)
            return new_block
        return "not in simulator mode, so didn't do anything"

    async def destroy(self):
        await self.node.close()

    async def keep_mempool_full(self):
        """Simulates full mempool, only for simulator"""
        w = ContractWallet(self.node, simple_seed="nobody")
        await self.node.farm_block(w.puzzle_hash)
        await self.node.farm_block(w.puzzle_hash)
        while True:
            try:
                await w.do_no_op_spend()
                await w.do_no_op_spend()
            except ValueError:
                log.exception("Error stuff")
                await asyncio.sleep(30)
            except:
                log.exception("serious error")
                await asyncio.sleep(30)
            await asyncio.sleep(3)

    async def create_rpc_app(self):
        def json_serialize_unknown_value(value):
            log.debug("Serializing: %s" % repr(value))
            try:
                if isinstance(value, bytes):
                    return "0x" + value.hex()
                elif isinstance(value, CoinRecord):
                    return dict(
                        {
                            (x, json_serialize_unknown_value(y))
                            for x, y in value.to_json_dict().items()
                        }
                    )
                elif isinstance(value, Streamable):
                    return value.to_json_dict()

                elif isinstance(value, G1Element):
                    return "0x" + str(value)
                return str(value)
            except:
                log.exception("Error serializing: %s" % repr(value))
                raise

        log.debug("Setting up RPC server")
        rpc_server = JsonRpcServer(
            middlewares=[
                aiohttp_rpc.middlewares.exception_middleware,
                aiohttp_rpc.middlewares.logging_middleware,
            ],
            json_serialize=partial(json.dumps, default=json_serialize_unknown_value),
        )

        async def root(request):
            raise web.HTTPFound("/dapp/index.html")

        # now node methods
        rpc_server.add_method(JsonRpcMethod(self.node.push, name="node.push"))
        rpc_server.add_method(JsonRpcMethod(self.invoke, name="contract.invoke"))
        rpc_server.add_method(JsonRpcMethod(self.mint, name="contract.mint"))
        rpc_server.add_method(JsonRpcMethod(self.inspect, name="contract.inspect"))
        rpc_server.add_method(JsonRpcMethod(self.get_wallet_info, name="wallet.poke"))
        rpc_server.add_method(
            JsonRpcMethod(self.get_min_fee_per_cost, name="node.get_min_fee_per_cost")
        )
        rpc_server.add_method(
            JsonRpcMethod(self.fee_for_invoke, name="contract.get_fee_for_invoke")
        )
        rpc_server.add_method(JsonRpcMethod(self.get_coins, name="contract.get_coins"))

        if isinstance(self.node, NodeSimulator):
            rpc_server.add_method(
                JsonRpcMethod(self.farm_block, name="node.farm_block")
            )
            asyncio.create_task(self.keep_mempool_full())
        app = web.Application()
        app.router.add_routes(
            [web.post("/rpc/", rpc_server.handle_http_request), web.get("/", root)]
        )
        app.add_routes([web.static("/dapp", "dapp")])

        return app


def remove_str(args) -> List:
    cleaned_args = []
    for arg in args:
        if isinstance(arg, str):
            arg = arg.encode("utf-8")
        cleaned_args.append(arg)
    return cleaned_args


def get_coinman() -> Coinman:
    global COINMAN
    return COINMAN
