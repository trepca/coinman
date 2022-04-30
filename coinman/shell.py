#!/usr/bin/env python3
import asyncio
from contextlib import asynccontextmanager
import sys
import click
from coinman.contract import Contract
from typing import Dict, List
import aiohttp_rpc

from chia.types.spend_bundle import SpendBundle
from .simulator import NodeSimulator

from coinman.util import syncify
from contextlib import contextmanager


def load_ipython_extension(ip):

    import asyncio

    from coinman.wallet import ContractWallet

    loop = asyncio.get_event_loop()
    run = loop.run_until_complete
    node = run(NodeSimulator.create())

    async def farm_block(wallet: ContractWallet = None):
        additions, removals = await node.farm_block(
            wallet.puzzle_hash if wallet else None
        )

        print("=> Farmed a new block: %s" % node.sim.block_height)
        if wallet:
            print("=> Rewarded `%s` with block reward coins." % wallet)
        if additions:
            print("=> Added coins:")
            for coin in additions:
                print(
                    f"\t - Coin name={coin.name()}\n\t\tparent={coin.parent_coin_info} \n\t\tpuzzle={coin.puzzle_hash} \n\t\tamount={coin.amount}\n"
                )
        else:
            print("=> No coins added.")
        if removals:
            print("=> Removed coins:")
            for coin in removals:
                print(
                    f"\t - Coin name={coin.name()}\n\t\tparent={coin.parent_coin_info} \n\t\tpuzzle={coin.puzzle_hash} \n\t\tamount={coin.amount}\n"
                )
        else:
            print("=> No coins removed.")

    def new_wallet(seed):
        wallet = ContractWallet(node, simple_seed=seed)
        wallet.mint = wallet.mint_chialisp
        return wallet

    async def push(*spends: List[SpendBundle]):
        """Push spends to the blockchain"""
        return await node.push(*spends)

    class Client:
        @staticmethod
        def rpc(method, *args):
            async def _run():
                async with aiohttp_rpc.JsonRpcClient("http://0.0.0.0:9000/rpc/") as c:
                    return await c.call(method, *args)

            return asyncio.run(_run())

        @staticmethod
        def methods():
            async def _run():
                async with aiohttp_rpc.JsonRpcClient("http://0.0.0.0:9000/rpc/") as c:
                    return await c.get_methods()

            return asyncio.run(_run())

    def chia_help():
        client_text = click.style("client", bold=True)
        print(
            f"""
- {client_text} -> use this to run driver methods on the RPC server (you'll need to run `coinman runserver`)
- {click.style('new_wallet(wallet_id)', bold=True)} -> creates a new wallet with wallet_id as seed
- {click.style('push(spend_bundle)', bold=True)} -> pushes bundle to mempool
- {click.style('load_contract(contract_puzzle_filename, state_dict, amount)', bold=True)} -> returns a Contract object
- {click.style('farm_block([wallet_id])', bold=True)} -> farms a new block if running in simulator mode
                             if wallet_id is provided, it assigns rewards to that wallet
- {click.style('node', bold=True)} -> access simulated node RPC API

{click.style('Wallet functions:', bold=True)}
  - {click.style('mint(chialisp, amount, fee)', bold=True)} -> mints a coin with `chialisp` program and amount
      For example, to send 10 mojos to `recipient_puzhash`
            alice.mint('mod (recipient_puzhash) (list (list 51 recipient_puzhash 1))))', amount=10)

  - {click.style('balance', bold=True)} -> shows the balance in the wallet
  - {click.style('run(contract, contract_method, contract_method_args, fee)', bold=True)} -> runs a contract method
  - {click.style('select_coins(mojos)', bold=True)} -> returns coins that would have to be spend to create a new coin with amount `mojos`
  -

"""
        )

    def load_contract(contract_filename: str, state: Dict, amount: int):
        contract = Contract(contract_filename, state, amount)
        return contract

    chia_help()
    ip.push(
        {
            "client": Client(),
            "chia": chia_help,
            "new_wallet": new_wallet,
            "push": push,
            "farm_block": farm_block,
            "node": node,
            "load_contract": load_contract,
        }
    )
