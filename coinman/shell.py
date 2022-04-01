#!/usr/bin/env python3
from typing import List
import aiohttp_rpc

from chia.types.spend_bundle import SpendBundle
from .simulator import NodeSimulator

from coinman.util import syncify


def load_ipython_extension(ip):

    import asyncio
    from coinman.wallet import Wallet

    loop = asyncio.get_event_loop()
    run = loop.run_until_complete
    node = run(NodeSimulator.create())

    def farm_block(wallet: Wallet = None):
        additions, removals = run(node.farm_block(wallet))
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
        wallet = Wallet(node, simple_seed=seed)
        wallet.mint = syncify(wallet.mint)
        wallet.spend = syncify(wallet.spend)
        wallet.balance = syncify(wallet.balance)
        return wallet

    def push(*spends: List[SpendBundle]):
        """Push spends to blockchain"""
        return run(node.push(*spends))

    class Client:
        @staticmethod
        def rpc(method, *args):
            async def _run():
                async with aiohttp_rpc.JsonRpcClient("http://0.0.0.0:9000/rpc/v1") as c:
                    return await c.call(method, *args)

            return asyncio.run(_run())

        @staticmethod
        def methods():
            async def _run():
                async with aiohttp_rpc.JsonRpcClient("http://0.0.0.0:9000/rpc/v1") as c:
                    return await c.get_methods()

            return asyncio.run(_run())

    def chia_help():
        print(
            """
- client -> use this to run driver methods on the server
- new_wallet(wallet_id) -> creates a new wallet with wallet_id as seed
- push(spend_bundle) -> pushes bundle to mempool
- farm_block([wallet_id]) -> farms a new block if running in simulator mode
                             if wallet_id is provided, it assigns rewards to that wallet
- node -> allows access to simulator or real node
        """
        )

    ip.push(
        {
            "client": Client(),
            "chia": chia_help,
            "new_wallet": new_wallet,
            "push": push,
            "farm_block": farm_block,
            "node": node,
        }
    )
