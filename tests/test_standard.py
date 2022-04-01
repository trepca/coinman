from chia.types.blockchain_format.coin import Coin
from chia.types.spend_bundle import SpendBundle
from chia.util.ints import uint64
from coinman.wallet import Wallet
from coinman.simulator import NodeSimulator
import pytest


@pytest.mark.asyncio
async def test_wallet_balance(node: NodeSimulator):
    w = Wallet(node)
    assert await w.balance() == 0


@pytest.mark.asyncio
async def test_wallet_transfer(node: NodeSimulator):
    w1 = Wallet(node)
    w2 = Wallet(node)
    await node.farm_block(w1)
    w1_balance = await w1.balance()
    assert w1_balance > 0
    coin: Coin = await w1.send_mojos(w2, uint64(100))
    assert coin.puzzle_hash == w2.puzzle_hash
    await node.farm_block()
    assert await w1.balance() == (w1_balance - 100)
    assert await w2.balance() == 100
