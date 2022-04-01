from typing import List
from chia.types.coin_record import CoinRecord
from chia.util.ints import uint64
from coinman.core import get_coinman


async def create_spend_bundle(wallet_id, coin_ids, puzzle_hash, solution, fee):
    coinman = get_coinman()
    wallet = coinman.get_wallet(wallet_id)
    return []


async def select_coins(wallet_id, mojos):
    return []


async def balance(wallet_id) -> uint64:
    """Return the actor's balance in standard coins as we understand it"""
    coinman = get_coinman()
    wallet = coinman.get_wallet(wallet_id)

    spendable_coins: List[
        CoinRecord
    ] = await coinman.node.get_coin_records_by_puzzle_hash(
        wallet.puzzle_hash, include_spent_coins=False
    )
    return uint64(sum(map(lambda x: x.coin.amount, spendable_coins)))
