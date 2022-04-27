import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from chia.consensus.default_constants import DEFAULT_CONSTANTS
from chia.rpc.full_node_rpc_client import FullNodeRpcClient
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.program import INFINITE_COST, Program
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.coin_record import CoinRecord
from chia.types.coin_spend import CoinSpend
from chia.types.condition_opcodes import ConditionOpcode
from chia.types.spend_bundle import SpendBundle
from chia.util.condition_tools import (
    ConditionOpcode,
    conditions_dict_for_solution,
    pkm_pairs_for_conditions_dict,
)
from chia.util.config import load_config
from chia.util.default_root import DEFAULT_ROOT_PATH
from chia.util.hash import std_hash
from chia.util.ints import uint16, uint32

from blspy import AugSchemeMPL, G1Element

log = logging.getLogger(__name__)


async def get_node_client(config_path=DEFAULT_ROOT_PATH) -> FullNodeRpcClient:
    try:
        if not config_path:
            config_path = DEFAULT_ROOT_PATH
        else:
            config_path = Path(config_path)
        config = load_config(config_path, "config.yaml")
        self_hostname = config["self_hostname"]
        full_node_rpc_port = config["full_node"]["rpc_port"]
        full_node_client: FullNodeRpcClient = await FullNodeRpcClient.create(
            self_hostname, uint16(full_node_rpc_port), DEFAULT_ROOT_PATH, config
        )
        return full_node_client
    except Exception as e:
        raise


class Node:

    client: FullNodeRpcClient

    simulator: bool

    @classmethod
    async def create(cls, config_path) -> "Node":
        self = cls()
        self.client = await get_node_client(config_path)
        return self

    async def close(self):
        self.client.close()
        await self.client.await_closed()

    async def push(self, *bundles) -> Dict[str, Union[str, List[Coin]]]:
        bundle = SpendBundle.aggregate([x for x in bundles])
        pushed: Dict[str, Union[str, List[Coin]]] = await self.push_tx(bundle)
        return pushed

    # Have the system farm one block with a specific beneficiary (nobody if not specified).
    async def farm_block(self, farmer=None) -> Tuple[List[Coin], List[Coin]]:
        """Given a farmer, farm a block with that actor as the beneficiary of the farm
        reward.

        Used for causing chia balance to exist so the system can do things.
        """
        raise NotImplementedError()

    async def get_coin_records_by_puzzle_hash(
        self, puzzle_hash: bytes32, **kwargs
    ) -> List[Coin]:
        coins = await self.client.get_coin_records_by_puzzle_hash(puzzle_hash, **kwargs)
        if coins:
            return coins
        return []

    async def get_coin_records_by_hint(self, hint: bytes32, **kwargs) -> List[Coin]:
        coins = await self.client.get_coin_records_by_hint(hint, **kwargs)
        if coins:
            return coins
        return []

    # 'peak' is valid
    async def get_blockchain_state(self) -> Dict:
        return await self.client.get_blockchain_state()

    async def get_block_record_by_height(self, height):
        return await self.client.get_block_record_by_height(height)

    async def get_additions_and_removals(self, header_hash):
        return await self.client.get_additions_and_removals(header_hash)

    async def get_coin_record_by_name(self, name: bytes32) -> Optional[CoinRecord]:
        return await self.client.get_coin_record_by_name(name)

    async def get_coin_records_by_names(
        self,
        names: List[bytes32],
        include_spent_coins: bool = True,
        start_height: Optional[int] = None,
        end_height: Optional[int] = None,
    ) -> List:
        result_list = []

        for n in names:
            single_result = await self.client.get_coin_record_by_name(n)
            if single_result is not None:
                result_list.append(single_result)

        return result_list

    async def get_coin_records_by_parent_ids(
        self,
        parent_ids: List[bytes32],
        include_spent_coins: bool = True,
        start_height: Optional[int] = None,
        end_height: Optional[int] = None,
    ) -> List:
        result = []

        peak_data = await self.get_blockchain_state()
        last_block = await self.client.get_block_record_by_height(peak_data["peak"])

        while last_block.height > 0:
            last_block_hash = last_block.header_hash
            additions, _ = await self.client.get_additions_and_removals(last_block_hash)
            for new_coin in additions:
                if new_coin.coin.parent_coin_info in parent_ids:
                    result.append(new_coin)

            last_block = await self.client.get_block_record_by_height(
                last_block.height - 1
            )

        return result

    async def get_all_mempool_items(self) -> Dict[bytes32, Dict]:
        spends = await self.client.get_all_mempool_items()
        return spends

    async def get_puzzle_and_solution(
        self, coin_id: bytes32, height: uint32
    ) -> Optional[CoinSpend]:
        return await self.client.get_puzzle_and_solution(coin_id, height)

    # Given a spend bundle, farm a block and analyze the result.
    async def push_tx(self, bundle: SpendBundle) -> Dict:
        """Given a spend bundle, try to farm a block containing it.  If the spend bundle
        didn't validate, then a result containing an 'error' key is returned.  The reward
        for the block goes to Network::nobody"""

        status = await self.client.push_tx(bundle)
        return status
