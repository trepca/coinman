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
    keys = Dict

    @classmethod
    async def create(cls, config_path) -> "Node":
        self = cls()
        self.client = await get_node_client(config_path)
        self.keys = {}
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

    async def get_puzzle_and_solution(
        self, coin_id: bytes32, height: uint32
    ) -> Optional[CoinSpend]:
        return await self.client.get_puzzle_and_solution(coin_id, height)

    # Given a spend bundle, farm a block and analyze the result.
    async def push_tx(self, bundle: SpendBundle) -> Dict:
        """Given a spend bundle, try to farm a block containing it.  If the spend bundle
        didn't validate, then a result containing an 'error' key is returned.  The reward
        for the block goes to Network::nobody"""

        status, error = await self.client.push_tx(bundle)
        if error:
            return {"error": str(error), "status": status}
        return {"status": status}

    def push_debug(
        self,
        spend_bundle,
        agg_sig_additional_data=DEFAULT_CONSTANTS.AGG_SIG_ME_ADDITIONAL_DATA,
    ):
        """
        Print a lot of useful information about a `SpendBundle` that might help with debugging
        its clvm.
        """

        pks = []
        msgs = []

        created_coin_announcements: List[List[bytes]] = []
        asserted_coin_announcements = []
        created_puzzle_announcements: List[List[bytes]] = []
        asserted_puzzle_announcements = []

        for coin_spend in spend_bundle.coin_spends:
            coin = coin_spend.coin
            puzzle_reveal = Program.from_bytes(bytes(coin_spend.puzzle_reveal))
            solution = Program.from_bytes(bytes(coin_spend.solution))
            coin_name = coin.name()

            log.debug("Running coin %s: %s", coin.name(), coin)
            error, conditions, cost = conditions_dict_for_solution(
                puzzle_reveal, solution, INFINITE_COST
            )
            log.debug("Got back cost: %s", cost)
            if error:
                log.debug("Error running coin %s: %s", coin.name(), error)
                raise ValueError(error)
            elif conditions is not None:
                for pk_bytes, m in pkm_pairs_for_conditions_dict(
                    conditions, coin_name, agg_sig_additional_data
                ):
                    pks.append(G1Element.from_bytes(pk_bytes))
                    msgs.append(m)
                cost, r = puzzle_reveal.run_with_cost(INFINITE_COST, solution)  # type: ignore
                if conditions and len(conditions) > 0:
                    log.debug("grouped conditions:")
                    as_prog = None
                    for condition_programs in conditions.values():
                        for c in condition_programs:
                            if len(c.vars) == 1:
                                as_prog = Program.to([c.opcode, c.vars[0]])
                            elif len(c.vars) == 2:
                                as_prog = Program.to([c.opcode, c.vars[0], c.vars[1]])
                            else:
                                continue
                            log.debug("%s", disassemble(as_prog))
                    created_coin_announcements.extend(
                        [coin_name] + _.vars
                        for _ in conditions.get(
                            ConditionOpcode.CREATE_COIN_ANNOUNCEMENT, []
                        )
                    )
                    asserted_coin_announcements.extend(
                        [
                            _.vars[0].hex()
                            for _ in conditions.get(
                                ConditionOpcode.ASSERT_COIN_ANNOUNCEMENT, []
                            )
                        ]
                    )
                    created_puzzle_announcements.extend(
                        [puzzle_reveal.get_tree_hash()] + _.vars
                        for _ in conditions.get(
                            ConditionOpcode.CREATE_PUZZLE_ANNOUNCEMENT, []
                        )
                    )
                    asserted_puzzle_announcements.extend(
                        [
                            _.vars[0].hex()
                            for _ in conditions.get(
                                ConditionOpcode.ASSERT_PUZZLE_ANNOUNCEMENT, []
                            )
                        ]
                    )
                else:
                    log.debug("No conditions for coin: %s", coin.name())
        created = set(spend_bundle.additions())
        spent = set(spend_bundle.removals())

        zero_coin_set = set(coin.name() for coin in created if coin.amount == 0)

        ephemeral = created.intersection(spent)
        created.difference_update(ephemeral)
        spent.difference_update(ephemeral)
        log.debug("Spent coins:")
        for coin in sorted(spent, key=lambda _: _.name()):
            log.debug("%s: %s", coin.name(), dump_coin(coin))
        log.debug("created coins")
        for coin in sorted(created, key=lambda _: _.name()):
            log.debug("%s: %s", coin.name(), dump_coin(coin))

        if ephemeral:
            log.debug("ephemeral coins")
            for coin in sorted(ephemeral, key=lambda _: _.name()):
                log.debug("%s: %s", coin.name(), dump_coin(coin))

        created_coin_announcement_pairs = [
            (_, std_hash(b"".join(_)).hex()) for _ in created_coin_announcements
        ]
        if created_coin_announcement_pairs:
            log.debug("created coin announcements")
            for announcement, hashed in sorted(
                created_coin_announcement_pairs, key=lambda _: _[-1]
            ):
                as_hex = [f"0x{_.hex()}" for _ in announcement]
                log.debug(f"  {as_hex} =>\n      {hashed}")

        eor_coin_announcements = sorted(
            set(_[-1] for _ in created_coin_announcement_pairs)
            ^ set(asserted_coin_announcements)
        )

        created_puzzle_announcement_pairs = [
            (_, std_hash(b"".join(_)).hex()) for _ in created_puzzle_announcements
        ]
        if created_puzzle_announcements:
            log.debug("created puzzle announcements")
            for announcement, hashed in sorted(
                created_puzzle_announcement_pairs, key=lambda _: _[-1]
            ):
                as_hex = [f"0x{_.hex()}" for _ in announcement]
                log.debug(f"  {as_hex} =>\n      {hashed}")

        eor_puzzle_announcements = sorted(
            set(_[-1] for _ in created_puzzle_announcement_pairs)
            ^ set(asserted_puzzle_announcements)
        )

        log.debug(f"zero_coin_set = {sorted(zero_coin_set)}")
        if created_coin_announcement_pairs or asserted_coin_announcements:
            log.debug(
                f"created  coin announcements = {sorted([_[-1] for _ in created_coin_announcement_pairs])}"
            )
            log.debug(
                f"asserted coin announcements = {sorted(asserted_coin_announcements)}"
            )
            log.debug(
                f"symdiff of coin announcements = {sorted(eor_coin_announcements)}"
            )
        if created_puzzle_announcement_pairs or asserted_puzzle_announcements:
            log.debug(
                f"created  puzzle announcements = {sorted([_[-1] for _ in created_puzzle_announcement_pairs])}"
            )
            log.debug(
                f"asserted puzzle announcements = {sorted(asserted_puzzle_announcements)}"
            )
            log.debug(
                f"symdiff of puzzle announcements = {sorted(eor_puzzle_announcements)}"
            )
        log.debug("=" * 80)
        validates = AugSchemeMPL.aggregate_verify(
            pks, msgs, spend_bundle.aggregated_signature
        )

        log.debug(f"aggregated signature check pass: {validates}")
        log.debug(f"pks: {pks}")
        log.debug(f"msgs: {[msg.hex() for msg in msgs]}")
        log.debug(f"  msg_data: {[msg.hex()[:-128] for msg in msgs]}")
        log.debug(f"  coin_ids: {[msg.hex()[-128:-64] for msg in msgs]}")
        log.debug(f"  add_data: {[msg.hex()[-64:] for msg in msgs]}")
        log.debug(f"signature: {spend_bundle.aggregated_signature}")
