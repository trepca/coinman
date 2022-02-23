import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from email.policy import default
from telnetlib import SE
from typing import List, Optional

from chia.consensus.default_constants import DEFAULT_CONSTANTS
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.program import INFINITE_COST, Program
from chia.types.condition_opcodes import ConditionOpcode
from chia.types.spend_bundle import SpendBundle
from chia.util.condition_tools import conditions_dict_for_solution, pkm_pairs_for_conditions_dict
from chia.util.hash import std_hash
from clvm import KEYWORD_FROM_ATOM
from clvm_tools.binutils import assemble
from clvm_tools.binutils import disassemble as bu_disassemble

from blspy import AugSchemeMPL, G1Element

CONDITIONS = dict((k, bytes(v)[0]) for k, v in ConditionOpcode.__members__.items())  # pylint: disable=E1101
KFA = {v: k for k, v in CONDITIONS.items()}
log = logging.getLogger(__name__)

hash_type = conbytes(
    min_length=32,
    max_length=32,
)


mempool = {}

@dataclass
class Coin:
    id: hash_type
    amount: int
    puzzle: str
    parent_id: hash_type


class Transaction:

    def __init__(self, additions, removals):
        self.additions = additions
        self.removals = removals

def _run_program(puzzle, solution) -> List:
    result = subprocess.run(["run", puzzle, solution], check=True, capture_output=True)
    if isinstance(assemble(result.stdout.decode("utf-8").strip()).as_python(), list):
       return result


def push(*coin_spends):
    results = map(lambda x: _run_program(x.puzzle, x.solution), coin_spends)
    # result just returns conditions for each coin that is spent
    # we now need to apply them together and see that they match
    # when block is farmed, until then we leave it a buffer/mempool
    sb = SpendBundle()

    return True


def coin_as_program(coin: Coin) -> Program:
    """
    Convenience function for when putting `coin_info` into a solution.
    """
    return Program.to([coin.parent_coin_info, coin.puzzle_hash, coin.amount])


def dump_coin(coin: Coin) -> str:
    return disassemble(coin_as_program(coin))


def disassemble(sexp):
    """
    This version of `disassemble` also disassembles condition opcodes like `ASSERT_ANNOUNCEMENT_CONSUMED`.
    """
    kfa = dict(KEYWORD_FROM_ATOM)
    kfa.update((Program.to(k).as_atom(), v) for k, v in KFA.items())
    return bu_disassemble(sexp, kfa)

def push(spend_bundle, agg_sig_additional_data=DEFAULT_CONSTANTS.AGG_SIG_ME_ADDITIONAL_DATA) -> Transaction:
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
        error, conditions, cost = conditions_dict_for_solution(puzzle_reveal, solution, INFINITE_COST)
        log.debug("Got back cost: %s")
        if error:
            log.debug("Error running coin %s: %s", coin.name(), error)
            raise ValueError(error)
        elif conditions is not None:
            for pk_bytes, m in pkm_pairs_for_conditions_dict(conditions, coin_name, agg_sig_additional_data):
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
                    [coin_name] + _.vars for _ in conditions.get(ConditionOpcode.CREATE_COIN_ANNOUNCEMENT, [])
                )
                asserted_coin_announcements.extend(
                    [_.vars[0].hex() for _ in conditions.get(ConditionOpcode.ASSERT_COIN_ANNOUNCEMENT, [])]
                )
                created_puzzle_announcements.extend(
                    [puzzle_reveal.get_tree_hash()] + _.vars
                    for _ in conditions.get(ConditionOpcode.CREATE_PUZZLE_ANNOUNCEMENT, [])
                )
                asserted_puzzle_announcements.extend(
                    [_.vars[0].hex() for _ in conditions.get(ConditionOpcode.ASSERT_PUZZLE_ANNOUNCEMENT, [])]
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

    created_coin_announcement_pairs = [(_, std_hash(b"".join(_)).hex()) for _ in created_coin_announcements]
    if created_coin_announcement_pairs:
        log.debug("created coin announcements")
        for announcement, hashed in sorted(created_coin_announcement_pairs, key=lambda _: _[-1]):
            as_hex = [f"0x{_.hex()}" for _ in announcement]
            log.debug(f"  {as_hex} =>\n      {hashed}")

    eor_coin_announcements = sorted(
        set(_[-1] for _ in created_coin_announcement_pairs) ^ set(asserted_coin_announcements)
    )

    created_puzzle_announcement_pairs = [(_, std_hash(b"".join(_)).hex()) for _ in created_puzzle_announcements]
    if created_puzzle_announcements:
        log.debug("created puzzle announcements")
        for announcement, hashed in sorted(created_puzzle_announcement_pairs, key=lambda _: _[-1]):
            as_hex = [f"0x{_.hex()}" for _ in announcement]
            log.debug(f"  {as_hex} =>\n      {hashed}")

    eor_puzzle_announcements = sorted(
        set(_[-1] for _ in created_puzzle_announcement_pairs) ^ set(asserted_puzzle_announcements)
    )

    log.debug(f"zero_coin_set = {sorted(zero_coin_set)}")
    if created_coin_announcement_pairs or asserted_coin_announcements:
        log.debug(f"created  coin announcements = {sorted([_[-1] for _ in created_coin_announcement_pairs])}")
        log.debug(f"asserted coin announcements = {sorted(asserted_coin_announcements)}")
        log.debug(f"symdiff of coin announcements = {sorted(eor_coin_announcements)}")
    if created_puzzle_announcement_pairs or asserted_puzzle_announcements:
        log.debug(f"created  puzzle announcements = {sorted([_[-1] for _ in created_puzzle_announcement_pairs])}")
        log.debug(f"asserted puzzle announcements = {sorted(asserted_puzzle_announcements)}")
        log.debug(f"symdiff of puzzle announcements = {sorted(eor_puzzle_announcements)}")
    log.debug("=" * 80)
    validates = AugSchemeMPL.aggregate_verify(pks, msgs, spend_bundle.aggregated_signature)
    log.debug(f"aggregated signature check pass: {validates}")
    log.debug(f"pks: {pks}")
    log.debug(f"msgs: {[msg.hex() for msg in msgs]}")
    log.debug(f"  msg_data: {[msg.hex()[:-128] for msg in msgs]}")
    log.debug(f"  coin_ids: {[msg.hex()[-128:-64] for msg in msgs]}")
    log.debug(f"  add_data: {[msg.hex()[-64:] for msg in msgs]}")
    log.debug(f"signature: {spend_bundle.aggregated_signature}")
    if validates and not eor_puzzle_announcements and not eor_coin_announcements:
        return Transaction(created, spent)
if __name__ == "__main__":
    log.debug(_run_program("(c () @)", "('100' '200')"))
