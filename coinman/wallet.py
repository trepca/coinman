import asyncio
from shutil import disk_usage

from clvm.SExp import SExp
from coinman.util import syncify
import datetime
import logging
import subprocess
from typing import List, Optional, Tuple

from blspy import AugSchemeMPL, G1Element
from chia.consensus.default_constants import DEFAULT_CONSTANTS
from chia.types.blockchain_format.program import (
    INFINITE_COST,
    Program,
    SerializedProgram,
)
from chia.types.condition_opcodes import ConditionOpcode
from chia.types.mempool_inclusion_status import MempoolInclusionStatus
from chia.util.bech32m import decode_puzzle_hash, encode_puzzle_hash
from chia.util.condition_tools import (
    conditions_dict_for_solution,
    pkm_pairs_for_conditions_dict,
)
from chia.util.hash import std_hash

from clvm_tools.clvmc import compile_clvm_text
from chia.wallet.puzzles.load_clvm import load_clvm, compile_clvm
from clvm import KEYWORD_FROM_ATOM
from clvm_tools.binutils import assemble
from clvm_tools.binutils import disassemble as bu_disassemble

import binascii
import datetime
import pytimeparse
import struct

from typing import Dict, List, Tuple, Optional, Union
from blspy import AugSchemeMPL, G1Element, G2Element, PrivateKey

from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.program import Program
from chia.types.spend_bundle import SpendBundle
from chia.types.coin_spend import CoinSpend
from chia.types.coin_record import CoinRecord
from chia.util.ints import uint32, uint64
from chia.util.condition_tools import ConditionOpcode
from chia.util.hash import std_hash
from chia.wallet.derive_keys import master_sk_to_wallet_sk
from chia.wallet.sign_coin_spends import sign_coin_spends
from chia.wallet.puzzles.p2_delegated_puzzle_or_hidden_puzzle import (  # standard_transaction
    puzzle_for_pk,
    calculate_synthetic_secret_key,
    DEFAULT_HIDDEN_PUZZLE_HASH,
)
from chia.consensus.default_constants import DEFAULT_CONSTANTS
from coinman.node import Node

CONDITIONS = dict(
    (k, bytes(v)[0]) for k, v in ConditionOpcode.__members__.items()
)  # pylint: disable=E1101
KFA = {v: k for k, v in CONDITIONS.items()}
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    filename="debug.log",
)
log = logging.getLogger(__name__)
mempool = {}


def program_for(code: Union[str, bytes, Program, List]) -> Program:
    if isinstance(code, Program):
        return code
    elif not isinstance(code, (bytes, str)):
        sexp = Program.to(code)
        return Program.fromhex(sexp.as_bin().hex())
    hex_data = compile_clvm_text(code, []).as_bin().hex()
    blob = bytes.fromhex(hex_data)
    return Program.from_bytes(blob)


class SpendResult:
    def __init__(self, result: Dict):
        """Constructor for internal use.

        error - a string describing the error or None
        result - the raw result from Network::push_tx
        outputs - a list of new Coin objects surviving the transaction
        """
        self.result = result
        if "error" in result:
            self.error: Optional[str] = result["error"]
            self.outputs: List[Coin] = []
        else:
            self.error = None
            self.outputs = result

    def find_standard_coins(self, puzzle_hash: bytes32) -> List[Coin]:
        """Given a Wallet's puzzle_hash, find standard coins usable by it.

        These coins are recognized as changing the Wallet's chia balance and are
        usable for any purpose."""
        return list(filter(lambda x: x.puzzle_hash == puzzle_hash, self.outputs))


class CoinWrapper(Coin):
    """A class that provides some useful methods on coins."""

    def __init__(
        self, parent: Coin, puzzle_hash: bytes32, amt: uint64, source: Program
    ):
        """Given parent, puzzle_hash and amount, give an object representing the coin"""
        super().__init__(parent, puzzle_hash, amt)
        self.source = source

    def puzzle(self) -> Program:
        """Return the program that unlocks this coin"""
        return self.source

    def puzzle_hash(self) -> bytes32:
        """Return this coin's puzzle hash"""
        return self.puzzle().get_tree_hash()

    def smart_coin(self) -> "SmartCoinWrapper":
        """Return a smart coin object wrapping this coin's program"""
        return SmartCoinWrapper(DEFAULT_CONSTANTS.GENESIS_CHALLENGE, self.source)

    def as_coin(self) -> Coin:
        return Coin(
            self.parent_coin_info,
            self.puzzle_hash,
            self.amount,
        )

    @classmethod
    def from_coin(cls, coin: Coin, puzzle: Program) -> "CoinWrapper":
        return cls(
            coin.parent_coin_info,
            coin.puzzle_hash,
            coin.amount,
            puzzle,
        )

    def create_standard_spend(self, priv: PrivateKey, conditions: List[List]):
        delegated_puzzle_solution = Program.to((1, conditions))
        solution = Program.to([[], delegated_puzzle_solution, []])

        coin_spend_object = CoinSpend(
            self.as_coin(),
            self.puzzle(),
            solution,
        )

        # Create a signature for each of these.  We'll aggregate them at the end.
        signature: G2Element = AugSchemeMPL.sign(
            calculate_synthetic_secret_key(priv, DEFAULT_HIDDEN_PUZZLE_HASH),
            (
                delegated_puzzle_solution.get_tree_hash()
                + self.name()
                + DEFAULT_CONSTANTS.AGG_SIG_ME_ADDITIONAL_DATA
            ),
        )

        return coin_spend_object, signature

    def __str__(self):
        return f"[Coin parent={self.parent.name()} puzzle={self.puzzle_hash()} amount={self.amount}]"


class SmartCoinWrapper:
    def __init__(self, genesis_challenge: bytes32, source: Program):
        """A wrapper for a smart coin carrying useful methods for interacting with chia."""
        self.genesis_challenge = genesis_challenge
        self.source = source

    def puzzle(self) -> Program:
        """Give this smart coin's program"""
        return self.source

    def puzzle_hash(self) -> bytes32:
        """Give this smart coin's puzzle hash"""
        return self.source.get_tree_hash()

    def custom_coin(self, parent: Coin, amt: uint64) -> CoinWrapper:
        """Given a parent and an amount, create the Coin object representing this
        smart coin as it would exist post launch"""
        return CoinWrapper(parent, self.puzzle_hash(), amt, self.source)


def _run_program(puzzle, solution):
    result = subprocess.run(["run", puzzle, solution], check=True, capture_output=True)
    log.debug(f"Got back result: {result}")
    if not result.stderr:
        data = assemble(result.stdout.decode("utf-8").strip()).as_python()
        return data


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


def make_coin(address):
    pass


from blspy import BasicSchemeMPL, PrivateKey, G1Element, AugSchemeMPL, G2Element

duration_div = 86400.0
block_time = (600.0 / 32.0) / duration_div


def generate_keys_from_simple_seed(seed: str):
    """returns a secret and public key"""
    hashed_blob = BasicSchemeMPL.key_gen(std_hash(seed.encode("utf-8")))
    r = int.from_bytes(hashed_blob, "big")
    sk = PrivateKey.from_bytes(r.to_bytes(32, "big"))
    return sk, sk.get_g1()


def generate_keys(mnemonic: str, passphrase: str = ""):
    seed = mnemonic_to_seed(mnemonic, passphrase)
    key = AugSchemeMPL.key_gen(seed)
    return key, key.get_g1()


from chia.util.keychain import mnemonic_to_seed
import random


class Wallet:
    def __init__(
        self,
        node: Node,
        simple_seed: str = None,
        mnemonic: str = None,
        passphrase: str = "",
    ):
        """Internal use constructor, use Network::make_wallet

        Fields:
        parent - The Network object that created this Wallet
        name - The textural name of the actor
        pk_ - The actor's public key
        sk_ - The actor's private key
        usable_coins - Standard coins spendable by this actor
        puzzle - A program for creating this actor's standard coin
        puzzle_hash - The puzzle hash for this actor's standard coin
        pk_to_sk_dict - a dictionary for retrieving the secret keys when presented with the corresponding public key
        """
        if mnemonic:
            self.generator_sk_, self.generator_pk_ = generate_keys(mnemonic, passphrase)
        elif simple_seed:
            self.generator_sk_, self.generator_pk_ = generate_keys_from_simple_seed(
                simple_seed
            )
        else:
            raise ValueError("You have to provide either `simple_seed` or `mnemonic`")

        self.node = node
        self.spent_coins: List[Coin] = []
        # Use an indexed key off the main key.
        self.sk_ = master_sk_to_wallet_sk(self.generator_sk_, 0)
        self.pk_ = self.sk_.get_g1()

        self.usable_coins: Dict[bytes32, Coin] = {}
        self.puzzle: Program = puzzle_for_pk(self.pk())
        self.puzzle_hash: bytes32 = self.puzzle.get_tree_hash()

        synth_sk: PrivateKey = calculate_synthetic_secret_key(
            self.sk_, DEFAULT_HIDDEN_PUZZLE_HASH
        )
        self.pk_to_sk_dict: Dict[str, PrivateKey] = {
            str(self.pk_): self.sk_,
            str(synth_sk.get_g1()): synth_sk,
        }

    def __repr__(self) -> str:
        return f"<Wallet puzzle_hash={self.puzzle_hash}, pk={self.pk_} >"

    #  RPC methods
    async def get_public_keys(self):
        (synthetic_fingerprint,) = struct.unpack("<I", bytes(self.pk_)[:4])
        return [synthetic_fingerprint]

    async def get_private_key(self, fp):
        return {"sk": binascii.hexlify(bytes(self.generator_sk_))}

    def pk_to_sk(self, pk: G1Element) -> PrivateKey:
        assert str(pk) in self.pk_to_sk_dict
        return self.pk_to_sk_dict[str(pk)]

    async def mint(
        self, puzzle: Union[str, Program], launcher: CoinWrapper = None, **kwargs
    ) -> List[Coin]:
        """Create a new smart coin based on a parent coin and return the smart coin's living
        coin to the user or None if the spend failed."""
        amt = uint64(1)
        if launcher:
            found_coin: CoinWrapper = launcher
        else:
            usable_coins = await self.node.get_coin_records_by_puzzle_hash(
                self.puzzle_hash, include_spent_coins=False
            )

            found_coin = CoinWrapper.from_coin(
                max(
                    [x.coin for x in usable_coins if x not in self.spent_coins],
                    key=lambda x: x.amount,
                ),
                self.puzzle,
            )
        if "amt" in kwargs:
            amt = kwargs["amt"]
        puzzle_program = program_for(puzzle)
        # Create a puzzle based on the incoming smart coin
        cw = SmartCoinWrapper(DEFAULT_CONSTANTS.GENESIS_CHALLENGE, puzzle_program)
        condition_args: List[List] = [
            [ConditionOpcode.CREATE_COIN, cw.puzzle_hash(), amt],
        ]
        if amt < found_coin.amount:
            condition_args.append(
                [ConditionOpcode.CREATE_COIN, self.puzzle_hash, found_coin.amount - amt]
            )

        delegated_puzzle_solution = Program.to((1, condition_args))
        solution = Program.to([[], delegated_puzzle_solution, []])

        # Sign the (delegated_puzzle_hash + coin_name) with synthetic secret key
        signature: G2Element = AugSchemeMPL.sign(
            calculate_synthetic_secret_key(self.sk_, DEFAULT_HIDDEN_PUZZLE_HASH),
            (
                delegated_puzzle_solution.get_tree_hash()
                + found_coin.name()
                + DEFAULT_CONSTANTS.AGG_SIG_ME_ADDITIONAL_DATA
            ),
        )

        spend_bundle: SpendBundle = SpendBundle(
            [
                CoinSpend(
                    found_coin.as_coin(),  # Coin to spend
                    self.puzzle,  # Puzzle used for found_coin
                    solution,  # The solution to the puzzle locking found_coin
                )
            ],
            signature,
        )
        pushed: Dict[str, Union[str, List[Coin]]] = await self.node.push_tx(
            spend_bundle
        )
        if "error" not in pushed:
            self.spent_coins.append(found_coin)
            return spend_bundle.additions()
        else:
            raise ValueError("Mint failed: %s" % pushed["error"])

    # Called each cycle before coins are re-established from the simulator.
    def _clear_coins(self):
        self.usable_coins = {}

    # Public key of wallet
    def pk(self) -> G1Element:
        """Return actor's public key"""
        return self.pk_

    def get_coin_by_name(self, name):
        return self.usable_coins[name]

    def get_coin(self):
        return list(self.usable_coins.values())[0]

    # Balance of wallet
    async def balance(self) -> uint64:
        """Return the actor's balance in standard coins as we understand it"""

        spendable_coins: List[
            CoinRecord
        ] = await self.node.get_coin_records_by_puzzle_hash(
            self.puzzle_hash, include_spent_coins=False
        )
        return uint64(sum(map(lambda x: x.coin.amount, spendable_coins)))

    # Spend a coin, probably a smart coin.
    # Allows the user to specify the arguments for the puzzle solution.
    # Automatically takes care of signing, etc.
    # Result is an object representing the actions taken when the block
    # with this transaction was farmed.
    async def spend(
        self,
        coin: CoinWrapper,
        solution: Union[str, Program] = None,
        signature=G2Element(),
        **kwargs,
    ) -> Union[SpendResult, SpendBundle]:
        """Given a coin object, invoke it on the blockchain, either as a standard
        coin if no arguments are given or with custom arguments in args="""
        amt = uint64(1)
        if "amt" in kwargs:
            amt = kwargs["amt"]

        delegated_puzzle_solution: Optional[SExp] = None
        if not solution:
            target_puzzle_hash: bytes32 = self.puzzle_hash
            # Allow the user to 'give this much chia' to another user.
            if "to" in kwargs:
                target_puzzle_hash = kwargs["to"].puzzle_hash

            # Automatic arguments from the user's intention.
            if "custom_conditions" not in kwargs:
                solution_list: List[List] = [
                    [ConditionOpcode.CREATE_COIN, target_puzzle_hash, amt]
                ]
            else:
                solution_list = kwargs["custom_conditions"]
            if "remain" in kwargs:
                remainer: Union[SmartCoinWrapper, Wallet] = kwargs["remain"]
                remain_amt = uint64(coin.amount - amt)
                if isinstance(remainer, SmartCoinWrapper):
                    solution_list.append(
                        [
                            ConditionOpcode.CREATE_COIN,
                            remainer.puzzle_hash(),
                            remain_amt,
                        ]
                    )
                elif isinstance(remainer, Wallet):
                    solution_list.append(
                        [ConditionOpcode.CREATE_COIN, remainer.puzzle_hash, remain_amt]
                    )
                else:
                    raise ValueError("remainer is not a wallet or a smart coin")

            delegated_puzzle_solution = Program.to((1, solution_list))
            # Solution is the solution for the old coin.
            solution = Program.to([[], delegated_puzzle_solution, []])
        else:
            delegated_puzzle_solution = program_for(solution)
            solution = delegated_puzzle_solution

        solution_for_coin = CoinSpend(
            coin.as_coin(),
            coin.puzzle(),
            solution,
        )
        print("=> Created a transaction to spend coin: %s" % coin.name())
        print("   - Puzzle reveal: %s" % disassemble(coin.puzzle()))
        print("   - Solution: %s" % disassemble(solution))
        # The reason this use of sign_coin_spends exists is that it correctly handles
        # the signing for non-standard coins.  I don't fully understand the difference but
        # this definitely does the right thing.
        try:
            spend_bundle: SpendBundle = await sign_coin_spends(
                [solution_for_coin],
                self.pk_to_sk,
                DEFAULT_CONSTANTS.AGG_SIG_ME_ADDITIONAL_DATA,
                DEFAULT_CONSTANTS.MAX_BLOCK_COST_CLVM,
            )
        except ValueError:
            spend_bundle = SpendBundle(
                [solution_for_coin],
                G2Element(),
            )

        spend_bundle = SpendBundle(
            [solution_for_coin],
            signature,
        )

        return spend_bundle

    async def send_mojos(self, target: "Wallet", amt: uint64) -> Coin:
        minted_coins = await self.mint(target.puzzle, amt=amt)
        return minted_coins[0]
