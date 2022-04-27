import asyncio
import binascii
import logging
import struct
from typing import Dict, List, Set, Union

from blspy import AugSchemeMPL, G1Element, G2Element, PrivateKey
from chia.consensus.cost_calculator import NPCResult
from chia.consensus.default_constants import DEFAULT_CONSTANTS
from chia.full_node.bundle_tools import simple_solution_generator
from chia.full_node.mempool_check_conditions import get_name_puzzle_conditions
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.program import (
    INFINITE_COST,
    Program,
    SerializedProgram,
)
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.coin_record import CoinRecord
from chia.types.coin_spend import CoinSpend
from chia.types.condition_opcodes import ConditionOpcode
from chia.types.spend_bundle import SpendBundle
from chia.util.condition_tools import ConditionOpcode
from chia.util.hash import std_hash
from chia.util.ints import uint64
from chia.wallet.derive_keys import master_sk_to_wallet_sk
from chia.wallet.puzzles.p2_delegated_puzzle_or_hidden_puzzle import (  # standard_transaction
    DEFAULT_HIDDEN_PUZZLE_HASH,
    calculate_synthetic_secret_key,
    puzzle_for_pk,
)
from chia.wallet.sign_coin_spends import sign_coin_spends
from clvm.casts import int_from_bytes
from clvm_tools.binutils import disassemble
from clvm_tools.clvmc import compile_clvm_text
from coinman.contract import Contract
from coinman.node import Node
from chia.util.bech32m import encode_puzzle_hash
from blspy import AugSchemeMPL, BasicSchemeMPL, G1Element, G2Element, PrivateKey

CONDITIONS = dict(
    (k, bytes(v)[0]) for k, v in ConditionOpcode.__members__.items()
)  # pylint: disable=E1101
KFA = {v: k for k, v in CONDITIONS.items()}

log = logging.getLogger(__name__)

mempool = {}


class FeeTooLowError(Exception):
    def __init__(self, spend_bundle: SpendBundle):
        self.spend_bundle = spend_bundle
        self.args = (self.get_fee_details(),)

    def get_fee_details(self):
        fee = self.spend_bundle.fees()
        program = simple_solution_generator(self.spend_bundle)
        npc: NPCResult = get_name_puzzle_conditions(
            program,
            INFINITE_COST,
            cost_per_byte=DEFAULT_CONSTANTS.COST_PER_BYTE,
            mempool_mode=True,
        )

        cost = npc.cost
        return dict(fee=fee, cost=cost, fee_per_cost=fee / cost)


def program_for(code: Union[str, bytes, Program, List]) -> Program:
    if isinstance(code, Program):
        return code
    elif not isinstance(code, (bytes, str)):
        sexp = Program.to(code)
        return Program.fromhex(sexp.as_bin().hex())
    hex_data = compile_clvm_text(code, []).as_bin().hex()
    blob = bytes.fromhex(hex_data)
    return Program.from_bytes(blob)


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


class ContractWallet:
    def __init__(
        self,
        node: Node,
        simple_seed: str = None,
        mnemonic: str = None,
        passphrase: str = "",
        network={},
    ):
        self.network = network
        self.testnet = network.get("testnet", False)
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

        self._pending_coins = set()
        self.usable_coins: Dict[bytes32, Coin] = {}
        self.puzzle: Program = puzzle_for_pk(self.pk())
        self.puzzle_hash: bytes32 = self.puzzle.get_tree_hash()
        agg_sig_me_add_data = network.get("agg_sig_me_additional_data")
        print("Got back %s" % (agg_sig_me_add_data))
        if agg_sig_me_add_data:
            self.agg_sig_me_add_data = bytes32.fromhex(agg_sig_me_add_data)
        else:
            self.agg_sig_me_add_data = DEFAULT_CONSTANTS.AGG_SIG_ME_ADDITIONAL_DATA
        synth_sk: PrivateKey = calculate_synthetic_secret_key(
            self.sk_, DEFAULT_HIDDEN_PUZZLE_HASH
        )
        self.pk_to_sk_dict: Dict[str, PrivateKey] = {
            str(self.pk_): self.sk_,
            str(synth_sk.get_g1()): synth_sk,
        }
        self.address = encode_puzzle_hash(
            self.puzzle_hash, "txch" if self.testnet else "xch"
        )

    def __repr__(self) -> str:
        return f"<Wallet puzzle_hash={self.puzzle_hash}, pk={self.pk_} >"

    #  RPC methods
    async def get_public_keys(self):
        (synthetic_fingerlog.debug,) = struct.unpack("<I", bytes(self.pk_)[:4])
        return [synthetic_fingerlog.debug]

    async def get_private_key(self, fp):
        return {"sk": binascii.hexlify(bytes(self.generator_sk_))}

    def pk_to_sk(self, pk: G1Element) -> PrivateKey:
        assert str(pk) in self.pk_to_sk_dict
        return self.pk_to_sk_dict[str(pk)]

    # Called each cycle before coins are re-established from the simulator.
    def _clear_coins(self):
        self.usable_coins = {}

    # Public key of wallet
    def pk(self) -> G1Element:
        """Return actor's public key"""
        return self.pk_

    # Balance of wallet
    async def balance(self) -> uint64:
        """Return the actor's balance in standard coins as we understand it"""

        spendable_coins: List[Coin] = await self.node.get_coin_records_by_puzzle_hash(
            self.puzzle_hash, include_spent_coins=False
        )
        return uint64(sum(map(lambda x: x.coin.amount, spendable_coins)))

    async def get_spendable_coins(self, contract):
        spendable_coins: List[Coin] = await self.node.get_coin_records_by_puzzle_hash(
            contract.puzzle_hash, include_spent_coins=False
        )

        return spendable_coins

    async def select_coins(self, mojos, excludes=[]):
        spendable_coins: List[Coin] = await self.node.get_coin_records_by_puzzle_hash(
            self.puzzle_hash, include_spent_coins=False
        )
        spendable_amount = uint64(sum(map(lambda x: x.coin.amount, spendable_coins)))
        if mojos > spendable_amount:
            raise ValueError(
                f"Sorry, spendable balance for wallet {self.puzzle_hash} is too low. Requested mojos: {mojos}, available: {spendable_amount}"
            )
        selected_coins: Set = set()
        sum_value = 0
        for coin_record in spendable_coins:
            if sum_value >= mojos and len(selected_coins) > 0:
                break
            if coin_record.coin.name() in excludes:
                continue
            sum_value += coin_record.coin.amount
            selected_coins.add(coin_record.coin)
        return selected_coins

    async def run(self, contract: Contract, method: bytes, *args, fee=None) -> Dict:
        async def pay_for_fees(fee):
            async def make_bundle(coin_to_spend, conditions) -> SpendBundle:
                delegated_puzzle_solution: Program = Program(
                    Program.to((1, conditions))
                )
                solution = Program.to([[], delegated_puzzle_solution, []])
                try:
                    spend_bundle: SpendBundle = await sign_coin_spends(
                        [CoinSpend(coin_to_spend, self.puzzle, solution)],
                        self.pk_to_sk,
                        self.agg_sig_me_add_data,
                        DEFAULT_CONSTANTS.MAX_BLOCK_COST_CLVM,
                    )
                except ValueError:
                    spend_bundle = SpendBundle(
                        [coin_to_spend],
                        G2Element(),
                    )
                return spend_bundle

            if not fee or fee < 1:
                return []
            coins_to_spend = [x for x in await self.select_coins(fee)]
            if not coins_to_spend:
                raise ValueError(f"could not find available coin containing {amt} mojo")
            amount_to_spend = uint64(sum(map(lambda x: x.amount, coins_to_spend)))

            # pick the first one to mint contract coin
            minter = coins_to_spend[0]
            condition_args = []
            if fee < (amount_to_spend - fee):
                condition_args.append(
                    [
                        ConditionOpcode.CREATE_COIN,
                        self.puzzle_hash,
                        amount_to_spend - fee,
                    ]
                )
            spend_bundle: SpendBundle = await make_bundle(minter, condition_args)
            bundles = [spend_bundle]
            for coin in coins_to_spend[1:]:
                bundles.append(await make_bundle(coin, []))
            final_bundle = SpendBundle.aggregate(bundles)
            return final_bundle

        solution = [method, [bytes(x) for x in args]]
        log.debug(
            "Running program %s with solution: %s"
            % (disassemble(contract.program), solution)
        )
        conditions = contract.program.run(solution)
        log.debug("Got back conditions: %s" % disassemble(conditions))
        for condition in conditions.as_python():
            # log.debug("checking conditions: %s" % condition)
            if condition[0] == Contract.QUERY_COINS:
                records = await self._run_query_condition(condition)
                return dict(type="query", status="ok", records=records)

        # fetch pending coins to see if any need to be removed and have been processed
        pending_coin_records = await self.node.get_coin_records_by_names(
            names=[x.name() for x in self._pending_coins], include_spent_coins=True
        )
        pending_coin_records = set([x.coin for x in pending_coin_records if x.spent])
        log.debug("Pending coins: %s" % str(pending_coin_records))
        self._pending_coins -= pending_coin_records

        coin_query = contract.get_coin_query(self.node)
        coins = await coin_query
        for coin in coins:
            if coin not in self._pending_coins:
                break
        else:
            raise ValueError(
                "Sorry, no coins available. Pending: %s" % len(self._pending_coins)
            )
        spend_bundle = await self.make_spend_bundle(coin.coin, contract, solution)
        fee_bundle = await pay_for_fees(fee)
        if fee_bundle:
            spend_bundle = SpendBundle.aggregate([spend_bundle, fee_bundle])
        push_status = await self._push(spend_bundle)
        log.debug("Push status: %s", push_status)
        if push_status:
            return dict(type="spend", status="ok", data=spend_bundle)
        return dict(type="spend", status="error")

    async def _run_query_condition(self, condition):
        query_params = condition[1]
        start, end = condition[2]
        if not start:
            start = None
        else:
            start = int_from_bytes(start)
        if not end:
            end = None
        else:
            end = int_from_bytes(end)
        filters = set([])
        spent = False
        for name, value in query_params:
            if name == b"spent":
                spent = True if bool(value) else False
            elif name == b"hint":
                for _ in value:
                    filters.add(("hint", tuple(value)))
            elif name == b"puzzle_hashes":
                filters.add(("puzzle_hashes", tuple(value)))
            elif name == b"parent_ids":
                filters.add(("parent_ids", tuple(value)))
            elif name == b"names":
                filters.add(("names", tuple(value)))
            else:
                raise ValueError("Filter %s not supported" % name)
        coroutines = []
        for filter_name, value in filters:
            if filter_name == "hint":
                for hint in value:
                    log.debug(
                        "Adding query (spent=%s) for hint: %s" % (spent, hint.hex())
                    )
                    coroutines.append(
                        self.node.get_coin_records_by_hint(
                            hint,
                            include_spent_coins=spent,
                            start_height=start,
                            end_height=end,
                        )
                    )

            elif filter_name == "puzzle_hashes":
                coroutines.append(
                    self.node.get_coin_records_by_puzzle_hash(
                        value,
                        include_spent_coins=spent,
                        start_height=start,
                        end_height=end,
                    )
                )
            elif filter_name == "parent_ids":
                coroutines.append(
                    self.node.get_coin_records_by_parent_ids(
                        value,
                        include_spent_coins=spent,
                        start_height=start,
                        end_height=end,
                    )
                )
            elif filter_name == "names":
                coroutines.append(
                    self.node.get_coin_records_by_names(
                        value,
                        include_spent_coins=spent,
                        start_height=start,
                        end_height=end,
                    )
                )
            else:
                raise ValueError("Filter %s is not supported" % filter_name)
        results = await asyncio.gather(*coroutines)
        log.debug("Got some yummy results: %s" % results)
        results = sum(results, [])
        results = [x for x in results if x.spent_block_index]
        results.sort(key=lambda x: x.spent_block_index)
        data = [
            (
                await self.node.get_puzzle_and_solution(
                    x.coin.name(), height=x.spent_block_index
                ),
                x,
            )
            for x in results
        ]
        records = []
        spent_coin: CoinSpend
        for (spent_coin, coin_record) in data:
            # uncurry puzzle reveal, gets args, and skip initial state to get contract specific state
            raw_state = spent_coin.puzzle_reveal.uncurry()[1].as_python()[0][2:]
            state = {}
            for raw_item in raw_state:
                if len(raw_item) == 1:
                    state[raw_item[0].decode("utf-8")] = b""
                else:
                    state[raw_item[0].decode("utf-8")] = raw_item[1]
            block_record = await self.node.get_block_record_by_height(
                coin_record.spent_block_index
            )
            timestamp = block_record.timestamp
            meta = {
                "id": spent_coin.coin.name(),
                "timestamp": timestamp,
                "height": coin_record.spent_block_index,
            }
            data = spent_coin.solution.to_program().as_python()[1]
            records.append(dict(state=state, data=data, meta=meta))

        return records

    async def get_status(self, spend_bundle_name) -> int:
        pass

    async def mint(self, contract, amount=1, fee=0) -> Coin:
        async def make_bundle(coin_to_spend, conditions) -> SpendBundle:
            delegated_puzzle_solution: Program = Program(Program.to((1, conditions)))
            solution = Program.to([[], delegated_puzzle_solution, []])
            try:
                spend_bundle: SpendBundle = await sign_coin_spends(
                    [CoinSpend(coin_to_spend, self.puzzle, solution)],
                    self.pk_to_sk,
                    self.agg_sig_me_add_data,
                    DEFAULT_CONSTANTS.MAX_BLOCK_COST_CLVM,
                )
            except ValueError:
                spend_bundle = SpendBundle(
                    [coin_to_spend],
                    G2Element(),
                )
            return spend_bundle

        amt = uint64(amount)

        coins_to_spend = [x for x in await self.select_coins(amt + fee)]
        log.debug(
            "Minting contract=%s with amount=%s and fee=%s", contract, amount, fee
        )
        if not coins_to_spend:
            raise ValueError(f"could not find available coin containing {amt} mojo")
        amount_to_spend = uint64(sum(map(lambda x: x.amount, coins_to_spend)))

        # pick the first one to mint contract coin
        minter = coins_to_spend[0]
        condition_args: List[List] = [
            [
                ConditionOpcode.CREATE_COIN,
                contract.puzzle_hash,
                amt,
                [bytes.fromhex(x) for x in contract.hints],
            ],
        ]

        if amt < (amount_to_spend - fee):
            condition_args.append(
                [
                    ConditionOpcode.CREATE_COIN,
                    self.puzzle_hash,
                    amount_to_spend - amt - fee,
                ]
            )
        spend_bundle: SpendBundle = await make_bundle(minter, condition_args)
        bundles = [spend_bundle]
        for coin in coins_to_spend[1:]:
            bundles.append(await make_bundle(coin, []))
        final_spend_bundle = SpendBundle.aggregate(bundles)
        await self._push(final_spend_bundle)
        return final_spend_bundle.additions()[0]

    async def calculate_fee(self, fee_per_cost, contract, method, args) -> int:
        solution = [method, [bytes(x) for x in args]]
        fake_coin = Coin(bytes32(b"a" * 32), bytes32(b"b" * 32), 1)
        spend_bundle = await self.make_spend_bundle(fake_coin, contract, solution)
        program = simple_solution_generator(spend_bundle)
        npc: NPCResult = get_name_puzzle_conditions(
            program,
            INFINITE_COST,
            cost_per_byte=DEFAULT_CONSTANTS.COST_PER_BYTE,
            mempool_mode=True,
        )
        fee = npc.cost * fee_per_cost
        return fee

    async def make_spend_bundle(
        self,
        coin: Coin,
        contract: Contract,
        args,
        **kwargs,
    ) -> SpendBundle:
        amt = uint64(1)
        if "amt" in kwargs:
            amt = kwargs["amt"]
        fee = kwargs.get("fee", 0)
        solution = SerializedProgram.from_program(Program.to(list(args)))
        puzzle = SerializedProgram.from_program(contract.program)
        log.debug("Spending coin: %s" % coin)
        log.debug("--> puzzle: %s" % puzzle)
        log.debug("--> solution: %s" % solution)
        solution_for_coin = CoinSpend(
            coin,
            puzzle,
            solution,
        )
        # The reason this use of sign_coin_spends exists is that it correctly handles
        # the signing for non-standard coins.  I don't fully understand the difference but
        # this definitely does the right thing.
        try:
            spend_bundle: SpendBundle = await sign_coin_spends(
                [solution_for_coin],
                self.pk_to_sk,
                self.agg_sig_me_add_data,
                DEFAULT_CONSTANTS.MAX_BLOCK_COST_CLVM,
            )
        except ValueError:
            spend_bundle = SpendBundle(
                [solution_for_coin],
                G2Element(),
            )
        except:
            spend_bundle = SpendBundle(
                [solution_for_coin],
                G2Element(),
            )
            spend_bundle.debug()
            raise

        return spend_bundle

    async def _push(self, spend_bundle: SpendBundle):
        try:
            pushed: Dict = await self.node.push_tx(spend_bundle)
            log.debug("Pushed: %s" % pushed)
            if pushed["status"] == "SUCCESS":
                self._pending_coins |= set(spend_bundle.removals())
                return True
            return False
        except ValueError as e:
            if "INVALID_FEE_TOO_CLOSE_TO_ZERO" in e.args[0]["error"]:
                raise FeeTooLowError(spend_bundle)
            spend_bundle.debug(self.agg_sig_me_add_data)
            # log.exception("Error pushing bundle: %s", spend_bundle)
            raise e
