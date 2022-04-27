import logging
import os
import pathlib
from pathlib import Path
from typing import Dict, List
from chia.full_node.mempool_check_conditions import get_name_puzzle_conditions
from chia.types.blockchain_format.coin import Coin

from chia.types.blockchain_format.program import Program
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.wallet.puzzles.load_clvm import compile_clvm
from clvm.SExp import SExp
from coinman.node import Node
from ir.reader import token_stream

log = logging.getLogger(__name__)

AWAIT_COROUTINE = b"'\x10"


import importlib
from importlib.metadata import files

pth = Path([x for x in files("coinman") if ".pth" in str(x)][0].read_text().strip())
CLSP_PATH = pth / Path("coinman/chialisp")


def _parse_args(full_path):
    s = open(full_path).read()
    stream = token_stream(s)
    next(stream)
    next(stream)
    next(stream)
    token = ("", 0)
    args = []
    while 1:
        token = next(stream)
        if token[0] == ")":
            break
        args.append(token[0])
    return args


class Contract:

    QUERY_COINS = b"\x0b\xba"

    def __init__(self, chialisp_file_path, contract_state: Dict, amount=0):
        full_path = str(chialisp_file_path)
        output_path = "%s.hex" % full_path
        log.debug("Compiling: %s" % full_path)
        compile_clvm(full_path, output_path, [CLSP_PATH, Path(".")])
        blob = bytes.fromhex(open(output_path).read())
        program = Program.from_bytes(blob)
        props, methods, _ = program.run([[], [], []]).as_python()[0][1:]
        log.debug("metadata: %s %s" % (props, methods))
        internal_state = [
            (b"#", program.get_tree_hash()),
            (b"$", amount),
        ]
        state = []
        for prop in props:
            val = contract_state.get(prop)
            if val is None:
                if isinstance(prop, bytes):
                    val = contract_state.get(prop.decode("utf-8"), b"")
                else:
                    val = b""
            if not isinstance(val, (bytes, int, str)):
                raise ValueError("Property values can only be int, bytes or str")
            if isinstance(val, str):
                val = val.encode("utf-8")
            state.append((prop, val))
        state = internal_state + state
        log.debug("Currying contract state: %s" % state)
        self.state = dict(state)
        self.props = props
        self.program = program.curry(state)
        log.debug(
            "puzzle hash is: before %s after %s"
            % (program.get_tree_hash(), self.program.get_tree_hash())
        )
        self.output_path = output_path
        self.full_path = full_path
        self.args = _parse_args(full_path)
        # need to run again to get hints with proper state
        _, _, hints = self.program.run(Program.to([[], []])).as_python()[0][1:]
        self.hints = []
        for hint in hints:
            log.debug("Found hint: %s (%s)" % (hint.hex(), hint))
            self.hints.append(hint.hex())
        self.methods = methods

    @property
    def puzzle_hash(self) -> bytes32:
        return self.program.get_tree_hash()

    def get_coin_query(self, node: Node):
        log.debug("querying coins %s" % self.puzzle_hash)
        return node.get_coin_records_by_puzzle_hash(
            self.puzzle_hash, include_spent_coins=False
        )

    def __repr__(self):
        return f"[Contract {self.full_path} args={self.args}]"


async def _run_contract_ops(ops) -> List:
    result = []
    top_op = ops[0]
    if top_op == AWAIT_COROUTINE:
        log.debug("invoking coroutine: %s" % ops)
        ops = [x.decode("utf-8") for x in ops]
        coroutine_name: str = ops[1].strip()
        module_name, func_name = coroutine_name.rsplit(".", 1)
        module = importlib.import_module("coinman.modules." + module_name)
        coroutine = getattr(module, func_name)
        return await coroutine(*await _run_contract_ops(ops[2:]))
    else:
        for op in ops:
            if isinstance(op, list):
                result.append(await _run_contract_ops(op))
            else:
                result.append(op)
    log.debug("Returning results for contract operation: %s", result)
    return result


async def transpile(contract_output) -> Dict:
    return await _run_contract_ops(contract_output)


def load_contracts(path) -> List[Contract]:
    contracts = []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".clvm"):
                full_path = pathlib.Path(dirpath, filename)
                contract = Contract(full_path)
                contracts.append(contract)
    return contracts
