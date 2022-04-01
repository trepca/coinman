from clvm.SExp import SExp
from clvm.operators import args_len
from ir.reader import token_stream
from coinman.wallet import disassemble
from collections import defaultdict
import os
import pathlib
from typing import Dict, List
from chia.types.blockchain_format.program import Program
import logging
from chia.wallet.puzzles.load_clvm import compile_clvm, load_clvm

log = logging.getLogger(__name__)

AWAIT_COROUTINE = b"'\x10"


import importlib


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


class Driver:
    def __init__(self, chialisp_file_path):
        full_path = str(chialisp_file_path)
        output_path = "%s.hex" % full_path
        print("Compiling: %s" % full_path)
        compile_clvm(full_path, output_path, ["includes"])
        module = output_path.rsplit("/", 2)[1].replace("/", "")
        name = output_path.rsplit("/", 1)[1].rsplit(".", 2)[0]
        blob = bytes.fromhex(open(output_path).read())
        Program.from_bytes(blob)
        self.output_path = output_path
        self.full_path = full_path
        self.args = _parse_args(full_path)
        self.module = module
        self.name = name

    def run(self, args) -> SExp:
        compile_clvm(self.full_path, self.output_path, ["includes"])
        blob = bytes.fromhex(open(self.output_path).read())
        program = Program.from_bytes(blob)
        return program.run(Program.to(list(args)))

    def __repr__(self):
        return f"[Driver {self.module}.{self.name} args={self.args}]"


async def _run_driver_ops(ops) -> List:
    result = []
    top_op = ops[0]
    if top_op == AWAIT_COROUTINE:
        log.debug("invoking coroutine: %s" % ops)
        ops = [x.decode("utf-8") for x in ops]
        coroutine_name: str = ops[1].strip()
        module_name, func_name = coroutine_name.rsplit(".", 1)
        module = importlib.import_module("coinman.modules." + module_name)
        coroutine = getattr(module, func_name)
        return await coroutine(*await _run_driver_ops(ops[2:]))
    else:
        for op in ops:
            if isinstance(op, list):
                result.append(await _run_driver_ops(op))
            else:
                result.append(op)
    log.debug("Returning results for driver operation: %s", result)
    return result


async def transpile(driver_output) -> Dict:
    return await _run_driver_ops(driver_output)


def load_drivers(path) -> List[Driver]:
    drivers = []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".clvm"):
                full_path = pathlib.Path(dirpath, filename)
                driver = Driver(full_path)
                drivers.append(driver)
    return drivers


if __name__ == "__main__":
    import asyncio

    print("Testing driver transpile")
    test_driver_output = [
        ["coins", [10000, "hello.hi", "one", "two", [10000, "hello.hi", "three"]]]
    ]
    out = asyncio.run(transpile(test_driver_output))
    assert out == {"coins": [["hello world 3"]]}, out
    print("OK.")

    print("Loading drivers...")
    drivers = load_drivers("./drivers")
    assert len(drivers) > 0, drivers
    import pprint

    pprint.pprint(drivers)
    print("OK.")
