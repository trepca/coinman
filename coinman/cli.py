#!/usr/bin/env python3
import asyncio
import json
from functools import wraps
from pathlib import Path
import sys
from typing import Dict

import click
from aiohttp import web
from chia.util.byte_types import hexstr_to_bytes

VERBOSE = False


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def debug(msg):
    global VERBOSE
    if VERBOSE:
        click.echo(msg)


def parse_launcher(ctx, param, value):
    try:
        if not value:
            raise ValueError
        if not isinstance(value, (str, bytes)):
            raise ValueError
        if len(value) != 66:
            raise click.BadArgumentUsage(
                "Launcher ID must start with 0x and be 66 chars long"
            )
        if value[:2] != "0x":
            raise click.BadArgumentUsage("Launcher ID must start with 0x")
        return hexstr_to_bytes(value)
    except click.BadArgumentUsage:
        raise
    except Exception as e:
        raise click.BadArgumentUsage("Not a valid launcher ID")


@click.group(name="coinman")
@click.option(
    "--config-path",
    "-c",
    help="Path to your config. Defaults to `./coinman.yaml`",
    default="./coinman.yaml",
)
@click.option("-v", "--verbose", help="Show more debugging info.", is_flag=True)
@click.option(
    "--simulator", help="Use simulator instead of connecting to Node", is_flag=True
)
@click.pass_context
def cli(ctx, config_path, verbose, simulator):
    """Manage contract puzzles and their coins.

    Build powerful apps on Chia with contract puzzle coins to easily that expose easy to use APIs"""
    if verbose:
        global VERBOSE
        VERBOSE = True
    from coinman.core import Coinman

    if ctx.invoked_subcommand != "init" and not Path(config_path).exists():
        click.echo(click.style("Config file not found: %s" % config_path, fg="red"))
        sys.exit(1)
    coinman = Coinman.create(config_path, simulate=simulator)
    ctx.obj = coinman


@click.command(help="Initialize a new coinman project")
@click.option(
    "--testnet",
    help="Set network to testnet",
    is_flag=True,
)
@click.argument("path")
@coro
@click.pass_context
async def init(ctx, path, testnet):
    from coinman.core import Coinman

    if Coinman.init(path, testnet):
        click.echo("Coinman project initialized and new wallet generated at %s" % path)
        coinman: Coinman
        async with ctx.obj as coinman:
            w = coinman.get_wallet()
            print(
                "Please send some %s (0.01 should be enough) to the wallet so you can make transactions.\nYour address: %s"
                % (coinman.currency, w.address)
            )


@click.command(help="Show wallet details.")
@click.option(
    "-w",
    "--wallet",
    help="Select a wallet to use. Defaults to first one.",
    default=None,
)
@coro
@click.pass_context
async def show_wallet(ctx, wallet):
    from coinman.core import Coinman

    coinman: Coinman
    async with ctx.obj as coinman:
        w = coinman.get_wallet(wallet)

        click.echo("Public key: ", nl=False)
        click.echo(str(w.pk()))
        click.echo("Address: ", nl=False)
        click.echo(w.address)
        mojos = await w.balance()
        click.echo(
            "Balance: "
            + click.style(
                "%s %s "
                % (
                    mojos / 1000000000000,
                    coinman.currency,
                ),
                bold=True,
            )
            + click.style("(%s mojos)" % str(mojos))
        )


@click.command(help="Get mempool fee related infomation")
@coro
@click.pass_context
async def get_fee_info(
    ctx,
):
    from coinman.core import Coinman

    coinman: Coinman

    async with ctx.obj as coinman:
        result = await coinman.node.get_blockchain_state()
        import pprint

        stats = dict(
            mempool_size=result["mempool_size"],
            mempool_cost=result["mempool_cost"],
            mempool_min_fees=dict(
                cost_5000000=result["mempool_min_fees"]["cost_5000000"]
            ),
            mempool_max_total_cost=result["mempool_max_total_cost"],
        )
        pprint.pprint(stats)


@click.command(help="Invoke a contract coin method.")
@click.option(
    "-w",
    "--wallet",
    help="Select a wallet to use. Defaults to first one.",
    default=None,
)
@click.option(
    "-s",
    "--state",
    help="JSON string to store as initial state. It should be a key/pair form: ",
)
@click.option(
    "-m", "--method", help="Name of the method to invoke (use inspect to list them)"
)
@click.option(
    "-a",
    "--arg",
    help="Argument for the method, you can use multiple times",
    multiple=True,
)
@click.option("-f", "--fee", help="Transaction fee, defaults to 0", default=0)
@click.option("-t", "--amount", help="Coin amount, defaults to 1", default=1)
@click.argument("filename")
@coro
@click.pass_context
async def invoke(
    ctx,
    wallet: str,
    state: Dict,
    method: str,
    arg,
    fee: int,
    amount: int,
    filename: str,
):
    from coinman.core import Coinman

    coinman: Coinman
    state = json.loads(state)

    click.echo("Parsed state: %s" % state)
    async with ctx.obj as coinman:
        result = await coinman.invoke(
            filename,
            state,
            method.encode("utf-8"),
            arg,
            fee=fee,
            amount=amount,
            wallet_id=wallet,
        )
        import pprint

        pprint.pprint(result)


@click.command(help="Inspect a contract.")
@click.option(
    "-a",
    "--amount",
    required=True,
    help="Coin amount contract should use",
)
@click.option(
    "-s",
    "--state",
    help="Chialisp program to store as initial state. It should be a key/pair form: ",
    required=True,
)
@click.argument("filename")
@coro
@click.pass_context
async def inspect(ctx, amount, state, filename):
    """Inspect a contract"""
    state = json.loads(state)
    async with ctx.obj as coinman:
        meta = await coinman.inspect(filename, state, amount)
        print("Inspecting %s" % meta["name"])
        print("Methods: %s" % meta["methods"])
        print("Properties: %s" % meta["properties"])
        print("Hints: %s" % meta["hints"])


@click.command(help="Mint a coin with contract puzzle.")
@click.option(
    "-w",
    "--wallet",
    help="Select a wallet to use. Defaults to first one.",
    default=None,
)
@click.option(
    "-a",
    "--amount",
    help="Coin amount contract should use",
)
@click.option(
    "-s",
    "--state",
    help="Chialisp program to store as initial state. It should be a key/pair form: ",
)
@click.option("-f", "--fee", help="Transaction fee, defaults to 0", default=0)
@click.argument("filename")
@coro
@click.pass_context
async def mint(ctx, wallet, amount, state, fee, filename):
    """Mint a coin with a contract puzzle from FILENAME"""
    state = json.loads(state)

    from coinman.core import Coinman

    coinman: Coinman
    async with ctx.obj as coinman:
        import pprint

        result = await coinman.mint(filename, state, amount, fee, wallet)
        pprint.pprint(result)


@click.command(help="Start coinman service.")
@coro
@click.pass_context
async def runserver(ctx):
    from coinman.core import Coinman

    coinman: Coinman
    async with ctx.obj as coinman:
        if coinman.simulator:
            w = coinman.get_wallet()
            await coinman.node.farm_block(w.puzzle_hash)
            await coinman.node.farm_block(w.puzzle_hash)
        else:
            w = coinman.get_wallet()
            bal = await w.balance() / 1000000000000
            if bal < 0.01:
                click.echo(
                    "Sorry, you need at least 0.01 %s (you currently have %s %s)"
                    % (coinman.currency, bal, coinman.currency)
                )
                click.echo("Send it to: %s" % w.address)
                return
        app = coinman.create_rpc_app()
        host = "127.0.0.1"
        try:
            host = coinman.config["server"]["host"]
        except KeyError:
            pass
        port = 9000
        try:
            port = int(coinman.config["server"]["port"])
        except (KeyError, ValueError):
            pass
        await web._run_app(app, host=host, port=port)


@click.command(help="Run coinman shell to experiment.")
@click.argument("filename", type=click.Path(exists=True), required=False)
def shell(filename=""):
    import IPython
    from IPython.terminal.prompts import Prompts, Token
    from traitlets.config import Config

    class MyPrompt(Prompts):
        def in_prompt_tokens(self, cli=None):
            return [(Token.Prompt, "🌱 >>> ")]

    c = Config()
    c.InteractiveShellApp.extensions = ["coinman.shell"]
    c.InteractiveShell.confirm_exit = False
    c.InteractiveShell.editor = "nano"
    c.InteractiveShell.xmode = "Context"
    c.InteractiveShell.prompts_class = MyPrompt
    c.IPCCompleter.greedy = False
    c.PrefilterManager.multi_line_specials = True
    c.TerminalIPythonApp.display_banner = False
    print(
        """

█▀▀ █ █ █ ▄▀█   █▀ █ █ █▀▀ █   █
█▄▄ █▀█ █ █▀█   ▄█ █▀█ ██▄ █▄▄ █▄▄


Interactive Chia shell.

Type `chia()` to see available commands.
"""
    )

    IPython.start_ipython(argv=[], config=c)


cli.add_command(runserver)
cli.add_command(get_fee_info)
cli.add_command(show_wallet)
cli.add_command(init)
cli.add_command(inspect)
cli.add_command(invoke)
cli.add_command(runserver)
cli.add_command(shell)
cli.add_command(mint)
if __name__ == "__main__":
    shell()
