#!/usr/bin/env python3
import asyncio

from aiohttp import web
from functools import wraps
import json
import click
import aiohttp_rpc
import logging

import logging.config

import os

logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "verbose": {
                "format": "[{levelname}] [{asctime}] [{module}:{funcName}():{lineno}] - {message}",
                "style": "{",
            },
            "simple": {
                "format": "{levelname} {message}",
                "style": "{",
            },
        },
        "disable_existing_loggers": True,
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "verbose"},
        },
        "root": {
            "handlers": ["console"],
            "level": os.environ.get("LOG_LEVEL", "DEBUG"),
        },
    }
)

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
    help="Path to your config. Defaults to `./config.yaml`",
    default="./config.yaml",
)
@click.option("-v", "--verbose", help="Show more debugging info.", is_flag=True)
@click.pass_context
def cli(ctx, config_path, verbose):
    """Manage beacon coins on Chia network.

    They can be used to store key information in a decentralized and durable way."""
    if verbose:
        global VERBOSE
        VERBOSE = True
    debug(f"Setting up...")
    from coinman.core import Coinman

    coinman = Coinman.create(config_path)
    ctx.obj = coinman


@click.command(help="Start coinman service.")
@coro
@click.pass_context
async def run(ctx):
    from coinman.core import Coinman

    coinman: Coinman
    async with ctx.obj as coinman:
        app = coinman.create_rpc_app()
        host = "0.0.0.0"
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
    from pkg_resources import parse_version  # installed with setuptools

    from traitlets.config import Config
    from IPython.terminal.prompts import Prompts, Token
    import os

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


cli.add_command(run)
cli.add_command(shell)
if __name__ == "__main__":
    shell()
