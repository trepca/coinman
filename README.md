# coinman

Coinman allows you to develop dApps on [Chia](https://chia.net) blockchain without Chialisp knowledge by abstracting coins as APIs that can be directly integrated into your existing apps or UIs.

Coinman introduces a concept of _contracts_ in Chialisp that expose methods and properties that can be used from external applications.

# Quickstart with an example

Runs in a simulator so has no other requirements but this. No full node required etc.

```sh
git clone git@github.com:trepca/coinman.git
poetry install
poetry shell

```

Clone Chirp, a messaging protocol, that includes UX too.

```
git clone git@github.com:trepca/chirp.git`

cd chirp

coinman init .

# run it in a  simulator

coinman --simulator runserver



```

## Chirp

This is how it looks, it support broadcasting to everyone and group channels. Try [Chirp](https://github.com/trepca/chirp/) out.

![Image](/chirp.png "Chirp - messaging dApp")

# Features

- local simulator
- RPC backend for dApps
- interactive shell to experiment
- supports a concept of **contracts** in [ChiaLisp](https://chialisp.com)

# CLI

```
$ ~ coinman --help
Usage: coinman [OPTIONS] COMMAND [ARGS]...

  Manage contract puzzles and their coins.

  Build powerful apps on Chia with contract puzzle coins to easily that
  expose easy to use APIs

Options:
  -c, --config-path TEXT  Path to your config. Defaults to `./coinman.yaml`
  -v, --verbose           Show more debugging info.
  --simulator             Use simulator instead of connecting to Node
  --help                  Show this message and exit.

Commands:
  get-min-fee  Get current minimum fee
  init         Initialize a new coinman project
  inspect      Inspect a contract.
  invoke       Invoke a contract coin method.
  mint         Mint a coin with contract puzzle.
  runserver    Start coinman service.
  shell        Run coinman shell to experiment.
  show-wallet  Show wallet details.
```

# Concept

Plan is to have many contracts that developers can use to combine and create powerful dApps on Chia without knowing Chialisp.
