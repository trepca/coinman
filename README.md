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

# Concept

Plan is to have many contracts that developers can use to combine and create powerful dApps on Chia without knowing Chialisp.
