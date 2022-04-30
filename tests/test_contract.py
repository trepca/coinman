from coinman.contract import Contract

from coinman.wallet import ContractWallet
from coinman.simulator import NodeSimulator
import pytest


@pytest.mark.asyncio
async def test_public_msg_coin_contract(node: NodeSimulator):
    alice = ContractWallet(node, simple_seed="alice")
    await node.farm_block(alice.puzzle_hash)

    print("Making a contract for pk: %s" % alice.pk())
    contract1 = Contract("tests/msg.clvm", {"pk": bytes(alice.pk())}, amount=1)

    coins = await alice.get_spendable_coins(contract1)
    print("Found coins: %s" % coins)
    if not coins:
        await alice.mint(contract1)
    await node.farm_block()
    result = await alice.run(
        contract1, b"send_message", b"yolo1"
    )  # find the contract coin and spends it, pushes to node too
    print("spent result: %s" % result)
    assert result["status"] == "ok"
    assert result["type"] == "spend"
    # running it again raise ContractWalletError(code=1, msg="all unspent coins have been run")

    await node.farm_block()
    result = await alice.run(contract1, b"get_messages", 0, node.sim.block_height)
    assert result["status"] == "ok"
    assert result["type"] == "query"
    assert len(result["records"]) == 1
    assert result["records"][0]["data"] == [b"yolo1"]
    assert result["records"][0]["state"]["pk"] == bytes(alice.pk())

    result = await alice.run(
        contract1, b"send_message", b"yolo2"
    )  # find the contract coin and spends it, pushes to node too
    print("spent result: %s" % result)
    assert result["status"] == "ok"
    assert result["type"] == "spend"

    await node.farm_block()

    result = await alice.run(contract1, b"get_messages", 0, node.sim.block_height)
    assert result["status"] == "ok"
    assert result["type"] == "query"
    assert len(result["records"]) == 2
    print("data results: %s" % result["records"])
    assert result["records"][1]["data"] == [b"yolo2"]

    # now bob joins the fun
    bob = ContractWallet(node, simple_seed="bob")
    await node.farm_block(bob.puzzle_hash)

    contract2 = Contract("tests/msg.clvm", {"pk": bytes(bob.pk())}, amount=1)
    with pytest.raises(ValueError):
        await bob.run(contract2, b"send_message", b"hi everyone!")
    await bob.mint(contract2)
    await node.farm_block()
    await bob.run(contract2, b"send_message", b"hi everyone!")
    await node.farm_block()
    result = await bob.run(contract2, b"get_messages", 0, node.sim.block_height)
    assert len(result["records"]) == 3
    assert result["records"][2]["data"] == [b"hi everyone!"]
    result = await alice.run(contract1, b"get_messages", 0, node.sim.block_height)
    assert len(result["records"]) == 3
    assert result["records"][2]["state"]["pk"] == bytes(bob.pk())
    assert result["records"][2]["data"] == [b"hi everyone!"]


@pytest.mark.asyncio
async def test_custom_channel_msg_coin_contract(node: NodeSimulator):
    alice = ContractWallet(node, simple_seed="alice")
    await node.farm_block(alice.puzzle_hash)

    bob = ContractWallet(node, simple_seed="bob")
    await node.farm_block(bob.puzzle_hash)

    bobs_contract = Contract(
        "tests/msg.clvm",
        {b"pk": bytes(bob.pk()), b"to": bytes(alice.pk())},
        amount=1,
    )
    alices_contract = Contract(
        "tests/msg.clvm",
        {b"pk": bytes(alice.pk()), b"to": bytes(bob.pk())},
        amount=1,
    )
    await alice.mint(alices_contract)
    await bob.mint(bobs_contract)
    await node.farm_block()

    result = await alice.run(
        alices_contract, b"send_message", b"yolo1"
    )  # find the contract coin and spends it, pushes to node too

    await node.farm_block()
    result = await alice.run(alices_contract, b"get_messages", 0, node.sim.block_height)
    assert result["status"] == "ok"
    assert result["type"] == "query"
    assert len(result["records"]) == 1
    assert result["records"][0]["data"] == [b"yolo1"]
    assert result["records"][0]["state"]["pk"] == bytes(alice.pk())

    result = await alice.run(
        alices_contract, b"send_message", b"yolo2"
    )  # find the contract coin and spends it, pushes to node too
    print("spent result: %s" % result)
    assert result["status"] == "ok"
    assert result["type"] == "spend"

    await node.farm_block()
    result = await bob.run(bobs_contract, b"get_messages", 0, node.sim.block_height)
    assert result["status"] == "ok"
    assert result["type"] == "query"
    assert len(result["records"]) == 2
    assert result["records"][0]["data"] == [b"yolo1"]
    assert result["records"][0]["state"]["pk"] == bytes(alice.pk())
    assert result["records"][1]["data"] == [b"yolo2"]

    await bob.run(bobs_contract, b"send_message", b"this is so cool!")

    await node.farm_block()
    result = await bob.run(bobs_contract, b"get_messages", 0, node.sim.block_height)
    assert result["status"] == "ok"
    assert result["type"] == "query"
    assert len(result["records"]) == 3
    assert result["records"][0]["data"] == [b"yolo1"]
    assert result["records"][0]["state"]["pk"] == bytes(alice.pk())
    assert result["records"][1]["data"] == [b"yolo2"]
    assert result["records"][2]["data"] == [b"this is so cool!"]
    assert result["records"][2]["state"]["pk"] == bytes(bob.pk())


@pytest.mark.asyncio
async def test__fees(node: NodeSimulator):
    alice = ContractWallet(node, simple_seed="alice")
    await node.farm_block(alice.puzzle_hash)
    await node.farm_block(alice.puzzle_hash)

    alices_contract = Contract(
        "tests/msg.clvm",
        {b"pk": bytes(alice.pk())},
        amount=1,
    )
    balance_before = await alice.balance()
    fee = 2 * (10 ** 12)
    # test fee large enough to require multiple coins to pay for
    await alice.mint(alices_contract, fee=fee)
    await node.farm_block()
    assert await alice.balance() == balance_before - fee - 1
    balance_before = await alice.balance()
    result = await alice.run(alices_contract, b"send_message", b"yolo1", fee=fee - 1)
    assert result["status"] == "ok"
    await node.farm_block()
    assert await alice.balance() == 0
