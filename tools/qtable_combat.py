"""Drive the real game's combat with the tabular Q-table trained in `rl/`.

    python tools/qtable_combat.py --selftest          # no game needed
    python tools/play.py --policy llm --qtable rl/coarse.pkl --act

The table was trained on `rl/exp1_combat.py`, a toy fight: one enemy, a fixed
10-card deck, and an opponent whose only move is to roll a number and hit you.
Three things make it addressable from the real game at all:

  1. **The deck is the same.** `DECK = (5, 4, 1)` is the real Ironclad starting
     deck, card for card (`STRIKE_IRONCLAD` x5, `DEFEND_IRONCLAD` x4, `BASH` x1).
     That only stays true while the run adds, removes and upgrades nothing -
     which is the experiment's premise, not a property of the game.
  2. **The intent is a number.** `intents[].label` is "12", not prose, so the
     incoming damage needs no text parsing.
  3. **There is no unit conversion at all.** `exp1_combat` was moved to real game
     units on 2026-09-16 - 80 max hp, damage bands in real damage numbers, enemy
     hp sampled per fight and carried in the state - so every number here is
     copied across as it stands. An earlier draft of this file rescaled 80 hp
     into the toy's 20 and it was the scaling, not the lookup, that produced the
     first bug: a real Act 1 hit for 12 became a 3, and 3 was a damage number the
     table had never been trained on.

What is *not* addressed, in the order most likely to break a run:

  * **Several enemies.** The toy has one. `incoming` sums every announced attack
    (they all land on us, so that part is exact), but `enemy_hp` can only be one
    enemy's, so it is the current target's. A three-enemy fight is presented to
    the table as a fight against one enemy that hits three times as hard.
  * **Enemies that do anything but attack.** Blocking, buffing, summoning: the
    table has never seen any of it and cannot represent it.
  * **Debuffs on us.** Weak cuts our damage by 25%; the toy has no such thing, so
    the table will overestimate its own output and misjudge lethal.

A miss is reported, never guessed at: `choose` returns `None` when the row is not
in the table, and the caller falls back to the hand-written policy. The miss rate
is the headline number of this experiment - the last attempt at transfer died of
a 100% miss rate, and a table that quietly rolled dice would have printed a win
rate anyway.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))
import exp1_combat as env  # noqa: E402

# Index i is toy action i. Ordering is load bearing: `env.CARDS` is
# (打击, 防御, 痛击) and the action space is its indices plus END_TURN.
CARD_IDS = ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH")
VULNERABLE_NAMES = ("易伤", "Vulnerable")

assert len(CARD_IDS) == len(env.CARDS), "card id list and the toy's CARDS disagree"


class Miss(Exception):
    """The real state cannot be expressed in the toy's terms. Caller falls back."""


# --- real state -> toy state --------------------------------------------------


def tier_of(incoming: int) -> int:
    """Which of `env.TIERS` this turn's announced damage falls in.

    The toy draws a tier once per fight and rolls within it; the real game does
    not expose any such thing, so it is read back off the number that is visible.
    That is a weaker signal - a light roll inside a hard fight reads as an easy
    fight - and it only matters if `encode_coarse` keeps the field at all.
    """
    for i, (low, high) in enumerate(env.TIERS):
        if incoming <= high:
            return i
    return len(env.TIERS) - 1  # above every band: the hardest one is the closest


def incoming_damage(state: dict) -> int:
    """Every announced attack this turn, summed. Same rule as `play.py:416`.

    Duplicated rather than imported because importing `play` pulls in its CLI and
    its card table; this module has to stay usable from a selftest with no game.
    """
    total = 0
    for enemy in state.get("enemies") or []:
        if (enemy.get("hp") or 0) <= 0:
            continue
        for intent in enemy.get("intents") or []:
            if intent.get("type") != "Attack":
                continue
            try:
                total += int(intent.get("label"))
            except (TypeError, ValueError):
                pass  # an attack with no number on it; ignore rather than guess
    return total


def pick_target(state: dict) -> dict:
    """The enemy the table's `enemy_hp` will describe, and the one we attack.

    Weakest first, matching `play.py`'s greedy policy: the toy's whole notion of
    progress is "the enemy's hp is going down", so the enemy we are counting down
    has to be the one we are hitting. Picking a different target would make
    `enemy_hp` describe a fight we are not having.
    """
    alive = [e for e in (state.get("enemies") or []) if (e.get("hp") or 0) > 0]
    if not alive:
        raise Miss("no living enemy")
    return min(alive, key=lambda e: (e.get("hp") or 0) + (e.get("block") or 0))


def vulnerable_on(enemy: dict) -> int:
    """Stacks of Vulnerable on `enemy`, or 0.

    ⚠️ The shape of `status` has never been seen in a capture with a debuff on
    it - `schemas/schemas.json` only has clean enemies. Both a list of dicts and
    a plain dict are accepted, and anything else raises rather than returning 0:
    a silent 0 would tell the table the enemy is not vulnerable right after we
    spent 2 energy making it vulnerable, and that is the kind of wrong that does
    not announce itself.
    """
    status = enemy.get("status")
    if not status:
        return 0
    if isinstance(status, dict):
        items = [{"name": k, "amount": v} for k, v in status.items()]
    elif isinstance(status, list):
        items = status
    else:
        raise Miss(f"unrecognised status shape {type(status).__name__}")

    for item in items:
        if not isinstance(item, dict):
            raise Miss(f"unrecognised status entry {item!r}")
        name = str(item.get("name") or item.get("id") or "")
        if any(v in name for v in VULNERABLE_NAMES):
            for key in ("amount", "stacks", "value", "turns"):
                if item.get(key) is not None:
                    return int(item[key])
            return 1  # present, but the count is not exposed
    return 0


def hand_counts(state: dict) -> tuple[int, ...]:
    """The toy's `hand`: how many of each of the three cards we are holding.

    Anything that is not one of the three is a foreign card, and a foreign card
    means the premise ("we add and remove nothing") has been broken - by a curse
    from an event, or a card the run picked up. Raising is the point: the table
    has no way to represent it and would otherwise answer as if the card were not
    there.
    """
    counts = [0] * len(CARD_IDS)
    for card in state.get("hand") or []:
        cid = card.get("id")
        if cid not in CARD_IDS:
            raise Miss(f"foreign card in hand: {cid or card.get('name')}")
        counts[CARD_IDS.index(cid)] += 1
    return tuple(counts)


def to_toy_state(state: dict) -> tuple[env.State, dict]:
    """Express the real combat state in the toy's terms. Raises `Miss` if it cannot."""
    player = state.get("player") or {}
    hp, max_hp = player.get("hp"), player.get("max_hp")
    if not hp or not max_hp:
        raise Miss("no player hp in state")

    # The one quantity that is still converted, and it has to be. `encode_coarse`
    # buckets the player by `player_hp * 5 // env.PLAYER_HP`, and PLAYER_HP is a
    # module constant of 80 - but the real character's max hp *grows*: the first
    # live run was already at 87/91 by floor 2, and events and relics push it
    # further. Feeding raw hp in would drift the buckets as the run goes on, and
    # silently: at 50/120 the real 42% would read as bucket 3 instead of 2.
    #
    # Block and incoming stay absolute, and that asymmetry is the point. hp is a
    # ratio because 15 hp means "nearly dead" at 80 max and "fine" at 20. Damage
    # is absolute because the deck is frozen - a 防御 is 5 block whatever else is
    # true, so "can this block cover this hit?" reads the same in both worlds.
    hp_in_toy_scale = max(1, round(hp / max_hp * env.PLAYER_HP))

    alive = [e for e in (state.get("enemies") or []) if (e.get("hp") or 0) > 0]
    target = pick_target(state)

    # Totals, not the target's. The toy wins when `enemy_hp` reaches zero, and
    # the real fight ends when every enemy is dead - so the number counting down
    # has to be the sum. Tracking the target alone would send it back up each
    # time one died, and the table would read that as making no progress.
    enemy_hp = sum(e.get("hp") or 0 for e in alive)
    enemy_max = sum(e.get("max_hp") or e.get("hp") or 0 for e in alive)
    if not enemy_max:
        raise Miss("living enemies report no max hp")

    incoming = incoming_damage(state)
    toy = env.State(
        player_hp=hp_in_toy_scale,
        player_block=player.get("block") or 0,
        enemy_hp=max(1, enemy_hp),
        enemy_max_hp=enemy_max,
        # The target's, not the pack's: vulnerable is what makes our next attack
        # hit harder, and we attack the target.
        enemy_vulnerable=vulnerable_on(target),
        energy=state.get("energy") or 0,
        turn=state.get("round") or 0,
        tier=tier_of(incoming),
        incoming=incoming,
        hand=hand_counts(state),
        # Dropped by `encode_coarse` (measured: merging on them costs nothing,
        # because enemy_hp and block already determine what has been played).
        # Filled with a marker rather than the truth so that `_assert_ignored`
        # can catch an encode_coarse that reads them by mistake.
        draw=(0, 0, 0),
        discard=(0, 0, 0),
    )
    return toy, target


def _assert_ignored(toy: env.State, encode) -> None:
    """Fail loudly if `encode_coarse` reads a field this adapter cannot supply.

    `draw`, `discard` and `turn` are filled with placeholders above. If the key
    changes when they change, the key depends on a lie, and every lookup from the
    real game has been answering a question about a made-up pile. Cheap to check,
    and it can only fire in the one situation where nothing else would.
    """
    other = toy._replace(draw=(1, 1, 1), discard=(2, 2, 2), turn=toy.turn + 7)
    if encode(toy) != encode(other):
        raise Miss("encode_coarse reads draw/discard/turn, which this adapter fakes")


# --- the policy ---------------------------------------------------------------


def choose(state: dict, table: dict, encode, rng: random.Random, adapter):
    """Pick a combat action from the table, or return `None` to fall back.

    Returns `(action, reason)`. The reason always names the toy state and whether
    the row was a hit, because the only thing that distinguishes "the table
    decided" from "the table rolled dice" is written there.
    """
    try:
        toy, target = to_toy_state(state)
        _assert_ignored(toy, encode)
    except Miss as exc:
        return None, f"qtable miss: {exc}"

    key = encode(toy)
    row = table.get(key)
    if row is None:
        return None, f"qtable miss: key not in table {key}"

    # The real game is the authority on what can be played: `can_play` already
    # folds in energy and every card-specific restriction. Intersecting with the
    # toy's own legality keeps us from proposing a card the game will ignore -
    # the silent-no-op failure that has bitten this project four times.
    playable = {}
    for card in state.get("hand") or []:
        if card.get("can_play") and card.get("id") in CARD_IDS:
            playable.setdefault(CARD_IDS.index(card["id"]), card)
    legal = sorted(playable) + [env.END_TURN]

    best = max(row[a] for a in legal)
    action = rng.choice([a for a in legal if row[a] == best])
    tied = sum(1 for a in legal if row[a] == best)

    where = (f"hp{toy.player_hp} blk{toy.player_block} enemy{toy.enemy_hp} "
             f"in{toy.incoming} hand{toy.hand} -> {key}")
    if action == env.END_TURN:
        return adapter.end_turn(), f"qtable end turn ({where}, tie={tied})"

    card = playable[action]
    extra = {"target": target["entity_id"]} if card.get("target_type") == "AnyEnemy" else {}
    return (adapter.play_card(card["index"], **extra),
            f"qtable {card.get('name')} ({where}, tie={tied})")


def load_table(path: str) -> dict:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj[0] if isinstance(obj, tuple) else obj


# --- selftest -----------------------------------------------------------------


def selftest() -> int:
    """Run the adapter over the captured schema sample. No game, no table needed."""
    here = Path(__file__).resolve().parent.parent
    store = json.loads((here / "schemas" / "schemas.json").read_text(encoding="utf-8"))

    samples = [
        (name, v["layers"]["normalized"]["sample"])
        for name, v in store["variants"].items()
        if name.startswith("combat_play") and "normalized" in v["layers"]
    ]
    if not samples:
        print("no combat_play sample in schemas.json")
        return 1

    failures = 0
    for name, sample in samples:
        print(f"--- {name}")
        player = sample.get("player") or {}
        print(f"    real: hp {player.get('hp')}/{player.get('max_hp')} "
              f"block {player.get('block')} energy {sample.get('energy')} "
              f"hand {len(sample.get('hand') or [])} "
              f"enemies {len(sample.get('enemies') or [])} "
              f"incoming {incoming_damage(sample)}")
        try:
            toy, target = to_toy_state(sample)
        except Miss as exc:
            # Not a failure by itself: the captures predate this experiment and
            # were taken in runs that had picked up cards. Report and move on.
            print(f"    Miss: {exc}")
            continue
        print(f"    toy : {toy}")
        print(f"    target: {target.get('entity_id')} "
              f"{target.get('hp')}/{target.get('max_hp')}")
        try:
            key = env.encode_coarse(toy)
        except NotImplementedError:
            print("    encode_coarse is still `raise NotImplementedError` - yours to write")
            continue
        print(f"    key : {key}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run the adapter over schemas/schemas.json and print what it makes")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    ap.print_help()


if __name__ == "__main__":
    main()
