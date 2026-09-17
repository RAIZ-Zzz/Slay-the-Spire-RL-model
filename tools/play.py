"""Drive a run with a swappable meta policy: fixed, heuristic, or an LLM.

    python tools/play.py --policy heuristic --act --max-steps 150
    python tools/play.py --policy llm --act
    python tools/play.py --policy llm --dry-run      # prompts only, no API calls

This is roadmap step A5. `autoplay.read_loop` is the one piece of this project
verified against the real game - nine bugs deep - and `autoplay.choose` is a
plain module-level function, so a policy is swapped in from outside without
touching either. Nothing here can break the loop, because the loop does not know
this file exists.

The three policies, all of which decide only the ~120 meta decisions per run and
leave the ~700 combat decisions on the verified fixed path:

  fixed      autoplay's own: first legal option, every time. Reproducible, and
             the right default while a write path was unproven - but it never
             looks at what it is choosing between.
  heuristic  scores the options by hand-written rules. The baseline any learned
             policy has to beat, and a demonstration of where hand-written rules
             run out: see REST_AT below.
  llm        asks an OpenAI-compatible endpoint (DeepSeek by default). The first
             real piece of Stage 5.

Combat stays on the fixed path for `fixed`, and on the greedy rules for
`heuristic`. For the llm policy that default is now
inverted: it decides every action, and `--no-combat` puts combat back on the
fixed rule. The cost is real - roughly seven times the calls, and the wall clock
of a run becomes the round-trip time multiplied by ~820 decisions rather than
~120.

Every meta decision is logged as JSONL with **the options it was given**, not
just the action taken. That is roadmap step A4, and the options are the half
`.run` save files omit - without them there is no preference model to learn.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import inspect
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# `rl/` is not a package and is not meant to become one - it is a scratch pad for
# the learning experiments. Imported by path so that `--qtable` can share the one
# `encode_coarse` the table was trained with, rather than a copy that drifts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))

_spec = importlib.util.spec_from_file_location(
    "autoplay", Path(__file__).with_name("autoplay.py")
)
autoplay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(autoplay)

# Captured *before* anything replaces `autoplay.choose`. Every policy below falls
# back to "the default" for decisions it does not override, and reading it back
# through the module at call time returns the replacement, i.e. itself. That
# recursed 1000 deep on the first combat state when this was first written.
DEFAULT_CHOOSE = autoplay.choose

_brief_spec = importlib.util.spec_from_file_location(
    "state_brief", Path(__file__).with_name("state_brief.py")
)
_brief_mod = importlib.util.module_from_spec(_brief_spec)
_brief_spec.loader.exec_module(_brief_mod)
brief = _brief_mod.brief

from cli_anything.slay_the_spire_ii.core import action_adapter  # noqa: E402
from cli_anything.slay_the_spire_ii.core.state_adapter import normalize_state  # noqa: E402
from cli_anything.slay_the_spire_ii.utils.sts2_backend import ApiError, Sts2RawClient  # noqa: E402

import game_process  # noqa: E402  (same tools/ dir, after the harness path is set up)


# --- heuristic policy ---------------------------------------------------------

# Map node preference, best first, for a healthy IRONCLAD in act 1.
#
# Treasure is a free relic. Unknown (the ? node) is events and relics, which is
# where act 1 value is. Monster is a card reward against a weak enemy. RestSite
# is held back until it is needed - healing at full HP wastes the node. Shop is
# near-useless *to this agent* because it cannot buy. Elite is last on purpose:
# the combat policy plays the first playable card in hand order, which is weak,
# and an elite punishes exactly that.
NODE_ORDER = ["Treasure", "Unknown", "Monster", "RestSite", "Shop", "Elite"]

# Below this fraction of max HP, a rest site outranks everything.
#
# 0.5 is a guess, and the run of 2026-09-13 showed it is a bad one. HP went
# 100 -> 85 -> 85 -> 85 -> 80 -> 60 -> 48 -> 26% over floors 1-8, losing 0-20
# points per fight. A rest site was on offer at floors 5 and 6 and this rule
# declined both, because 80% and 60% are above the threshold; by floor 8 the run
# was at 24/91 and effectively over. Nothing here knows that - the only way the
# number was found to be wrong was to watch a run and read the HP curve
# afterwards. Setting it from outcomes instead of from intuition is what the RL
# stages are for.
REST_AT = 0.5

# Event option text matching. Brittle by construction: these are localised free
# strings, not enum values, so a game update or a language change silently
# reshuffles the priorities. Logged with every choice so a bad match is visible
# afterwards rather than inferred from a lost run.
EVENT_GOOD = ("最大生命", "遗物", "治疗", "恢复")
EVENT_BAD = ("诅咒", "失去", "受到", "伤害")


def score_node(option: dict, hp_fraction: float) -> tuple:
    """Sort key for a map node: lower is better."""
    node_type = option.get("type") or ""
    if hp_fraction < REST_AT and node_type == "RestSite":
        rank = -1
    else:
        rank = NODE_ORDER.index(node_type) if node_type in NODE_ORDER else len(NODE_ORDER)
    # Tie-break on how many ways out the node has. Three identical Monsters on
    # act 1 floor 1 differ only in this, and keeping two branches open is worth
    # more than keeping one. Negated because the sort wants "lower is better".
    branches = -len(option.get("leads_to") or [])
    return (rank, branches, option.get("index", 0))


# Card-reward thresholds. Taking every card offered is wrong, and the reason is
# dilution: a 10-card starting deck draws each card often, and every card added
# makes the good ones come up less. Past some point a mediocre card is worse than
# no card, so `skip_card_reward` is a real move - and until 2026-09-13 nothing
# here could make it, because the reward screen reports only the cards on offer.
# The bridge now carries the whole deck as counts; these two numbers decide what
# to do with that.
#
# ⚠️ Both are guesses, exactly like REST_AT above, and REST_AT is already known
# to be a bad one. They are the user's to set - this is game knowledge, not
# programming - and the honest way to settle them is the same as for REST_AT:
# measure which value wins more runs, rather than argue about them.
DECK_SOFT_CAP = 18     # beyond this, only a cheap attack is worth the dilution
MAX_COPIES = 4         # more copies of the same card stop adding much


def worth_taking(card: dict, deck: dict | None) -> tuple[bool, str]:
    """Should this card be added to this deck? Returns (verdict, why).

    The `why` is returned rather than logged here because it belongs in the same
    reason string as the action - a run that skipped eleven cards should say
    eleven times what it was weighing, or the decision cannot be reviewed later.
    """
    if not deck:
        # Old bridge, or the deck could not be read. Taking is the previous
        # behaviour; say which of the two it is rather than pretending to judge.
        return True, "no deck info from the bridge, so not judging - taking"

    name = card.get("name")
    size = deck.get("size") or 0
    copies = (deck.get("cards") or {}).get(name, 0)

    if card.get("type") == "Power":
        # A Power pays off over the turns after it lands. The combat policy plays
        # the first playable card in hand order and never holds anything, so it
        # will play this at a random moment and get a fraction of the value -
        # while the card takes a deck slot for the whole run.
        return False, f"{name} is a Power and the fixed combat policy cannot set one up"

    if copies >= MAX_COPIES:
        return False, f"already {copies}x {name} in a {size}-card deck"

    try:
        cost = int(card.get("cost"))
    except (TypeError, ValueError):
        cost = 9

    if size >= DECK_SOFT_CAP and not (card.get("type") == "Attack" and cost <= 1):
        return False, (f"deck is {size} cards (cap {DECK_SOFT_CAP}) and {name} "
                       f"is not a cheap attack")

    return True, f"deck is {size} cards, {copies}x {name} so far"


def score_card(card: dict) -> tuple:
    """Sort key for a card reward: lower is better.

    Tuned to the combat policy that will have to play it. That policy plays the
    first *playable* card in hand order with no plan, so a cheap Attack is worth
    more to it than an expensive Power whose payoff needs setting up. This is the
    heuristic most likely to be wrong once combat gets a real policy - it is a
    statement about the agent, not about the cards.
    """
    by_type = {"Attack": 0, "Skill": 1, "Power": 2}
    try:
        cost = int(card.get("cost"))
    except (TypeError, ValueError):
        cost = 9          # "X" and friends: unknown cost, treat as expensive
    return (by_type.get(card.get("type"), 3), cost, card.get("index", 0))


def choose_event(state):
    """Events: avoid curses and damage, prefer max HP and relics, else first."""
    if state.get("in_dialogue"):
        return action_adapter.advance_dialogue(), "advance event dialogue"

    options = state.get("options") or []
    unlocked = [o for o in options if not o.get("is_locked")]
    if not unlocked:
        return None, f"event {state.get('event_id')} has no unlocked option"

    # A single `is_proceed` option is the event saying "you already chose, leave".
    # Observed on NEOW 2026-09-13: after the blessing the event returns one option
    # titled 继续 with is_proceed=true. Treating that as a fourth choice would be
    # reading a dismissal as a decision.
    if len(unlocked) == 1 and unlocked[0].get("is_proceed"):
        return action_adapter.choose_event_option(0), "leave event (only 继续 left)"

    real = [o for o in unlocked if not o.get("is_proceed")] or unlocked

    def event_score(option):
        text = f"{option.get('title') or ''} {option.get('description') or ''}"
        good = any(w in text for w in EVENT_GOOD)
        bad = any(w in text for w in EVENT_BAD)
        return (bad - good, unlocked.index(option))

    pick = min(real, key=event_score)
    position = unlocked.index(pick)
    text = f"{pick.get('title')} - {pick.get('description')}"
    return action_adapter.choose_event_option(position), f"event @{position}: {text}"


def choose_rest(state):
    """Rest site: heal when hurt, otherwise upgrade.

    Sent by *name*, which doubles as the test for a known unverified assumption:
    the state numbers `restSiteRoom.Options` (model order) while the handler
    indexes `FindAll<NRestSiteButton>` (scene-tree order, unsorted), and nothing
    proves the two agree. The bridge echoes which option it actually clicked, so
    a mismatch between the name asked for and the name echoed is the proof - and
    it shows up in the log the moment it happens instead of being reasoned about.
    """
    options = state.get("options") or []
    enabled = [o for o in options if o.get("is_enabled")]
    if not enabled:
        if state.get("can_proceed"):
            return action_adapter.proceed(), "leave rest site (nothing enabled)"
        return None, "rest_site with no enabled option and no proceed"

    player = state.get("player") or {}
    hp, max_hp = player.get("hp") or 0, player.get("max_hp") or 1
    want_heal = hp / max_hp < 0.7

    def rest_score(option):
        name = f"{option.get('name') or ''} {option.get('id') or ''}"
        is_rest = any(w in name for w in ("休息", "REST", "Rest"))
        is_upgrade = any(w in name for w in ("强化", "升级", "SMITH", "UPGRADE"))
        if want_heal:
            return (0 if is_rest else 1 if is_upgrade else 2, enabled.index(option))
        return (0 if is_upgrade else 1 if is_rest else 2, enabled.index(option))

    pick = min(enabled, key=rest_score)
    position = enabled.index(pick)
    return action_adapter.choose_rest_option(position), (
        f"rest @{position} asked for {pick.get('name')!r} "
        f"(hp {hp}/{max_hp}, want_heal={want_heal}) - compare with the bridge echo"
    )


def heuristic_choose(state, grid_picks: int = 0):
    """Score the options instead of taking the first one."""
    decision = state.get("decision")

    if decision == "combat_play" and CARDS:
        # Only with a card table: without the numbers this would be the fixed
        # policy wearing a different name, and the fixed policy is the baseline
        # this one has to be measurably better than.
        payload, reason = greedy_combat(state, CARDS)
        if payload is not None:
            return payload, reason
        # Nothing worth playing. The fixed policy owns end-of-turn, because the
        # three transient guards that decide *when* a turn is really over live
        # in read_loop and were the hard-won part of 2026-09-12.
        return DEFAULT_CHOOSE(state, grid_picks)

    if decision == "map_select":
        choices = state.get("choices") or []
        if not choices:
            return None, "map with no choices"
        player = state.get("player") or {}
        frac = (player.get("hp") or 0) / (player.get("max_hp") or 1)
        pick = min(choices, key=lambda o: score_node(o, frac))
        position = choices.index(pick)
        leads = [c.get("type") for c in (pick.get("leads_to") or [])]
        return action_adapter.choose_map_node(position), (
            f"node @{position} {pick.get('type')} -> {leads} "
            f"(hp {frac:.0%}, of {[c.get('type') for c in choices]})")

    if decision == "card_reward":
        cards = state.get("cards") or []
        if cards:
            pick = min(cards, key=score_card)
            take, why = worth_taking(pick, state.get("deck"))
            others = [c.get('name') for c in cards if c is not pick]
            if take:
                return action_adapter.select_card_reward(pick["index"]), (
                    f"take [{pick['index']}] {pick.get('name')} {pick.get('type')} "
                    f"cost {pick.get('cost')} {pick.get('rarity')}, over {others} "
                    f"({why})")
            if state.get("can_skip"):
                return action_adapter.skip_card_reward(), (
                    f"skip all of {[pick.get('name')] + others}: {why}")
            # Nothing to skip with. Taking the best of a bad set beats stalling,
            # but the log should not pretend this was the wanted outcome.
            return action_adapter.select_card_reward(pick["index"]), (
                f"take [{pick['index']}] {pick.get('name')} reluctantly - "
                f"wanted to skip ({why}) but can_skip is false")

    if decision == "event_choice":
        return choose_event(state)
    if decision == "rest_site":
        return choose_rest(state)

    # Everything else keeps the verified behaviour: combat, rewards, treasure,
    # card grids, and a shop that buys nothing.
    return DEFAULT_CHOOSE(state, grid_picks)


# --- the card table, and a combat policy that can read it ---------------------

# One file for the whole game library - cards, relics, potions, events - because
# it is one round trip to the bridge and one thing to refresh after a patch.
LIBRARY = Path(__file__).resolve().parent.parent / "data" / "library.json"

# The numbers are not on CardModel - the game computes damage and block through
# hooks at play time - so the description text is the only place they appear.
# `GET /api/v1/cards` dumps the game's own `ModelDb.AllCards`, and these two
# patterns read it. Measured against the v0.107.1 dump of 578 cards:
#
#   194 Attack cards          190 parse (98%)
#   76 cards with gains_block  68 parse (89%)
#
# All twelve misses are cards with no constant to find - damage equal to your
# block, to a debuff's stacks, to cards played this combat; block equal to an
# enemy's poison, to your discard pile, doubled. They are marked variable rather
# than treated as zero, because "this card has no fixed value" and "this card
# does nothing" are different and only the second one should be skipped.
DAMAGE_RE = re.compile(r"造成\s*(\d+)\s*点伤害")
BLOCK_RE = re.compile(r"获得\s*(\d+)\s*点格挡")
VARIABLE_RE = re.compile(r"等[同量]于|每有一张|翻倍|剩余牌数|所打出牌数")


def load_cards(path: Path = LIBRARY) -> dict:
    """id -> {damage, block, variable} for every card the game knows.

    Derived at load time rather than stored, so the file on disk stays the raw
    dump - the thing the game said, not the thing this parser thought. Re-derive
    after a patch with `--dump-cards`; if the parse ever goes wrong, the evidence
    is still sitting next to it.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    table = {}
    for c in raw.get("cards", []):
        desc = c.get("description") or ""
        dmg = DAMAGE_RE.search(desc)
        blk = BLOCK_RE.search(desc)
        table[c["id"]] = {
            "name": c.get("name"),
            "type": c.get("type"),
            "damage": int(dmg.group(1)) if dmg else None,
            "block": int(blk.group(1)) if blk else None,
            "variable": bool(VARIABLE_RE.search(desc)),
            "gains_block": c.get("gains_block"),
        }
    return table


# Loaded once at import. Empty when data/cards.json is missing, which the
# combat branch checks for rather than silently playing worse.
CARDS = load_cards()


def load_library(path: Path = LIBRARY) -> dict:
    """The raw dump, by section, keyed on id.

    Events and relics get no parsing at all - unlike cards there is no number to
    extract, and their text is the effect. What the table buys is identity: an
    event can be recognised by `id` before its options are read, instead of a
    policy matching 失去 / 获得 against localised prose and hoping. That matching
    is what the user called guessing, and it was.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        section: {row["id"]: row for row in raw.get(section, [])}
        for section in ("relics", "potions", "events")
    }


LIBRARY_DB = load_library()


def describe_relics(held: list) -> list:
    """Held relics, with what they do pulled from the library dump.

    The state names the relics you own but not their effects; the library has the
    effects but not what you own. Neither half answers "does this relic change
    what I should take" on its own. Degrades to just the names when the dump has
    not been taken yet, which is the state of things until `--dump-library` runs.
    """
    relics = LIBRARY_DB.get("relics") or {}
    out = []
    for r in held or []:
        row = {"id": r.get("id"), "name": r.get("name")}
        known = relics.get(r.get("id"))
        if known and known.get("description"):
            row["description"] = known["description"]
        out.append(row)
    return out


def incoming_damage(state: dict) -> int:
    """What the enemies have announced they will do to us this turn.

    Structured, unlike our own damage: `intents[].label` is the number the game
    puts on the intent icon, so this needs no text parsing at all.
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
                pass          # an attack with no number on it; ignore rather than guess
    return total


def greedy_combat(state: dict, cards: dict):
    """Play with a plan instead of taking the first playable card.

    Four rules, in order, and each one only uses something the state or the card
    table actually says:

      1. If one card kills the weakest living enemy, play it. One fewer enemy is
         one fewer attack every remaining turn.
      2. If the announced incoming damage would get through our block, add block.
      3. Otherwise hit the weakest enemy with the biggest attack.
      4. Status and Curse cards go last, on leftover energy. Not because they
         are worthless - the card table corrected that: STS2's 黏液 is
         "抽1张牌。 消耗。", so playing it swaps itself for a real card and
         leaves the fight's deck one card thinner. (STS1's Slimed does nothing,
         which is where the idea that this was a wasted play came from.) Most
         Status and Curse cards are 不能被打出 and never reach `can_play` at all;
         the handful that can be played mostly exhaust themselves. They still go
         last, because a card that draws a card is worth less than the card it
         would have drawn.
    """
    hand = state.get("hand") or []
    playable = [c for c in hand if c.get("can_play")]
    if not playable:
        return None, None

    alive = [e for e in (state.get("enemies") or []) if (e.get("hp") or 0) > 0]
    if not alive:
        return None, None
    weakest = min(alive, key=lambda e: (e.get("hp") or 0) + (e.get("block") or 0))

    def info(card):
        return cards.get(card.get("id")) or {}

    def targeted(card):
        return {"target": weakest["entity_id"]} if card.get("target_type") == "AnyEnemy" else {}

    real = [c for c in playable if c.get("type") not in ("Status", "Curse")]

    # 1. lethal
    for card in sorted(real, key=lambda c: -(info(c).get("damage") or 0)):
        dmg = info(card).get("damage")
        if dmg and dmg >= (weakest.get("hp") or 0) + (weakest.get("block") or 0):
            return (action_adapter.play_card(card["index"], **targeted(card)),
                    f"lethal: {card.get('name')} {dmg} >= {weakest['entity_id']} "
                    f"{weakest.get('hp')}+{weakest.get('block') or 0} block")

    # 2. block, if the announced attacks would land
    incoming = incoming_damage(state)
    have = (state.get("player") or {}).get("block") or 0
    if incoming > have:
        blockers = [c for c in real if (info(c).get("block") or 0) > 0]
        if blockers:
            card = max(blockers, key=lambda c: info(c)["block"])
            return (action_adapter.play_card(card["index"], **targeted(card)),
                    f"block {info(card)['block']} vs {incoming} incoming "
                    f"(have {have})")

    # 3. biggest hit on the weakest enemy
    attackers = [c for c in real if (info(c).get("damage") or 0) > 0]
    if attackers:
        card = max(attackers, key=lambda c: info(c)["damage"])
        return (action_adapter.play_card(card["index"], **targeted(card)),
                f"hit {weakest['entity_id']} ({weakest.get('hp')} hp) for "
                f"{info(card)['damage']} with {card.get('name')}")

    # 4. anything else, including the junk - cheapest first
    rest = real or playable
    card = min(rest, key=lambda c: (c.get("type") in ("Status", "Curse"),
                                    int(c["cost"]) if str(c.get("cost")).isdigit() else 9))
    junk = " (self-exhausting status card, thins the deck)" if card.get("type") in ("Status", "Curse") else ""
    return (action_adapter.play_card(card["index"], **targeted(card)),
            f"no scored play left, {card.get('name')}{junk}")


# --- llm policy ---------------------------------------------------------------

OFFICIAL_BASE_URL = "https://api.deepseek.com"
CONFIG_PATH = Path(__file__).with_name("llm_config.json")
CONFIG_TEMPLATE = {
    "base_url": OFFICIAL_BASE_URL,
    "api_key": "",
    "model": "deepseek-flash",
    "_note": "Edit this file, or override any of it with --api-base / --api-key "
             "/ --model, or with LLM_BASE_URL / LLM_API_KEY in the environment. "
             "Contains a secret in plain text; do not commit.",
}

# USD per 1M tokens, off-peak, from DeepSeek's pricing page (read 2026-09-13).
# Peak is exactly double and runs 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri.
#
# The input split is the number worth staring at: a cache hit costs $0.003
# against $0.15 for a miss on deepseek-flash - fifty times less. DeepSeek caches
# prefixes automatically with no request parameter, so the only thing that earns
# it is keeping the system prompt byte-identical across calls, which is why
# SYSTEM is a module constant with no timestamps or run ids in it.
PRICES = {
    "deepseek-flash": {"hit": 0.003, "miss": 0.15, "out": 0.60},
    "deepseek-v4-pro": {"hit": 0.022, "miss": 0.66, "out": 1.98},
}

# Included verbatim in the prompt because DeepSeek's JSON mode guarantees *valid
# JSON*, not a particular schema - there is nothing server-side enforcing these
# keys. Its docs also require the literal word "json" in the prompt and an
# example of the wanted shape, so this string satisfies both requirements.
REPLY_EXAMPLE = """{"action": "choose_map_node", "index": 1, "card_index": null, "target": null, "slot": null, "reason": "the ? node keeps two branches open and we are at full hp"}"""

# ⚠️ The dilution paragraph in SYSTEM below is **not** known to improve anything.
# It was A/B'd once, on a live act-1 floor-12 card reward (15-card deck, offered
# 旋风斩 X-cost / 怨恨 0-cost / 拆卸 1-cost), and both versions picked the same
# card - 怨恨, because a 0-cost attack barely dilutes at all. What changed was the
# reasoning: without the paragraph the model's justification never mentioned the
# deck; with it, it did ("Dismantle is just a modest 1-cost attack the deck
# already has plenty of").
#
# So the claim it supports is "the deck is now a thing the model weighs", not
# "this wins more runs". One state is not evidence of the latter, and the way to
# get that evidence is to run both and compare outcomes - the same answer as for
# REST_AT and DECK_SOFT_CAP.

# Whether the model plays combat changes what it should be told, and for a long
# time it did not: SYSTEM said "combat is played by a fixed rule, not by you"
# unconditionally, while `--policy llm` had been made to take combat over by
# default. The model believed it. Of the 286 combat decisions logged on
# 2026-09-14, 89 gave a reason that recited that rule back ("the fixed combat
# rule plays the first playable card in hand order, so..."), and 130 of 215
# play_card calls went to card_index 0. It was paying an API to imitate a free
# function, badly - the greedy policy at least focus-fires and reads intents.
#
# Still two module constants rather than an f-string built per call: the
# automatic prefix cache matches on a byte-identical system message, and a hit
# costs 1/50th of a miss. Each variant is constant within a run.
COMBAT_IS_YOURS = """- Combat is yours to play, one action at a time. You are shown the hand with `can_play` already computed, the enemies with their `intents`, and your block. Trust `can_play`: it folds in energy and every debuff, so never try to work out affordability yourself (`cost` is a string, and can be "X"). Kill a wounded enemy when one hit finishes it, block when the incoming intents total more than your block, and otherwise hit the weakest with your biggest attack."""

COMBAT_IS_FIXED = """- Combat is played by a fixed rule, not by you. It plays the first playable card in hand order, targets the first living enemy, and ends the turn when nothing is playable. It does not plan, focus fire, or hold cards. Judge every card you take by what that rule will do with it: a cheap attack is worth more than an expensive card whose payoff needs setting up."""

# Combat by `greedy_combat`. Written out because the whole point of the 2026-09-14
# fix was that the model has to be told the rule that is *actually* running: it
# judges card rewards by what the combat policy will do with them, and 89 of that
# day's 286 combat reasons were the model reciting a rule that had been swapped
# out from under it.
COMBAT_IS_GREEDY = """- Combat is played for you by a policy with a plan, not by you, and it is stronger than it sounds: it finishes off an enemy when one card is lethal, adds block when the announced intents total more than the block it has, and otherwise hits the weakest enemy with its biggest attack. It focus-fires and it reads intents. It still does not hold cards across turns or set up a Power, so judge a card reward by whether this policy can use it on the turn it is drawn."""

COMBAT_IS_TABLE = """- Combat is played for you by a Q-table trained on this exact deck, one action at a time, and it falls back to a planning policy whenever it meets a situation it was not trained on. You never choose a card to play. Nothing about combat is yours to weigh."""


# The frozen-deck experiment (2026-09-16): "no cards added, none removed, none
# upgraded - see how far the starting deck goes." Two paragraphs replace the two
# the normal prompt spends on improving the deck, because leaving those in would
# have the model reasoning about a lever it no longer has.
#
# ⚠️ The prompt is the *explanation*, not the enforcement. `with_frozen_deck`
# refuses the actions. A rule that lives only in a prompt is a rule that gets
# broken on some floor at 2am, and this one cannot be broken even once: the whole
# reason a table trained on `exp1_combat` can address the real game is that the
# deck stays (5打击, 4防御, 1痛击). One extra card and the `hand` component of
# every key is wrong - silently, because a foreign card has an id the encoder has
# never seen and the lookup just misses.
DECK_IS_FROZEN = """- **The deck never changes, and that is the experiment.** No card is added, removed or upgraded for the whole run. Card rewards are skipped before you ever see them, rest sites never smith, and a purchase of a card or of a card-removal is refused. Do not plan around improving the deck.
- On a combat rewards screen the `card` entry is not offered to you. Claim the gold, potions and relics, then `proceed` and leave the card entry sitting in the list. The game allows that; a listed reward does not have to be taken.
- So the route is the only real lever, and it matters more than usual. The deck's damage is fixed at roughly 15 a turn and cannot grow, while enemies get bigger every act. A fight is decided by the enemy's total hp and how hard it hits, far more than by how it is played. Prefer weak monster nodes, take rest sites to heal, and avoid elites unless the hp cushion is large. Gold is still worth having for potions and relics."""

# The deck-shaping advice the normal run needs, lifted out of SYSTEM_TEMPLATE so
# that the frozen variant can replace it wholesale rather than contradict it.
DECK_IS_OPEN = """- Removing a card in a shop (`category: card_removal`) is usually the strongest buy for this deck, because it thins without adding.

A card reward can be skipped, and often should be. The deck is drawn through repeatedly, so every card added makes each existing card come up less often. Take a card only if it is better than the average card already in the deck - you are shown the whole deck as counts. A cheap card dilutes less than an expensive one, because it can always be afforded. Skipping is not a wasted reward; it is a faster, more consistent deck.

On a card reward, the row below the cards is `alternatives`, and it is not all "skip": a relic can add a sacrifice, and a reroll re-draws the offer without closing the screen. `skip_card_reward` takes the `index` of the entry you mean - read the `title` and `option_id` and send that one. Sending the wrong index there is accepted without complaint."""

# Set once from `main()`. A module global rather than a parameter because
# `legal_verbs` is called from three places and the flag never changes within a
# process - but it does have to reach `legal_verbs`, because a verb the model is
# told is its only option is a verb it will send.
FROZEN_DECK = False

# --- declining a card reward, our side ----------------------------------------
#
# Floors where we have decided not to take the card. The user's idea, and it is
# the right shape: **stop trying to make the game agree that the reward is
# declined, and just remember not to click it again.**
#
# Seven bridge-side attempts on 2026-09-16 all failed, and each one was accepted
# while changing nothing: `NRewardsScreen.RewardSkippedFrom` (bookkeeping, 26
# calls, no effect), `CardReward.OnSkipped` (found, called, no effect),
# `NRewardButton`'s `RewardSkipped` signal, and the card screen's own 跳过 -
# which the game declares `EndSelectionAndDoNotCompleteReward`, so closing that
# screen leaves the reward outstanding **by design**.
#
# So the reward cannot be resolved. It can, however, be ignored. `skip_card_reward`
# already closes the card screen; what was missing was any memory that we had
# decided not to take it, so the next poll saw a claimable card entry and clicked
# it again - forever. One bit of history per floor fixes it, the same way
# `last_acted_round` fixed "is the hand empty because I played it all, or because
# the cards have not been dealt yet".
#
# ⚠️ This is a general fix, not a --frozen-deck one. `worth_taking` in the
# heuristic decides to skip dilutive cards during ordinary play, and until now
# that decision led straight into the same loop.
#
# 🔴 One thing is still unmeasured: whether the rewards room lets go while a
# card entry sits unclaimed. It looked like it would not, but every observation
# of that was tangled up with a `proceed` that had been broken and a screen that
# earlier attempts had corrupted. `read_loop`'s stuck detector will answer it
# loudly on the next fight - a single repeated decision is exactly what it
# catches.
DECLINED: set[tuple] = set()


def _floor_key(state: dict) -> tuple:
    """Which rewards screen this is. One combat rewards screen per floor."""
    ctx = state.get("context") or {}
    return (ctx.get("act"), ctx.get("floor"))


def with_declined_rewards(inner, tally: dict):
    """Remember a declined card reward and stop clicking it.

    Hooked on the **outgoing action** rather than on any one policy, because
    `skip_card_reward` is emitted from four places - the fixed policy, the
    heuristic's `worth_taking`, the frozen-deck guard, and the model itself.
    Catching the action catches all four with one rule.
    """
    def choose(state: dict, grid_picks: int = 0):
        decision = state.get("decision")
        key = _floor_key(state)

        if decision == "combat_rewards" and key in DECLINED:
            items = state.get("items") or []
            others = [i for i in items if i.get("type") != "card"]
            if not others:
                if state.get("can_proceed"):
                    return action_adapter.proceed(), (
                        f"declined the card on floor {key[1]}; nothing else to "
                        f"claim, leaving the room")
                # Loud, not a retry. If this is what the room does, then a card
                # reward genuinely cannot be declined and the premise has to
                # change - which is worth stopping to learn.
                return None, (
                    f"declined the card on floor {key[1]}, but the room offers no "
                    f"proceed. If this repeats, the rewards screen cannot be left "
                    f"with a card entry outstanding and declining is impossible.")

        payload, reason = inner(state, grid_picks)

        # Record the decision the moment it is made, whoever made it.
        if (decision == "card_reward" and payload
                and payload.get("action") == "skip_card_reward"):
            DECLINED.add(key)
            tally["declined"] += 1
            reason = f"{reason}  [declined floor {key[1]}: will not be claimed again]"

        # Veto a claim on a card we already declined - the heuristic and the
        # fixed policy do not consult `legal_verbs`.
        if (decision == "combat_rewards" and key in DECLINED
                and payload and payload.get("action") == "claim_reward"):
            items = state.get("items") or []
            index = payload.get("index")
            item = next((i for i in items if i.get("index") == index), None)
            if item is None and isinstance(index, int) and 0 <= index < len(items):
                item = items[index]
            if (item or {}).get("type") == "card":
                if state.get("can_proceed"):
                    return action_adapter.proceed(), (
                        f"declined floor {key[1]}'s card; leaving instead of "
                        f"re-claiming it (policy wanted: {reason})")
                return None, (
                    f"declined floor {key[1]}'s card and there is no proceed "
                    f"(policy wanted: {reason})")
        return payload, reason
    return choose


FROZEN_ACTIONS = {
    # verb -> why it is refused. `select_card` and `confirm_selection` are absent
    # on purpose: they are also how combat's own card-selection screens are
    # answered, so refusing them by verb would wedge a fight. The screens that
    # change the deck are reached through a rest site or a shop, both of which
    # are refused above, so they should never come up.
    "select_card_reward": "it adds a card to the deck",
}


def deck_change_reason(payload: dict, state: dict) -> str | None:
    """Why this action would change the deck, or None if it is safe.

    Checked on the action rather than on the decision, because the same decision
    can be answered safely or not: a shop is fine to walk out of and fine to buy
    a potion in, and only the card entries are off limits.
    """
    verb = payload.get("action")
    if verb in FROZEN_ACTIONS:
        return FROZEN_ACTIONS[verb]

    if verb == "shop_purchase":
        items = state.get("items") or []
        index = payload.get("index")
        item = next((i for i in items if i.get("index") == index), None)
        if item is None and isinstance(index, int) and 0 <= index < len(items):
            item = items[index]
        category = (item or {}).get("category")
        if category in ("card", "card_removal"):
            return f"the shop entry is a {category}"
        if item is None:
            # Unknown entry: refusing is the safe direction. Buying by an index
            # nothing matches is how a card gets bought by accident, and the
            # bridge accepts a wrong index without complaint.
            return f"shop entry {index!r} could not be identified"

    if verb == "claim_reward" and state.get("can_proceed"):
        # Do not open a card reward's screen: walk out of the rewards room and
        # leave the entry behind, which is what a human does ("遇到奖励直接跳过,
        # 一个不拿"). Non-card entries are claimed normally first, because
        # `legal_verbs` only hides the card and gold is not a deck change.
        #
        # 🔴 Five versions of this on 2026-09-16. The record, because every one
        # of them looked like the obvious next thing:
        #
        #   v1  refuse the claim, always      -> nothing to substitute when
        #                                        can_proceed was false; stalled
        #   v2  refuse it when can_proceed    -> `proceed` did nothing
        #   v3  NRewardsScreen
        #       .RewardSkippedFrom            -> bookkeeping. 26 calls, no change
        #   v3.5 make `proceed` click the
        #       screen's own button           -> still nothing
        #   v4  the card screen's 跳过 via
        #       the screen's own handler      -> game declares that option
        #                                        `EndSelectionAndDoNotCompleteReward`
        #   v5  CardReward.OnSkipped()        -> CanSkip=True, nothing changed
        #
        # What they have in common is that four of them assumed the failing part
        # was *which method*. It was the *click*: `ForceClick()` presses without
        # releasing, and a Godot button's handler hangs off release. So this is
        # v2's shape again, with the click fixed in the bridge rather than the
        # method swapped for another guess.
        items = state.get("items") or []
        index = payload.get("index")
        item = next((i for i in items if i.get("index") == index), None)
        if item is None and isinstance(index, int) and 0 <= index < len(items):
            item = items[index]
        if (item or {}).get("type") == "card":
            return "the card entry is left behind; the room can be left without it"

    if verb == "choose_rest_option":
        options = state.get("options") or []
        enabled = [o for o in options if o.get("is_enabled")]
        index = payload.get("index")
        if isinstance(index, int) and 0 <= index < len(enabled):
            name = f"{enabled[index].get('name') or ''} {enabled[index].get('id') or ''}"
            if any(w in name for w in ("强化", "升级", "SMITH", "UPGRADE")):
                return "the rest site option is an upgrade"
    return None


def skip_this_reward(state: dict) -> tuple[dict, str]:
    """Answer a card reward with a skip, picking the right alternative index.

    `alternatives` arrived on 2026-09-14 and the lesson with it was that the
    button's own label is localised - `_optionName` came back as 「跳过」 - so the
    only stable handle is the game's `option_id == "Skip"`. Matching on the title
    would work until someone ran the game in English.
    """
    alternatives = state.get("alternatives")
    if alternatives:
        for alt in alternatives:
            if alt.get("option_id") == "Skip" and alt.get("is_enabled", True):
                return (action_adapter.skip_card_reward(alt["index"]),
                        f"frozen deck: skip via alternative[{alt['index']}] "
                        f"{alt.get('title')!r}")
        # There is a row but no Skip in it. Naming what was there beats sending
        # index 0 and hoping - index 0 might be a sacrifice or a reroll.
        titles = [(a.get("index"), a.get("title"), a.get("option_id")) for a in alternatives]
        return None, f"frozen deck: no Skip among alternatives {titles}"
    if state.get("can_skip"):
        return action_adapter.skip_card_reward(), "frozen deck: skip (legacy can_skip)"
    return None, "frozen deck: card reward offers no way to skip"


def with_frozen_deck(inner, tally: dict):
    """Wrap a policy so that nothing it returns can change the deck.

    Card rewards are answered here instead of being passed down - there is
    nothing to decide, and asking an LLM costs a call to be told the one answer
    it is allowed to give. Everything else is passed through and then checked, so
    a refusal is logged with what was refused rather than quietly rewritten.
    """
    def choose(state: dict, grid_picks: int = 0):
        if state.get("decision") == "card_reward":
            tally["skipped_rewards"] += 1
            return skip_this_reward(state)

        payload, reason = inner(state, grid_picks)
        if payload is None:
            return payload, reason

        why = deck_change_reason(payload, state)
        if why is None:
            return payload, reason

        tally["refused"] += 1
        refused = f"{payload.get('action')}({payload})"
        if state.get("can_proceed"):
            return action_adapter.proceed(), (
                f"frozen deck: REFUSED {refused} because {why}; leaving instead "
                f"(policy wanted: {reason})")
        # Nothing safe to substitute. Returning None hands the screen back to
        # `read_loop`, which stalls loudly rather than picking something.
        return None, (f"frozen deck: REFUSED {refused} because {why}, and there "
                      f"is no proceed to fall back on (policy wanted: {reason})")
    return choose


def with_qtable_combat(inner, path: str, stats: dict):
    """Wrap a policy so combat comes from a trained Q-table, with a fallback.

    Only `combat_play` is intercepted; every other decision goes to `inner`
    untouched, which is what makes "LLM plays the meta, the table plays the
    fights" a composition rather than a rewrite.

    On a miss the greedy policy answers instead, and the miss is counted with its
    reason. That is the headline number: the previous attempt to carry a table
    into the real game missed on 100% of lookups, and the run would still have
    printed a plausible-looking log, because a missing row ties across every
    legal action and `max` hands back a perfectly ordinary-looking choice.
    """
    # Imported here rather than at module scope so that a run without --qtable
    # does not depend on rl/ existing at all.
    import exp1_combat
    import qtable_combat

    table = qtable_combat.load_table(path)
    encode = exp1_combat.encode_coarse
    rng = random.Random(0)
    print(f"loaded {len(table):,} rows from {path}")

    def choose(state: dict, grid_picks: int = 0):
        if state.get("decision") != "combat_play":
            return inner(state, grid_picks)

        payload, reason = qtable_combat.choose(state, table, encode, rng,
                                               action_adapter)
        if payload is not None:
            stats["hit"] += 1
            return payload, reason

        stats["miss"] += 1
        # Keep the *kind* of miss, not the key. The first version split on "{"
        # and the key is a tuple, so every distinct key became its own kind -
        # thousands of rows burying the one line that says which assumption
        # broke. Splitting on " (" cuts the key off; a reason that names a
        # foreign card keeps the card's id, which is the part worth having.
        kind = reason.split(" (")[0].split("{")[0].strip() if reason else "unknown"
        stats["reasons"][kind] = stats["reasons"].get(kind, 0) + 1
        fallback, why = heuristic_choose(state, grid_picks)
        return fallback, f"[{reason}] fell back to greedy: {why}"
    return choose


COMBAT_PARAGRAPH = {
    "llm": COMBAT_IS_YOURS,
    "greedy": COMBAT_IS_GREEDY,
    "fixed": COMBAT_IS_FIXED,
    "qtable": COMBAT_IS_TABLE,
}

# ⚠️ Whether a potion is worth buying depends on **who plays combat**, and it was
# nearly written as a flat "never buy potions" on 2026-09-16 - which would have
# been a lie in exactly the mode where it matters. Only the model ever drinks
# one: `use_potion` is in `legal_verbs` for combat_play and the potions are in the
# brief with their descriptions and `target_type`, but `greedy_combat` never
# touches them and `autoplay.pick_potion` returns None on purpose.
#
# Same family as the bug this file already carries a note about: a prompt that
# describes a policy other than the one running. That one cost 89 of 286 combat
# decisions spent reciting a rule that had been swapped out.
POTIONS_ARE_YOURS = """- Potions are worth buying, and drinking them is yours: in combat you are given `use_potion(slot=..., target=...)` along with every potion's description, `can_use_in_combat` and `target_type`. A potion left undrunk is wasted gold, so buy one when the description is good and then actually use it."""

POTIONS_ARE_DEAD = """- Do **not** buy potions. Nothing in this configuration drinks them - combat is not played by you, and neither the greedy policy nor the fixed one ever calls `use_potion` - so a bought potion sits in a slot for the rest of the run and the gold is gone. This is a fact about the program, not about the game."""


def build_system(combat_mode: str, frozen: bool = False) -> str:
    out = SYSTEM_TEMPLATE.replace("{{COMBAT}}", COMBAT_PARAGRAPH[combat_mode])
    out = out.replace("{{POTIONS}}",
                      POTIONS_ARE_YOURS if combat_mode == "llm" else POTIONS_ARE_DEAD)
    return out.replace("{{DECK}}", DECK_IS_FROZEN if frozen else DECK_IS_OPEN)


SYSTEM_TEMPLATE = """You are playing Slay the Spire 2 as IRONCLAD on ascension 0, through a bridge that reports the game state and accepts one action at a time.

You will be given one decision. Reply with a single json object and nothing else, in exactly this shape:

""" + REPLY_EXAMPLE + """

Set every key; use null for the ones the chosen verb does not need. `reason` is one sentence saying why this beats the alternatives you were shown.

Facts about this particular agent that change what is actually good. These matter more than general Slay the Spire advice, because they describe what the rest of the program can and cannot do:

{{COMBAT}}
- The shop can be used: `shop_purchase` takes the `index` of an entry that is both `is_stocked` and `can_afford`. Each entry comes with its description, so judge it on what it does rather than on its name. Gold has value, so weigh it in events.
- **A relic is the best thing to spend gold on.** It is permanent, it costs the deck nothing, and unlike a card it cannot dilute your draws. Read `relic_description` and buy it unless it is plainly useless to this agent. Leaving a shop with gold unspent is usually a mistake, because gold carries no benefit of its own.
{{POTIONS}} A slot can now be emptied: on a combat reward screen offering a potion you have no room for, `discard_potion(slot=...)` throws away one you are carrying, and `claim_reward` then takes the new one. You are shown `held_potions` with their slots on every screen, so compare before swapping - and leaving the new one is a fine answer when what you hold is better.
{{DECK}}

Index conventions are not uniform. Each decision names the one it wants - a list position, or an item's own index field. Send what it names: a wrong index is accepted silently and selects something else.

Prefer whatever keeps the run alive. Outside a rest site, hp does not come back."""


def legal_verbs(state: dict) -> list[str]:
    """What the bridge will accept **in this state**, not merely on this screen.

    This used to be a static table keyed on the decision, and every bug of the
    2026-09-13 session had the same shape: a verb that belongs to the screen but
    not to the moment. `proceed` on a rewards screen with an unclaimed item -
    the room's button is not live until the list is empty, so the call comes back
    "No proceed button available or enabled", eight times in a row. `proceed` in
    a shop, where `can_proceed` is permanently false because the state builder
    re-opens the inventory on every read. `skip_card_reward` on a card_select
    screen, whose handler only accepts `NCardRewardSelectionScreen`.

    Listing a verb the state cannot accept is not a neutral act: the model has no
    way to know the precondition, so it picks the reasonable-sounding one and the
    run wedges. Every condition below is already enforced by the fixed and
    heuristic policies and has been exercised against the real game today - this
    is telling the model what the other two policies already knew.
    """
    d = state.get("decision")

    if d == "map_select":
        return ["choose_map_node"] if state.get("choices") else []

    if d == "card_reward":
        verbs = ["select_card_reward"] if state.get("cards") else []
        # An alternative has to be *enabled*, not merely present. Falls back to
        # `can_skip` only when the bridge predates `alternatives`, because an
        # older bridge sends None there and an empty row also sends [] - and
        # treating "cannot tell" as "none available" would silently drop skipping
        # on every screen.
        alts = state.get("alternatives")
        if alts is None:
            if state.get("can_skip"):
                verbs.append("skip_card_reward")
        elif any(a.get("is_enabled") for a in alts):
            verbs.append("skip_card_reward")
        return verbs

    if d == "overlay":
        # No branch for this screen, so the only honest offer is the exit - and
        # only when the bridge found a live proceed button. Listing anything else
        # would be the 2026-09-13 mistake again: a verb that belongs to the
        # situation in spirit but that this state will refuse.
        return ["proceed"] if state.get("can_proceed") else []

    if d == "combat_rewards":
        # Order matters as a hint: while anything is claimable, that is the only
        # thing the screen will do.
        #
        # "Claimable" is not "listed". A potion offered with no free slot stays
        # in `items` and stays enabled - the game is willing to be clicked, it
        # just cannot hand the potion over. Measured at floor 9 on 2026-09-13:
        # `claim_reward` came back `{"status":"ok","message":"Claiming reward:
        # potion (灰水)"}` and the state was byte-identical 3.2 seconds later.
        #
        # `choose()` has filtered this since that day (autoplay.py, `potions_full`)
        # and this function did not, so the two disagreed about the same screen:
        # with the slots full, the only verb offered here was one that does
        # nothing, and taking the offer wedges the run silently. Found on
        # 2026-09-14 by re-reading the one logged case where the model answered
        # `proceed` anyway and the fallback saved it.
        #
        # `== 0`, not `or 0`: if the field is missing, treating it as "full"
        # would drop every potion reward for the rest of the run in silence.
        potions_full = (state.get("player") or {}).get("open_potion_slots") == 0
        # A card entry is not claimable under --frozen-deck, the same way a
        # potion is not claimable with full slots: clicking it cannot lead
        # anywhere this run will go. `and can_proceed` because hiding it with no
        # way out leaves this function returning nothing at all - which is how
        # the floor-2 stall happened.
        # A card entry we already declined is not claimable - the same shape as
        # a potion with no free slot: the game will take the click and nothing
        # this run wants will happen. `DECLINED` is keyed by floor, so this only
        # hides the entry after a decision was actually made about it; before
        # that the card has to stay claimable, because claiming it is the only
        # way to see which three cards are on offer.
        declined_here = _floor_key(state) in DECLINED
        claimable = [i for i in (state.get("items") or [])
                     if not (potions_full and i.get("type") == "potion")
                     and not (declined_here and i.get("type") == "card")]
        # Same order `choose()` uses: while anything is claimable that is the
        # only offer, so the model cannot walk away from its own card reward.
        # `proceed` appears once the screen is finished - which now includes a
        # screen holding nothing but potions it cannot hand over.
        if claimable:
            return ["claim_reward"]

        verbs = []
        # Nothing claimable. If the only thing in the way is full slots, a slot
        # can now be emptied - the bridge gained `discard_potion` on 2026-09-14,
        # and before it the agent simply forfeited every potion reward from the
        # moment its slots filled. Offered only when all four things hold, since
        # each of them is a separate way for the call to be refused.
        if (potions_full
                and any(i.get("type") == "potion" for i in (state.get("items") or []))
                and state.get("can_discard_potions")
                and state.get("held_potions")):
            verbs.append("discard_potion")
        if state.get("can_proceed"):
            verbs.append("proceed")
        return verbs

    if d == "event_choice":
        if state.get("in_dialogue"):
            return ["advance_dialogue"]
        return ["choose_event_option"] if state.get("options") else []

    if d == "shop":
        # The index mapping here is still the one that could not be confirmed from
        # source: the builder numbers cards, then relics, then potions, then the
        # card removal, while the handler indexes `AllEntries` - and nothing
        # proves those orders agree. Buying was disabled for that reason, which
        # left an agent standing in a shop with enough gold doing nothing.
        #
        # Enabled now, because the log makes the first purchase self-verifying:
        # every shop decision records `brief` (the item list with costs and
        # is_stocked, plus gold), so two consecutive shop states bracket the
        # purchase. If gold fell by exactly `items[k].cost` and `items[k]`
        # is the entry that went out of stock, index k was aligned; if some other
        # entry went out of stock, it was not, and that is the answer rather than
        # another round of reasoning about it.
        #
        # `proceed` is offered unconditionally because `can_proceed` is a false
        # negative here: the handler closes the shop inventory itself first.
        verbs = ["proceed"]
        if any(i.get("is_stocked") and i.get("can_afford") for i in (state.get("items") or [])):
            verbs.insert(0, "shop_purchase")
        return verbs

    if d == "rest_site":
        if any(o.get("is_enabled") for o in (state.get("options") or [])):
            return ["choose_rest_option"]
        return ["proceed"] if state.get("can_proceed") else []

    if d == "treasure":
        if state.get("relics"):
            return ["claim_treasure_relic"]
        return ["proceed"] if state.get("can_proceed") else []

    if d == "relic_select":
        verbs = ["select_relic"] if state.get("relics") else []
        if state.get("can_skip"):
            verbs.append("skip_relic_selection")
        return verbs

    if d == "card_select":
        # Confirm first: after a pick lands, `can_confirm` is the only thing that
        # changed, so offering the card list again re-picks for ever.
        if state.get("can_confirm"):
            return ["confirm_selection"]
        verbs = ["select_card"] if state.get("cards") else []
        if state.get("can_skip"):
            verbs.append("cancel_selection")
        return verbs

    if d == "hand_select":
        if state.get("can_confirm"):
            return ["combat_confirm_selection"]
        return ["combat_select_card"] if state.get("cards") else []

    if d == "combat_play":
        verbs = []
        if any(c.get("can_play") for c in (state.get("hand") or [])):
            verbs.append("play_card")
        # `if potions:`, not `any(potions)`. `any` tests the truthiness of the
        # elements, so a list of potions would still be falsy if the dicts were
        # empty - it happens to work only because real potion entries are
        # non-empty. Testing something other than what you meant is the same
        # family as `or []` flattening "key missing" into "list empty".
        if (state.get("player") or {}).get("potions"):
            verbs.append("use_potion")
        verbs.append("end_turn")
        return verbs

    return []


def fit_kwargs(fn, reply: dict) -> dict:
    """Map a model's reply onto whatever parameter names `fn` actually wants.

    The verbs disagree about what to call the same number:
    `choose_map_node(index)`, `select_card_reward(card_index)`,
    `play_card(card_index, target)`, `use_potion(slot, target)`. That asymmetry
    is the bridge's, and making the model memorise it is asking it to be right
    about something it has no way to check - on the first real call, with the
    correct name printed in the prompt, it answered `{"index": 0}` to a verb
    that wanted `card_index`, and the whole decision fell back.

    So the reply carries "the number" under whichever of those keys it likes, and
    this puts it where the function expects it. `target` is passed by name
    because it is the one parameter with no synonyms.
    """
    params = inspect.signature(fn).parameters
    kwargs = {}

    number = next((reply[k] for k in ("index", "card_index", "slot")
                   if reply.get(k) is not None), None)
    if number is not None:
        slot_name = next((p for p in ("index", "card_index", "slot") if p in params), None)
        if slot_name:
            kwargs[slot_name] = number

    if "target" in params and reply.get("target") is not None:
        kwargs["target"] = reply["target"]
    return kwargs


def fix_target(kwargs: dict, state: dict) -> tuple[dict, str | None]:
    """Make `target` an entity_id that exists, or say why it cannot be.

    The bridge does `targetElem.GetString()`, so a JSON number does not fail
    politely - it raises inside the handler and comes back as HTTP 500. On the
    first `--combat` run the model answered `"target": 0`, meaning the first
    enemy, and every play in that fight died that way.

    An integer is a reasonable thing to have meant, so it is treated as a
    position among the living enemies rather than rejected; anything else that
    does not name a live enemy is refused by name, because sending a target the
    bridge will not recognise is how you get a silent misfire.
    """
    target = kwargs.get("target")
    if target is None:
        return kwargs, None

    alive = [e for e in (state.get("enemies") or []) if (e.get("hp") or 0) > 0]
    ids = [e.get("entity_id") for e in alive]
    if target in ids:
        return kwargs, None

    if isinstance(target, bool):
        pass                      # bool is an int in Python; not a position
    elif isinstance(target, int) and 0 <= target < len(alive):
        kwargs["target"] = ids[target]
        return kwargs, f"read target {target} as {ids[target]}"

    return kwargs, f"target {target!r} is not one of {ids}"


def load_config() -> dict:
    """Read tools/llm_config.json, creating it from the template if absent.

    Created rather than merely defaulted so there is a file to open and edit -
    the point of it is a key that survives closing the terminal, which
    PowerShell's `$env:X = "..."` does not.
    """
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(CONFIG_TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {CONFIG_PATH} - put your key in it, or use --api-key")
        return dict(CONFIG_TEMPLATE)
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        # Reported, not silently replaced with defaults: a config that does not
        # parse means the key you think you are using is not the key being used.
        print(f"{CONFIG_PATH} is not valid JSON ({e}) - ignoring it")
        return {}


def resolve(flag, env_names: tuple[str, ...], config: dict, key: str, default=None):
    """First of: the flag, the named env vars, the config file, the default.

    Returns (value, where_it_came_from). The provenance is not decoration: with
    several sources in play, "which key is this actually using" is not knowable
    by looking at any one of them, and the failure mode is a 401 that looks
    exactly like a typo.
    """
    if flag:
        return flag, "flag"
    for name in env_names:
        if os.environ.get(name):
            return os.environ[name], f"${name}"
    if config.get(key):
        return config[key], CONFIG_PATH.name
    return default, "default"


def is_peak(now: dt.datetime | None = None) -> bool:
    """DeepSeek's peak window: 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.weekday() >= 5:
        return False
    return 1 <= now.hour < 4 or 6 <= now.hour < 10


def format_prompt(state: dict) -> str:
    summary = brief(state)
    verbs = legal_verbs(state)
    return "\n".join([
        json.dumps(summary, ensure_ascii=False, indent=1),
        "",
        f"Legal verbs in THIS state (nothing else will be accepted): {verbs}",
        "Reply with one json object in the shape given above.",
    ])


def make_llm_choose(client, model: str, combat_mode: str, dry_run: bool,
                    usage: dict, frozen: bool = False):
    import openai

    # Built once, here, so the string sent is constant for the whole run - the
    # cache wants a byte-identical prefix - and so it describes the combat
    # arrangement this run is actually using.
    system = build_system(combat_mode, frozen)
    # Who answers `combat_play` when it is not the model. `fixed` is the
    # first-playable-card rule; `greedy` reads intents and focus-fires and is
    # what COMBAT_IS_GREEDY describes. `qtable` is handled a layer up, in
    # `with_qtable_combat`, so combat never reaches here at all.
    combat_fallback = heuristic_choose if combat_mode == "greedy" else DEFAULT_CHOOSE

    def fallback(state, grid_picks, problem):
        usage["fallbacks"] += 1
        payload, why = DEFAULT_CHOOSE(state, grid_picks)
        print(f"       !! llm unusable ({problem}) - fell back")
        return payload, f"[fallback: {problem}] {why}"

    def llm_choose(state, grid_picks: int = 0):
        decision = state.get("decision")
        if decision == "combat_play" and combat_mode != "llm":
            return combat_fallback(state, grid_picks)
        if not legal_verbs(state):
            return DEFAULT_CHOOSE(state, grid_picks)

        prompt = format_prompt(state)
        if dry_run:
            print(f"\n----- would ask about {decision} -----\n{prompt}\n")
            return DEFAULT_CHOOSE(state, grid_picks)

        try:
            response = client.chat.completions.create(
                model=model,
                # System first and byte-identical every call: that prefix is what
                # the automatic cache matches on, and a hit costs 1/50th of a miss.
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=512,
            )
        # Most specific first, and every one degrades to the fixed policy rather
        # than ending the run - a rate limit is a worse decision for one screen,
        # not a reason to abandon a run that was going fine.
        except openai.AuthenticationError as e:
            return fallback(state, grid_picks, f"401, check the key: {e}")
        except openai.RateLimitError as e:
            # DeepSeek returns 402 for an empty balance and the client surfaces it
            # here too. Worth separating: one clears by waiting, the other never
            # does.
            return fallback(state, grid_picks, f"429/402 - rate limit or balance: {e}")
        except openai.APIStatusError as e:
            return fallback(state, grid_picks, f"api {e.status_code}: {e}")
        except openai.APITimeoutError as e:
            # Its own branch, above APIConnectionError which it subclasses: "the
            # endpoint is slow" and "the endpoint is unreachable" need different
            # answers from whoever reads the log.
            return fallback(state, grid_picks, f"timed out: {e}")
        except openai.APIConnectionError as e:
            return fallback(state, grid_picks, f"connection: {e}")

        usage["calls"] += 1
        u = response.usage
        # Read defensively: a missing field should degrade the cost estimate, not
        # raise mid-run.
        hit = getattr(u, "prompt_cache_hit_tokens", None)
        miss = getattr(u, "prompt_cache_miss_tokens", None)
        if hit is None or miss is None:
            hit, miss = 0, u.prompt_tokens
        usage["hit"] += hit
        usage["miss"] += miss
        usage["out"] += u.completion_tokens

        text = (response.choices[0].message.content or "").strip()
        if not text:
            # Documented behaviour: "The API may occasionally return empty
            # content" under JSON mode. Named so an empty reply is never mistaken
            # for a decision.
            return fallback(state, grid_picks, "empty content (known JSON-mode issue)")

        try:
            reply = json.loads(text)
        except json.JSONDecodeError as e:
            return fallback(state, grid_picks, f"unparseable {text[:120]!r}: {e}")

        verb = reply.get("action")
        allowed = legal_verbs(state)
        if verb not in allowed:
            return fallback(state, grid_picks,
                            f"{verb!r} is not accepted in this state; legal: {allowed}")

        fn = getattr(action_adapter, verb)
        try:
            kwargs = fit_kwargs(fn, reply)
            kwargs, note = fix_target(kwargs, state)
            if note and "not one of" in note:
                return fallback(state, grid_picks, note)
            payload = fn(**kwargs)
        except TypeError as e:
            # Survives the mapping only if the verb wanted an argument that was
            # not supplied at all. Named rather than sent: the bridge takes a
            # plausible wrong index without complaint.
            return fallback(state, grid_picks, f"{verb}({reply}) does not fit: {e}")

        why = reply.get("reason") or "no reason given"
        if note:
            why = f"{why}  ({note})"
        return payload, f"[{model}] {why}"

    return llm_choose


GAME_MODS = Path("A:/SteamLibrary/steamapps/common/Slay the Spire 2/mods/STS2_Bridge")
BUILT_DLL = (Path(__file__).resolve().parent.parent
             / "agent-harness/bridge/plugin/bin/Release/net9.0/STS2_Bridge.dll")


def preflight_bridge() -> list[str]:
    """Is the bridge the game is running the one that was last built?

    Three evenings' worth of confusion on 2026-09-16 came from this and nothing
    else. Twice a fix was built, reported as deployed, and tested against the
    previous DLL, because the file is locked while the game runs and the copy
    silently did not happen. Once it was copied but the game had been started
    before the copy, so the old code was already loaded.
    Both look identical from the outside: the fix "does not work".

    Two checks, because they are two different failures:
      * built != deployed  -> the copy never happened. Quit the game and redeploy.
      * deployed newer than the process -> copied, but this process loaded the
        old one. Restart the game.

    Returns a list of problems, empty when everything lines up. Deliberately does
    not exit: a stale bridge is usually worth knowing about and sometimes worth
    ignoring, and a hard stop on a path that might be wrong on another machine
    would be its own trap.
    """
    problems = []
    deployed = GAME_MODS / "STS2_Bridge.dll"
    if not deployed.exists() or not BUILT_DLL.exists():
        return problems  # not this layout; say nothing rather than guess

    import hashlib

    def md5(p):
        return hashlib.md5(p.read_bytes()).hexdigest()

    dep_hash, built_hash = md5(deployed), md5(BUILT_DLL)
    if dep_hash != built_hash:
        problems.append(
            f"the built bridge is NOT the deployed one\n"
            f"       built    {built_hash}  {BUILT_DLL}\n"
            f"       deployed {dep_hash}  {deployed}\n"
            f"       -> quit the game, then copy the built one over the deployed one")
        return problems

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue | "
             "Select-Object -First 1 -ExpandProperty StartTime"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        if out:
            started = dt.datetime.strptime(out, "%A, %B %d, %Y %I:%M:%S %p")
            copied = dt.datetime.fromtimestamp(deployed.stat().st_mtime)
            if copied > started:
                problems.append(
                    f"the deployed bridge is newer than the running game\n"
                    f"       dll copied  {copied:%Y-%m-%d %H:%M:%S}\n"
                    f"       game started {started:%Y-%m-%d %H:%M:%S}\n"
                    f"       -> this process loaded the old code; restart the game")
    except Exception:
        pass  # a locale or a missing powershell costs the second check, not the run
    return problems


def report_cost(usage: dict, model: str, api_base: str) -> None:
    print(f"llm calls: {usage['calls']}   fallbacks: {usage['fallbacks']}")
    print(f"input hit/miss: {usage['hit']}/{usage['miss']}   output: {usage['out']}")
    rates = PRICES.get(model)
    if not rates:
        print(f"  (no recorded price for {model!r}; the token counts above are the "
              f"real measurement - price them yourself)")
        return
    if api_base.rstrip("/") != OFFICIAL_BASE_URL:
        # The token counts are measured; the dollar figure is an assumption about
        # somebody else's price list. Say which is which rather than printing a
        # number that looks equally solid.
        print(f"  (endpoint is {api_base}, not the official one - the cost below "
              f"applies DeepSeek's published rates and may not be what you pay)")
    mult = 2.0 if is_peak() else 1.0
    cost = (usage["hit"] / 1e6 * rates["hit"]
            + usage["miss"] / 1e6 * rates["miss"]
            + usage["out"] / 1e6 * rates["out"]) * mult
    print(f"estimated cost: ${cost:.4f} ({'peak' if mult > 1 else 'off-peak'} rates, "
          f"off-peak base ${rates['hit']}/${rates['miss']}/${rates['out']} per 1M, "
          f"read 2026-09-13)")
    if usage["calls"] and not usage["hit"]:
        print("  (zero cache hits - the system prefix is not being reused; check "
              "that SYSTEM has not picked up anything per-request)")


# --- shared plumbing ----------------------------------------------------------


def with_logging(inner, log_path: Path, counts: dict):
    """Wrap a policy so every meta decision is written to the trajectory log.

    One wrapper for all three policies, so that what gets recorded does not
    depend on which one is running - otherwise comparing them later means
    trusting that three separate log-writing blocks agreed on what a record is.
    """
    log = log_path.open("a", encoding="utf-8")

    def logged(state, grid_picks: int = 0):
        decision = state.get("decision")
        payload, reason = inner(state, grid_picks)
        # Combat used to be excluded, because a fixed rule playing the first
        # legal card is not a decision worth a record. With a model deciding
        # every action that exclusion hides the thing most worth reviewing - and
        # it is also the half that has to be compared against the fixed and
        # greedy policies later. Roughly 700 more lines per run, ~1 KB each.
        if True:
            counts[decision] = counts.get(decision, 0) + 1
            ctx = state.get("context") or {}
            log.write(json.dumps({
                "t": time.strftime("%H:%M:%S"),
                "act": ctx.get("act"),
                "floor": ctx.get("floor"),
                "decision": decision,
                # Exactly what the decider was shown, not a re-derived subset.
                # The merge briefly logged only `options`, which is a different
                # claim: comparing two policies later means knowing they saw the
                # same thing, and `brief` *is* that thing - including the deck,
                # which is what makes "should I take this card" answerable at all.
                "brief": brief(state),
                "action": payload if payload not in (None, autoplay.WAIT)
                          else ("WAIT" if payload is autoplay.WAIT else None),
                "reason": reason,
            }, ensure_ascii=False) + "\n")
            log.flush()
        return payload, reason

    return logged



def enter_run(game, character: str, ascension: int, mode: str = "auto",
              timeout: float = 60.0) -> str | None:
    """Make sure a run is in progress, from wherever we are. None on success.

    Covers all three places this can be called from, which is the point: run 1
    can start at the main menu, run 5 starts on the game-over screen of run 4,
    and calling it while already playing does nothing. One function, so the
    "first run" and "next run" paths cannot drift apart.

        already in a run   ->  nothing
        game over          ->  return_to_main_menu, then start
        main menu          ->  continue_game or start_new_game, per `mode`

    `mode` is auto / new / continue. `auto` resumes a save if the menu offers
    one and starts fresh otherwise, so the same call serves both of the things
    this is for: new game -> die -> menu -> new game, and continue -> die ->
    menu -> new game.

    Each step waits for the state to actually change rather than sleeping a
    guessed amount - the same rule the combat guards settled on after four
    attempts at picking a duration.
    """
    def wait_for(predicate, what):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = normalize_state(game.get_state())
            except ApiError:
                time.sleep(1.0)
                continue
            if predicate(state):
                return state
            time.sleep(0.5)
        return None

    state = normalize_state(game.get_state())

    # Already playing: nothing to do. Makes this safe to call at the top of every
    # run, which is what lets one function cover "we just started" and "the last
    # one just ended" instead of two paths that drift apart.
    if state.get("decision") not in ("menu", "game_over"):
        return None

    if state.get("decision") == "game_over":
        # Sent repeatedly, not once. The death screen is two stages: straight
        # after dying only Continue exists (`can_return_to_main_menu: false`,
        # measured live 2026-09-14 at floor 17), and the return button is built
        # after that click. `ExecuteReturnToMainMenu` presses whichever is
        # available and, having pressed Continue, looks for the return button in
        # the same call - which cannot be there yet, so it answers
        # "Continuing past game over screen" and stops.
        #
        # One call therefore gets through stage one and reports success, and the
        # old code then waited 60 seconds for a menu nobody had asked for. Each
        # call is safe to repeat: it clicks what is live and nothing else.
        deadline = time.monotonic() + timeout
        last_refusal = None
        while time.monotonic() < deadline:
            # Only knock when somebody is home. The death screen animates its
            # score tally in, and for the first moments *neither* button is
            # enabled - a run on 2026-09-14 died at floor 12 and the immediate
            # call came back "Game over screen is open but no usable main-menu
            # button is available". Unlike `can_start_new_game` on the main menu,
            # these two flags are real: they are read off the buttons.
            if state.get("can_continue") or state.get("can_return_to_main_menu"):
                try:
                    reply = game.post_action("return_to_main_menu")
                except ApiError as e:
                    return f"could not leave the game-over screen: {e}"
                # Read, not discarded - but a refusal here means "too early", not
                # "wrong action", so it is kept and only reported if the deadline
                # runs out. Treating it as fatal is what turned a screen that was
                # a second away from being ready into a dead session.
                if isinstance(reply, dict) and reply.get("status") == "error":
                    last_refusal = reply.get("error")
            state = normalize_state(game.get_state())
            if state.get("decision") == "menu":
                break
            time.sleep(0.5)
        else:
            return (f"still not at the menu {timeout:.0f}s after repeated "
                    f"return_to_main_menu (last decision={state.get('decision')!r}, "
                    f"can_continue={state.get('can_continue')!r}, "
                    f"can_return_to_main_menu={state.get('can_return_to_main_menu')!r}"
                    + (f", last refusal: {last_refusal!r}" if last_refusal else "")
                    + ")")

    if state.get("decision") != "menu":
        return f"not at a menu or game-over screen (decision={state.get('decision')!r})"

    # `auto` resumes a save when there is one and starts fresh otherwise, which
    # is what makes the same call work for run 1 and for run 5: after a death
    # there is nothing to continue, so it starts a new one on its own.
    menu = state.get("menu") or {}
    resume = mode == "continue" or (mode == "auto" and menu.get("can_continue_game"))
    verb = "continue_game" if resume else "start_new_game"
    kwargs = {} if resume else {"character": character, "ascension": ascension}

    try:
        result = game.post_action(verb, **kwargs)
    except ApiError as e:
        return f"{verb} failed: {e}"
    if isinstance(result, dict) and result.get("status") == "error":
        return f"{verb} refused: {result.get('error')}"

    if wait_for(lambda st: st.get("decision") != "menu", "a run") is None:
        return f"{verb} returned ok but the menu is still up after {timeout:.0f}s"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", choices=("fixed", "heuristic", "llm"),
                    default="heuristic")
    ap.add_argument("--act", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the agent's own random choices (map node, card reward, ...). Omitted = a fresh one, printed either way. Note this does NOT seed the game itself, so it replays the decision sequence, not the run.")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--bridge-url", default="http://localhost:15526",
                    help="the game bridge, not the model endpoint")
    ap.add_argument("--log", type=Path, default=None,
                    help="default trajectories/<policy>.jsonl")
    # llm only
    ap.add_argument("--model", default=None, help="[llm] model id")
    ap.add_argument("--api-base", default=None,
                    help=f"[llm] OpenAI-compatible endpoint (default {OFFICIAL_BASE_URL})")
    ap.add_argument("--api-key", default=None,
                    help="[llm] prefer the config file or env var so it stays out "
                         "of your shell history")
    ap.add_argument("--combat", choices=("llm", "greedy", "fixed"), default="llm",
                    help="[llm] who plays combat. 'llm' is every action by the "
                         "model (~7x the calls); 'greedy' is the policy that "
                         "reads intents and focus-fires, and is the strongest "
                         "hand-written one; 'fixed' is first-playable-card. "
                         "--qtable overrides this. Whichever is chosen is what "
                         "the system prompt says is running")
    ap.add_argument("--no-combat", action="store_true",
                    help="[llm] deprecated alias for --combat fixed")
    ap.add_argument("--dry-run", action="store_true",
                    help="[llm] print the prompts and use the fixed policy")
    ap.add_argument("--frozen-deck", action="store_true",
                    help="refuse every action that would add, remove or upgrade "
                         "a card, and tell the model why. The premise of the "
                         "2026-09-16 experiment: how far does the starting deck "
                         "go on its own, with the route as the only lever")
    ap.add_argument("--qtable", metavar="PATH",
                    help="play combat from a Q-table trained by rl/stage1_tabular/qlearn_exp1.py "
                         "(implies --frozen-deck: the table's keys assume the "
                         "starting deck). Falls back to the greedy policy on a "
                         "miss, and reports the miss rate at the end")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="[llm] seconds to wait for one API call before giving "
                         "up on it and using the fixed policy for that decision")
    ap.add_argument("--runs", type=int, default=1,
                    help="play this many runs, restarting after each one ends "
                         "cleanly. 0 means keep going until something stops it. "
                         "A wedged run stops the whole thing either way - the "
                         "default of 1 plays one run and exits, which is why a "
                         "death on its own looks like nothing happened.")
    ap.add_argument("--start", choices=("auto", "new", "continue"), default="auto",
                    help="how to get into a run when sitting at the main menu. "
                         "auto resumes a save if there is one and starts a new "
                         "run otherwise, which is also what happens between runs "
                         "- after a death there is nothing left to continue.")
    ap.add_argument("--restart-game", action="store_true",
                    help="when a run ends wedged rather than finished, kill and "
                         "relaunch the game itself, then carry on with the next "
                         "run. Without this a wedged run stops everything.")
    ap.add_argument("--max-game-restarts", type=int, default=3,
                    help="[--restart-game] stop after this many process restarts "
                         "in one session. A wedge is usually reproducible, so an "
                         "uncapped loop rediscovers one bug all night - and pays "
                         "for it under --policy llm. Not reset by a clean run.")
    ap.add_argument("--give-up-after", type=int, default=10,
                    help="how many times in a row the loop will put up with a "
                         "decision it cannot act on, or an action the bridge "
                         "refuses, before calling the run wedged. Counts "
                         "consecutive failures, not polls: waiting on an "
                         "animation does not spend it.")
    ap.add_argument("--character", default="IRONCLAD",
                    help="for --runs restarts (the fixed experimental setting)")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--dump-library", action="store_true",
                    help="refetch data/library.json from the running game and exit; "
                         "do this after a game patch")
    args = ap.parse_args()

    if args.dump_library:
        import urllib.request
        # The endpoint kept its original path; it now returns cards, relics,
        # potions and events, which is why the flag and the file are named
        # for the library rather than for cards.
        url = args.bridge_url.rstrip("/") + "/api/v1/cards"
        with urllib.request.urlopen(url, timeout=60) as resp:
            dump = json.load(resp)
        LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        LIBRARY.write_text(json.dumps(dump, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        print(f"wrote {LIBRARY} from bridge v{dump.get('bridge_version')}: "
              f"{dump.get('counts')}")
        return 0

    log_path = args.log or Path(f"trajectories/{args.policy}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    usage = {"hit": 0, "miss": 0, "out": 0, "calls": 0, "fallbacks": 0}
    model = api_base = None

    # `--qtable` implies `--frozen-deck` rather than warning about it: the table's
    # `hand` component counts copies of exactly three cards, so a run that adds a
    # fourth does not make the table worse, it makes every lookup meaningless.
    frozen = args.frozen_deck or bool(args.qtable)
    if frozen:
        # 🔴 Blocked rather than allowed to wedge. The premise needs a way to
        # decline a combat card reward, and on 2026-09-16 four bridge-side
        # targets were built, deployed and measured, all accepted and all
        # changing nothing: `NRewardsScreen.RewardSkippedFrom`,
        # `CardReward.OnSkipped`, `NRewardButton`'s `RewardSkipped` signal, and
        # the card screen's own 跳过 - which the game itself declares
        # `EndSelectionAndDoNotCompleteReward`. Meanwhile the rewards room will
        # not be left while a reward is outstanding, so there is no walking past
        # it either.
        #
        # Without that, --frozen-deck does not freeze the deck; it stalls on the
        # first card reward and takes ten tries to say so. An error here costs a
        # message, a stall costs a run.
        print("--frozen-deck / --qtable are disabled: declining a combat card "
              "reward has no working path in the bridge.")
        print("  Four methods were tried and measured on 2026-09-16; all were "
              "accepted and changed nothing. See action_adapter.py.")
        print("  Next attempt: dump the live control tree under the rewards "
              "screen and find the button a human presses.")
        print("  For now run without them - the deck grows, which changes the "
              "question from 'how far does the starting deck go' to 'how far "
              "does this agent go'.")
        return 1
    global FROZEN_DECK
    FROZEN_DECK = frozen
    frozen_tally = {"skipped_rewards": 0, "refused": 0, "declined": 0}
    qstats = {"hit": 0, "miss": 0, "reasons": {}}

    # One string, used for both the routing and the prompt, so the two cannot
    # disagree - which they did for a day and a half in 2026-09-14.
    combat_mode = "qtable" if args.qtable else ("fixed" if args.no_combat else args.combat)

    if args.policy == "fixed":
        inner = DEFAULT_CHOOSE
    elif args.policy == "heuristic":
        inner = heuristic_choose
    else:
        import openai
        config = load_config()
        api_base, base_src = resolve(args.api_base, ("LLM_BASE_URL",), config,
                                     "base_url", OFFICIAL_BASE_URL)
        model, model_src = resolve(args.model, ("LLM_MODEL",), config, "model",
                                   "deepseek-flash")
        client = None
        if not args.dry_run:
            # `OPENAI_API_KEY` is deliberately NOT in this chain. It was, and on
            # this machine it happened to hold an unrelated provider's key - which
            # the first test run silently picked up and would have sent to
            # api.deepseek.com. A credential going somewhere it was not issued for
            # is not the kind of convenience worth having.
            api_key, key_src = resolve(args.api_key, ("LLM_API_KEY", "DEEPSEEK_API_KEY"),
                                       config, "api_key")
            if not api_key:
                print("No API key found. Any one of these works:")
                print(f"  - put it in {CONFIG_PATH} (survives closing the terminal)")
                print('  - $env:LLM_API_KEY = "sk-..."   (this window only)')
                print("  - --api-key sk-...              (ends up in shell history)")
                print("  (OPENAI_API_KEY is ignored on purpose - set LLM_API_KEY "
                      "if that is the key you mean)")
                print("  Or --dry-run to see the prompts without calling anything.")
                return 1
            print(f"api key from: {key_src}")          # the source, never the key
            # An explicit timeout, because the SDK's default is 600 seconds
            # with two retries - so one hung request stops an unattended run for
            # up to half an hour, and the game just sits there mid-turn with the
            # loop blocked inside a socket read. Observed 2026-09-13: three
            # minutes of silence with the state idle and five playable cards in
            # hand. 60s is many times the ~9s a healthy call takes, so it only
            # fires on a request that is not coming back.
            client = openai.OpenAI(api_key=api_key, base_url=api_base,
                                   timeout=args.timeout, max_retries=2)
        print(f"api: {api_base} (from {base_src})   model: {model} (from {model_src})")
        inner = make_llm_choose(client, model, combat_mode, args.dry_run,
                                usage, frozen)

    if args.qtable:
        inner = with_qtable_combat(inner, args.qtable, qstats)
        print(f"combat: Q-table {args.qtable}")

    if frozen:
        inner = with_frozen_deck(inner, frozen_tally)
        print("deck: FROZEN (no cards added, removed or upgraded)")

    # Always on, whatever the policy. Declining a dilutive card is ordinary play
    # - `worth_taking` does it - and without this the decision loops forever.
    inner = with_declined_rewards(inner, frozen_tally)

    counts: dict[str, int] = {}
    autoplay.choose = with_logging(inner, log_path, counts)

    game = Sts2RawClient(base_url=args.bridge_url, timeout=10.0)
    print(f"policy: {args.policy}   mode: {'act' if args.act else 'observe'}   "
          f"log: {log_path}")

    # ASCII only. This console is GBK and `⚠️` (U+26A0) is not in it, so an
    # emoji here would raise UnicodeEncodeError before the run even starts -
    # a warning that breaks the thing it is warning about.
    for problem in preflight_bridge():
        print(f"!! BRIDGE IS STALE: {problem}")

    rc = 0
    game_restarts = 0
    # `--runs 0` means "keep going". Written as an unbounded counter rather than
    # a big number so the exit conditions stay the only way out: a wedged run, a
    # refused restart, or Ctrl-C. Those are what should stop an overnight loop,
    # not an arbitrary ceiling nobody chose on purpose.
    forever = args.runs <= 0
    run_numbers = itertools.count(1) if forever else range(1, args.runs + 1)
    total = "inf" if forever else args.runs
    for run_index in run_numbers:
        if forever or args.runs > 1:
            print(chr(10) + f"===== run {run_index}/{total} =====")
        counts.clear()
        # Floors repeat across runs, so a latch kept from the last run would hide
        # a card reward on floor 4 that nobody has looked at yet.
        DECLINED.clear()
        before = dict(usage)

        # Get into a run first. Without this, launching at the main menu made
        # read_loop return "Player is in main_menu" immediately and the whole
        # thing exited having played nothing - the run had to be started by hand
        # every time. Safe when a run is already going: it returns at once.
        if args.act:
            problem = enter_run(game, args.character, args.ascension, args.start)
            if problem:
                print(f"could not get into a run: {problem}")
                return 1

        reason = None
        try:
            reason = autoplay.read_loop(game, args.max_steps, args.act,
                                        give_up_after=args.give_up_after,
                                        seed=args.seed)
            print(f"stopped: {reason}")
        except ApiError as e:
            # The game bridge, not the model. One line rather than forty of
            # traceback: "game not running" is the commonest way to start this
            # wrong and it is not a bug.
            print(f"bridge unreachable at {args.bridge_url}: {e}")
            print("  start the game and tick STS2_Bridge in the launcher popup")
            return 1
        finally:
            print(f"decisions: {counts}")
            if args.policy == "llm":
                report_cost(usage, model, api_base)
            if frozen_tally["declined"]:
                print(f"declined {frozen_tally['declined']} card rewards "
                      f"(latched per floor, never re-claimed)")
            if frozen:
                print(f"frozen deck: skipped {frozen_tally['skipped_rewards']} "
                      f"card rewards, refused {frozen_tally['refused']} actions")
            if args.qtable:
                looked = qstats["hit"] + qstats["miss"]
                # The number this whole line of work exists to produce. Printed
                # even at zero lookups, because "the table was never consulted"
                # and "the table answered everything" are both results and the
                # absence of a line does not distinguish them.
                print(f"qtable: {qstats['hit']}/{looked} lookups answered"
                      + (f" ({100 * qstats['hit'] / looked:.1f}%)" if looked else ""))
                for kind, n in sorted(qstats["reasons"].items(), key=lambda kv: -kv[1]):
                    print(f"   miss x{n}: {kind}")

        # One line per run. Until this existed the only record of how a run ended
        # was a line printed to a terminal that then scrolled away - awkward for
        # a project whose measuring instrument is the win rate. Floor comes from
        # the last state, not from the stop message, because a run can stop for
        # reasons that have nothing to do with dying.
        try:
            last = normalize_state(game.get_state())
        except ApiError:
            last = {}
        ctx = last.get("context") or {}
        player = last.get("player") or {}
        ended = last.get("decision")
        summary = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run": run_index,
            "policy": args.policy,
            "model": model,
            "act": ctx.get("act"),
            "floor": ctx.get("floor"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "ended_on": ended,
            "decisions": dict(counts),
            "stopped": reason,
            "llm_calls": usage["calls"] - before["calls"],
            "fallbacks": usage["fallbacks"] - before["fallbacks"],
            "tokens": {"in_hit": usage["hit"] - before["hit"],
                       "in_miss": usage["miss"] - before["miss"],
                       "out": usage["out"] - before["out"]},
        }
        runs_log = log_path.parent / "runs.jsonl"
        with runs_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + chr(10))
        print(f"run summary -> {runs_log}: act {summary['act']} floor "
              f"{summary['floor']}, hp {summary['hp']}/{summary['max_hp']}, "
              f"ended on {ended!r}")

        if not forever and run_index >= args.runs:
            break

        # Restart only from a clean ending. `game_over` is a death or a win;
        # `menu` means somebody already left. Anything else - a stall, an
        # unhandled screen, eight refusals in a row - is a bug, and restarting
        # into it would spend the night rediscovering the same one.
        if ended not in ("game_over", "menu"):
            if not args.restart_game:
                print(f"not restarting: the run ended on {ended!r}, which is not a "
                      f"finished run. Fix that first - an unattended loop should not "
                      f"drive past a bug.  (--restart-game restarts the process "
                      f"instead, up to --max-game-restarts times)")
                rc = 1
                break

            # Bouncing the process is the one recovery that works when the loop
            # cannot answer the screen it is on: nothing inside the game can be
            # clicked past a wedge the bridge has no branch for.
            #
            # Capped, and the cap is the whole safety argument. A wedge is
            # usually reproducible, so an uncapped restart is a machine that
            # rediscovers one bug all night - and with `--policy llm`, pays per
            # rediscovery. The counter is not reset by a later clean run for the
            # same reason: five dirty endings in a session is a bug worth a human
            # even if good runs happen in between.
            game_restarts += 1
            if game_restarts > args.max_game_restarts:
                print(f"giving up: {game_restarts - 1} game restarts already used "
                      f"and run {run_index} still ended on {ended!r}. This is "
                      f"reproducible - fix it rather than restarting past it.")
                rc = 1
                break
            print(f"run ended on {ended!r} - restarting the game "
                  f"({game_restarts}/{args.max_game_restarts})")
            problem = game_process.restart(game)
            if problem:
                print(f"could not restart the game: {problem}")
                rc = 1
                break
            print("game is back at the main menu")

            # Throw the wedged save away. Without this the recovery walks
            # straight back into the bug it just escaped: `restart()` only kills
            # and relaunches, so the wedged run is still on disk, and `enter_run`
            # with the default `--start auto` resumes a save whenever the menu
            # offers one. Process bounced, same wedge, three restarts spent.
            #
            # Only on this path. A clean `game_over` leaves nothing to continue,
            # and abandoning on a healthy run would delete a run somebody meant
            # to keep - so the discard is tied to the wedge, not to restarting.
            problem = game_process.abandon_saved_run(game)
            if problem:
                print(f"could not abandon the wedged run: {problem}")
                rc = 1
                break
            print("wedged run abandoned")
            # Falls through to enter_run below, which starts from the menu
            # rather than from a game-over screen - a case it already handles.

        problem = enter_run(game, args.character, args.ascension, args.start)
        if problem:
            print(f"could not start run {run_index + 1}: {problem}")
            rc = 1
            break
        print(f"started run {run_index + 1} ({args.character}, ascension {args.ascension})")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
