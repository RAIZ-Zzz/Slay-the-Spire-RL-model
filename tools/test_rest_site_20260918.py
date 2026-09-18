"""Offline checks for the 2026-09-18 rest site fix. No game, no network.

Under `--frozen-deck` the campfire was refused and then *left*: the guard caught
强化, and `with_frozen_deck`'s generic exit walked the run out through `proceed`
and a blind node 0. That exit is right for a card reward, where the screen holds
nothing safe, and wrong for a campfire, where 休息 is safe, sitting on the same
screen, and is also the only way the room gets left at all.

The state here is the one the game actually posts: every rest site in
`trajectories/` is 休息/HEAL and 强化/SMITH, both enabled.

Run: python tools/test_rest_site_20260918.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import play  # noqa: E402

PASS = FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}\n         got  {got!r}\n         want {want!r}")


def rest_site(hp: int, options=None, floor: int = 6) -> dict:
    return {
        "decision": "rest_site",
        "context": {"act": 1, "floor": floor},
        "player": {"hp": hp, "max_hp": 70},
        "options": options if options is not None else [
            {"name": "休息", "id": "HEAL", "is_enabled": True},
            {"name": "强化", "id": "SMITH", "is_enabled": True},
        ],
    }


# `with_frozen_deck` calls `inner(state, grid_picks)`; `choose_rest` is the
# single-argument branch that `heuristic_choose` dispatches to.
def REST_POLICY(state, grid_picks=0):
    return play.choose_rest(state)


def tally() -> dict:
    return {"skipped_rewards": 0, "refused": 0, "declined": 0, "walked_past": 0,
            "rested": 0, "events_rerouted": 0, "deck_changed": 0,
            "exit_tries": {}, "rest_tries": {}}


# --- the policy still prefers smithing at high hp -----------------------------
#
# The fix must not be "stop wanting the upgrade". 60/70 = 0.857, which is not
# below the 0.7 heal threshold, so the unfrozen policy asks for 强化 - and that
# is the input the frozen wrapper has to handle.

HEALTHY = rest_site(60)
payload, why = play.choose_rest(HEALTHY)
check("healthy: policy asks for the upgrade", payload, {"action": "choose_rest_option", "index": 1})
check("...and the guard refuses it", play.deck_change_reason(payload, HEALTHY),
      "the rest site option changes the deck")

HURT = rest_site(40)  # 40/70 = 0.571 < 0.7
payload, why = play.choose_rest(HURT)
check("hurt: policy asks to rest", payload, {"action": "choose_rest_option", "index": 0})
check("...and the guard allows it", play.deck_change_reason(payload, HURT), None)


# --- the wrapper substitutes instead of leaving -------------------------------

t = tally()
frozen = play.with_frozen_deck(REST_POLICY, t)
got, why = frozen(rest_site(60))
check("frozen campfire rests instead of leaving", got, {"action": "choose_rest_option", "index": 0})
check("...not proceed", got.get("action") == "proceed", False)
check("...not a blind map node", got.get("action") == "choose_map_node", False)
check("...counted as a refusal", t["refused"], 1)
check("...counted as a rest", t["rested"], 1)
check("...spent no exit try", t["exit_tries"], {})
check("...says what it did", "resting instead" in why and "REFUSED" in why, True)

# The substitute must itself survive the guard - that is the whole invariant.
check("...substitute is deck-safe", play.deck_change_reason(got, rest_site(60)), None)


# --- a fire with nothing safe on it still leaves ------------------------------
#
# Hypothetical: no rest site has yet shown a removal option. If one ever does,
# and it is the only thing enabled, the exits are correct again.

ONLY_SMITH = rest_site(60, [{"name": "强化", "id": "SMITH", "is_enabled": True}])
sub, note = play.rest_without_deck_change(ONLY_SMITH)
check("no safe option -> no substitute", sub, None)
check("...and names what was there", "强化/SMITH" in note, True)

t = tally()
frozen = play.with_frozen_deck(REST_POLICY, t)
got, why = frozen(ONLY_SMITH)
check("...wrapper falls back to proceed", got, {"action": "proceed"})
check("...and rests nothing", t["rested"], 0)


# --- a removal option is refused, not silently taken --------------------------

REMOVAL = rest_site(60, [
    {"name": "休息", "id": "HEAL", "is_enabled": True},
    {"name": "移除卡牌", "id": "PURGE", "is_enabled": True},
])
check("removal option is a deck change",
      play.deck_change_reason({"action": "choose_rest_option", "index": 1}, REMOVAL),
      "the rest site option changes the deck")
sub, note = play.rest_without_deck_change(REMOVAL)
check("...and the fallback routes around it", sub, {"action": "choose_rest_option", "index": 0})


# --- disabled options do not shift the index ----------------------------------
#
# Every index in play is a position in the *enabled* list. With HEAL and SMITH
# both enabled that is indistinguishable from the raw options order, so this is
# the only case that can tell the two conventions apart.

DISABLED_FIRST = rest_site(60, [
    {"name": "挖掘", "id": "DIG", "is_enabled": False},
    {"name": "休息", "id": "HEAL", "is_enabled": True},
    {"name": "强化", "id": "SMITH", "is_enabled": True},
])
sub, note = play.rest_without_deck_change(DISABLED_FIRST)
check("enabled-order index skips the disabled option", sub,
      {"action": "choose_rest_option", "index": 0})
check("...and it is 休息 that was asked for", "'休息'" in note, True)


# --- the substitution is bounded ----------------------------------------------
#
# If the click never lands the state comes back unchanged. Substituting forever
# would be an infinite loop, and worse if the unproven index mapping in
# `choose_rest`'s docstring is wrong: it would be pressing 强化 each time.

t = tally()
frozen = play.with_frozen_deck(REST_POLICY, t)
actions = [frozen(rest_site(60))[0].get("action") for _ in range(5)]
check("substitution is capped, then the run leaves", actions,
      ["choose_rest_option"] * play.REST_SUBSTITUTION_LIMIT
      + ["proceed", "choose_map_node"])

# A different floor gets its own budget.
t = tally()
frozen = play.with_frozen_deck(REST_POLICY, t)
frozen(rest_site(60, floor=6))
got, why = frozen(rest_site(60, floor=13))
check("a later floor's fire is not charged for the earlier one", got,
      {"action": "choose_rest_option", "index": 0})


# (checkpoint: the campfire half ends here; the run keeps going)


# =============================================================================
# The rewards-screen exit, same day. Reported as "it clicks proceed one extra
# time after the rewards", which turned out to be the visible end of a refusal
# that flickered: `claim_reward` on a card was refused only while `can_proceed`
# was true, and the substituted `proceed` set it false, so the next poll allowed
# the very claim just refused. Every successful frozen run escaped that way.
# =============================================================================

def rewards(items, can_proceed=True, floor=14):
    return {
        "decision": "combat_rewards",
        "context": {"act": 1, "floor": floor},
        "player": {"hp": 60, "max_hp": 70, "open_potion_slots": 0,
                   "potion_slots": 3},
        "items": items,
        "can_proceed": can_proceed,
    }


CARD_AND_GOLD = [{"index": 0, "type": "gold", "gold_amount": 45},
                 {"index": 1, "type": "card"}]

# --- the claim is no longer refused, and no longer flickers -------------------

for cp in (True, False):
    check(f"claiming a card is allowed (can_proceed={cp})",
          play.deck_change_reason({"action": "claim_reward", "index": 1},
                                  rewards(CARD_AND_GOLD, can_proceed=cp)),
          None)

# What actually protects the deck is the verb that adds the card.
check("select_card_reward is still refused",
      play.deck_change_reason({"action": "select_card_reward", "card_index": 0},
                              {"decision": "card_reward"}),
      "it adds a card to the deck")

# The frozen wrapper must pass the claim through untouched - that is the whole
# point of opening the screen.
t = tally()
frozen = play.with_frozen_deck(lambda s, g=0: ({"action": "claim_reward", "index": 1},
                                               "claim [1] card"), t)
got, why = frozen(rewards(CARD_AND_GOLD))
check("frozen wrapper lets the card claim through", got,
      {"action": "claim_reward", "index": 1})
check("...and refuses nothing", t["refused"], 0)


# --- once 跳过 is pressed, the exit is the map, not proceed -------------------

play.DECLINED.clear()
t = tally()
declined = play.with_declined_rewards(
    lambda s, g=0: ({"action": "skip_card_reward", "index": 0}, "skip via alternative[0]"), t)
got, why = declined({"decision": "card_reward", "context": {"act": 1, "floor": 14}})
check("the skip latches the floor", (1, 14) in play.DECLINED, True)

# Nothing but the card left: the first branch.
declined = play.with_declined_rewards(lambda s, g=0: (None, "unused"), t)
got, why = declined(rewards([{"index": 1, "type": "card"}]))
check("card only -> proceed first", got, {"action": "proceed"})
check("...and the map is the second try",
      declined(rewards([{"index": 1, "type": "card"}]))[0],
      {"action": "choose_map_node", "index": 0})

# Floor 15's shape: an unclaimable potion keeps `others` non-empty, so the first
# branch never fires and the veto below has to carry the same exit.
play.DECLINED.clear()
play.DECLINED.add((1, 15))
t = tally()
declined = play.with_declined_rewards(
    lambda s, g=0: ({"action": "claim_reward", "index": 1}, "claim [1] card"), t)
F15 = rewards([{"index": 0, "type": "potion", "potion_name": "爆炸安瓿"},
               {"index": 1, "type": "card"}], floor=15)
check("card + unclaimable potion -> the shared exit",
      declined(F15)[0], {"action": "proceed"})
check("...not conditional on can_proceed",
      declined({**F15, "can_proceed": False})[0],
      {"action": "choose_map_node", "index": 0})
check("...the blind walk is counted", t["walked_past"], 1)

play.DECLINED.clear()



# =============================================================================
# Events were the hole in --frozen-deck: `deck_change_reason` never looked at
# one, and Neow is an event on floor 1. Options below are copied verbatim from
# `trajectories/`, including the one the heuristic actually picked.
# =============================================================================

def event(options, floor=1):
    return {"decision": "event_choice", "context": {"act": 1, "floor": floor},
            "player": {"hp": 80, "max_hp": 80}, "options": options}


# Order copied from the logged run, and the order is the whole mechanism: none
# of these three contains a word from EVENT_GOOD or EVENT_BAD, so all three
# score a flat 0 and `min` takes the earliest. Whether a frozen run's deck
# survived floor 1 came down to which option the game happened to list first.
NEOW = [
    {"title": "卷轴箱", "description": "从2个卡牌包中选择1包加入你的牌组。"},
    {"title": "轰鸣海螺", "description": "在精英战的战斗开始时，额外抽2张牌并获得能量。"},
    {"title": "精准剪刀", "description": "从你的牌组中移除1张牌。"},
]

# The bug, reproduced: this is the pick from heuristic.jsonl at 20:51:37.
payload, why = play.choose_event(event(NEOW))
check("unguarded policy still wants the card pack", payload,
      {"action": "choose_event_option", "index": 0})
check("...and the guard now catches it",
      play.deck_change_reason(payload, event(NEOW)),
      "the event option changes the deck (牌组/加入)")

# 额外抽2张牌 is a draw effect, not a deck change - it must stay takeable, and it
# is usually the only safe thing Neow offers.
check("drawing extra cards is not a deck change",
      play.deck_change_reason({"action": "choose_event_option", "index": 1}, event(NEOW)),
      None)
check("removing a card is a deck change",
      play.deck_change_reason({"action": "choose_event_option", "index": 2}, event(NEOW)),
      "the event option changes the deck (牌组/移除)")

t = tally()
frozen = play.with_frozen_deck(lambda s, g=0: play.choose_event(s), t)
got, why = frozen(event(NEOW))
check("frozen run reroutes to the safe option", got,
      {"action": "choose_event_option", "index": 1})
check("...counted as a reroute", t["events_rerouted"], 1)
check("...and not as a broken deck", t["deck_changed"], 0)
check("...substitute is deck-safe", play.deck_change_reason(got, event(NEOW)), None)

# Locked options are listed by the game but not sendable; indices are positions
# in the unlocked list, so a locked entry in front must not shift them.
LOCKED_FIRST = [{"title": "锁住的", "description": "升级一张牌。", "is_locked": True}] + NEOW
sub, note = play.event_without_deck_change(event(LOCKED_FIRST))
check("locked options do not shift the index", sub,
      {"action": "choose_event_option", "index": 1})
check("...and it is the conch", "轰鸣海螺" in note, True)

# Among safe options the policy's own ranking survives - this only filters.
RANKED = [
    {"title": "受伤的", "description": "失去9点生命。"},
    {"title": "卷轴箱", "description": "从2个卡牌包中选择1包加入你的牌组。"},
    {"title": "营养牡蛎", "description": "获得11点最大生命值。"},
]
sub, note = play.event_without_deck_change(event(RANKED))
check("safe set is still ranked, not taken in order", sub,
      {"action": "choose_event_option", "index": 2})

# Nothing safe: take it, count it, say so. Wedging would end an overnight run at
# the first such event; contaminating silently would waste the whole night.
ALL_BAD = [
    {"title": "涅奥的苦痛", "description": "将1张涅奥之怒加入你的牌组。"},
    {"title": "橙型香盒", "description": "升级一张牌。"},
]
sub, note = play.event_without_deck_change(event(ALL_BAD))
check("no safe event option -> no substitute", sub, None)

t = tally()
frozen = play.with_frozen_deck(lambda s, g=0: play.choose_event(s), t)
import io as _io, contextlib as _ctx
_buf = _io.StringIO()
with _ctx.redirect_stdout(_buf):
    got, why = frozen(event(ALL_BAD))
check("...the run continues rather than wedging", got is not None, True)
check("...counted as a broken deck", t["deck_changed"], 1)
check("...and shouted about", "DECK CHANGED" in _buf.getvalue(), True)
check("...with BROKEN in the logged reason", "BROKEN" in why, True)


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
