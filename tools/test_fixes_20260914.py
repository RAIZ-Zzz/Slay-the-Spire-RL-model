"""Offline checks for the 2026-09-14 fixes. No game, no network.

Three bugs and one lie, each exercised against a hand-written state rather than
a live run, because every one of them is a case the real game shows rarely:
Pael's Wing has to be on the run before a second alternative exists at all, and
the unhandled overlay is one screen out of fourteen.

Run: python tools/test_fixes_20260914.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autoplay  # noqa: E402
import play  # noqa: E402
from state_brief import brief  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-harness"))
from cli_anything.slay_the_spire_ii.core import action_adapter, state_adapter  # noqa: E402

PASS = FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}\n         got  {got!r}\n         want {want!r}")


# --- fixtures ----------------------------------------------------------------

def card_reward_raw(alternatives, cards=()):
    """What the bridge posts for a card reward screen."""
    return {
        "state_type": "card_reward",
        "context": {"act": 1, "floor": 19},
        "card_reward": {
            "cards": list(cards),
            "alternatives": alternatives,
            "can_skip": bool(alternatives),
        },
    }


# The row seen with PAELS_WING on the run: sacrifice sits at position 0, which
# is exactly the index the old handler clicked unconditionally.
# `option_id` is the discriminator, and it is the third thing tried. Two live runs
# on 2026-09-14 killed the first two: the button's own name field holds the
# localised label ("跳过", so `"SKIP" in ...` cannot fire in a Chinese client), and
# every alternative including the plain skip turns out to be in `_extraOptions`,
# so "is it an extra" separated nothing. `OptionId == "Skip"` is measured, from a
# real card reward screen.
SACRIFICE_FIRST = [
    {"index": 0, "option_id": "PaelsWingSacrifice", "title": "献祭", "is_enabled": True},
    {"index": 1, "option_id": "Skip", "title": "跳过", "is_enabled": True},
]


# --- 1. the alternative row survives normalisation ----------------------------

print("1. card reward alternatives")

s = state_adapter.normalize_state(card_reward_raw(SACRIFICE_FIRST))
check("normalize keeps every alternative", len(s["alternatives"]), 2)
check("normalize keeps the ids", [a["option_id"] for a in s["alternatives"]],
      ["PaelsWingSacrifice", "Skip"])

# An older bridge sends no `alternatives` key at all. None and [] have to stay
# distinguishable: "cannot tell" is not "the screen offers none".
old = state_adapter.normalize_state(
    {"state_type": "card_reward",
     "context": {},
     "card_reward": {"cards": [], "can_skip": True}})
check("old bridge -> None, not []", old["alternatives"], None)

check("brief shows the row", [a["title"] for a in brief(s)["alternatives"]],
      ["献祭", "跳过"])


# --- 2. the fixed policy no longer clicks position 0 blind --------------------

print("2. choose() picks by id, not by position")

payload, why = autoplay.choose(s)
check("skips rather than sacrifices", payload,
      {"action": "skip_card_reward", "index": 1})
check("names what it clicked", "跳过" in why, True)
check("...matched on the stable id", "option_id is Skip" in why, True)

# Only a sacrifice on offer: there is no skip to prefer, so it takes what there
# is - but the reason has to say so, because "skip card reward" would be a lie
# of exactly the kind the bridge used to tell.
only_sac = state_adapter.normalize_state(card_reward_raw([SACRIFICE_FIRST[0]]))
payload, why = autoplay.choose(only_sac)
check("only a sacrifice -> takes index 0", payload,
      {"action": "skip_card_reward", "index": 0})
check("...and says it is picking blind", "NO Skip id" in why, True)

# A disabled alternative is not an option. `can_skip` could not express this.
disabled = state_adapter.normalize_state(card_reward_raw(
    [{"index": 0, "option_id": "Skip", "title": "跳过", "is_enabled": False}]))
payload, why = autoplay.choose(disabled)
check("all disabled -> no action", payload, None)

# Old bridge: index 0 is all there is, and the reason admits it is a guess.
payload, why = autoplay.choose(old)
check("old bridge still works", payload["action"], "skip_card_reward")
check("...and calls index 0 a guess", "guess" in why, True)

# Taking a card is unchanged by any of this.
with_cards = state_adapter.normalize_state(card_reward_raw(
    SACRIFICE_FIRST, cards=[{"index": 0, "name": "打击", "rarity": "Common"}]))
payload, _ = autoplay.choose(with_cards)
check("a card on offer still wins", payload["action"], "select_card_reward")

check("legal_verbs offers the alternative", play.legal_verbs(s),
      ["select_card_reward", "skip_card_reward"] if s.get("cards")
      else ["skip_card_reward"])
check("legal_verbs drops it when disabled", play.legal_verbs(disabled), [])

check("action_adapter carries the index",
      action_adapter.skip_card_reward(1),
      {"action": "skip_card_reward", "index": 1})
check("action_adapter default is still 0",
      action_adapter.skip_card_reward(),
      {"action": "skip_card_reward", "index": 0})


# --- 3. an unhandled overlay names itself and can be left ---------------------

print("3. unhandled overlay")

def overlay_raw(can_proceed):
    return {
        "state_type": "overlay",
        "context": {"act": 1, "floor": 1},
        "overlay": {
            "screen_type": "NCrystalSphereScreen",
            "can_proceed": can_proceed,
            "message": "...",
        },
    }


ov = state_adapter.normalize_state(overlay_raw(True))
check("screen_type reaches the top level", ov["screen_type"], "NCrystalSphereScreen")
# The whole point: on 2026-09-14 this field existed in the bridge's reply and
# never reached the log, so the run that died on floor 1 is unattributable.
check("brief records the class name", brief(ov)["screen_type"], "NCrystalSphereScreen")
check("legal_verbs offers the exit", play.legal_verbs(ov), ["proceed"])

payload, why = autoplay.choose(ov)
check("choose leaves it", payload, {"action": "proceed"})
check("...and warns it may forfeit", "forfeit" in why, True)

stuck = state_adapter.normalize_state(overlay_raw(False))
check("no proceed -> no verb", play.legal_verbs(stuck), [])
payload, why = autoplay.choose(stuck)
check("choose stops loudly", payload, None)
check("...naming the screen", "NCrystalSphereScreen" in why, True)


# --- 4. the system prompt matches who is actually playing combat --------------

print("4. system prompt")

# `build_system` took a bool until 2026-09-16, when combat grew a third and
# fourth player (the greedy policy and a trained Q-table) and the argument had to
# become the mode itself. The point of the test is unchanged and is the one that
# matters: whatever is actually playing combat is what the prompt says.
on, off = play.build_system("llm"), play.build_system("fixed")
check("combat-on says it is yours", "Combat is yours to play" in on, True)
check("combat-on drops the fixed-rule claim",
      "Combat is played by a fixed rule" in on, False)
check("combat-off keeps it", "Combat is played by a fixed rule" in off, True)
check("no placeholder left", "{{COMBAT}}" in on or "{{COMBAT}}" in off, False)
check("both mention the alternative index",
      ("`skip_card_reward` takes the `index`" in on
       and "`skip_card_reward` takes the `index`" in off), True)

# Each mode describes itself, and none of them describes another. This is the
# 2026-09-14 bug generalised: then there were two modes and one was mislabelled.
MODE_TAG = {
    "llm": "Combat is yours to play",
    "greedy": "It focus-fires and it reads intents",
    "fixed": "Combat is played by a fixed rule",
    "qtable": "played for you by a Q-table",
}
for mode, tag in MODE_TAG.items():
    text = play.build_system(mode)
    check(f"{mode} prompt describes {mode}", tag in text, True)
    others = [t for m, t in MODE_TAG.items() if m != mode and t not in tag]
    check(f"{mode} prompt describes nothing else",
          any(t in text for t in others), False)

# The frozen-deck variant has to *replace* the deck-shaping advice, not sit next
# to it: telling the model to weigh dilution while refusing every card it could
# take is how a prompt starts arguing with the program.
frozen = play.build_system("greedy", frozen=True)
check("frozen says the deck never changes", "The deck never changes" in frozen, True)
check("frozen drops the take-a-card advice",
      "A card reward can be skipped" in frozen, False)
check("frozen drops the card-removal advice",
      "category: card_removal" in frozen, False)
check("open keeps the take-a-card advice",
      "A card reward can be skipped" in play.build_system("greedy"), True)
check("no placeholder left in frozen", "{{" in frozen, False)



# --- 5. potion slots: full is no longer the end of the story ------------------

print("5. combat rewards and the potion swap")

def rewards_raw(items, open_slots, *, can_proceed=True, held=(), can_discard=True):
    return {
        "state_type": "combat_rewards",
        "context": {"act": 1, "floor": 9},
        "held_potions": list(held),
        "can_discard_potions": can_discard,
        "rewards": {
            "items": list(items),
            "can_proceed": can_proceed,
            "player": {"open_potion_slots": open_slots, "potion_slots": 3},
        },
    }


GOLD = {"index": 0, "type": "gold", "gold_amount": 10}
POTION = {"index": 0, "type": "potion", "potion_name": "能量药水"}
HELD = [{"slot": 0, "name": "灰水", "description": "..."},
        {"slot": 2, "name": "火焰药水", "description": "..."}]

# Slots free: unchanged, claim it.
free = state_adapter.normalize_state(rewards_raw([POTION], 1, held=HELD))
check("slot free -> claim", play.legal_verbs(free), ["claim_reward"])

# Slots full and the only item is that potion. Before today this returned
# ["claim_reward"] - a verb that returns status:ok and does nothing.
full = state_adapter.normalize_state(rewards_raw([POTION], 0, held=HELD))
check("slots full -> not claim_reward", "claim_reward" in play.legal_verbs(full), False)
check("slots full -> swap or leave", play.legal_verbs(full), ["discard_potion", "proceed"])
check("brief names the swap", "discard_potion(slot=" in brief(full)["send"], True)

# Something else is still claimable: the swap must not tempt the model away from
# a card or gold it can simply take.
mixed = state_adapter.normalize_state(rewards_raw([GOLD, POTION], 0, held=HELD))
check("gold still claimable -> claim only", play.legal_verbs(mixed), ["claim_reward"])

# Each precondition on its own removes the verb, because each is a separate way
# for the bridge to refuse.
no_discard = state_adapter.normalize_state(
    rewards_raw([POTION], 0, held=HELD, can_discard=False))
check("CanRemovePotions false -> no swap", play.legal_verbs(no_discard), ["proceed"])
nothing_held = state_adapter.normalize_state(rewards_raw([POTION], 0, held=[]))
check("nothing held -> no swap", play.legal_verbs(nothing_held), ["proceed"])

# An older bridge sends no `can_discard_potions`; "cannot tell" must not read as
# yes, or every attempt is refused.
old_bridge = state_adapter.normalize_state(
    {"state_type": "combat_rewards", "context": {},
     "held_potions": HELD,
     "rewards": {"items": [POTION], "can_proceed": True,
                 "player": {"open_potion_slots": 0, "potion_slots": 3}}})
check("old bridge -> field is None", old_bridge["can_discard_potions"], None)
check("old bridge -> no swap offered", play.legal_verbs(old_bridge), ["proceed"])

# The fixed policy still declines to choose which potion to throw away.
payload, why = autoplay.choose(full)
check("fixed policy leaves it", payload, {"action": "proceed"})
check("...and says the verb exists", "discard_potion could" in why, True)

check("action_adapter shape", action_adapter.discard_potion(2),
      {"action": "discard_potion", "slot": 2})


# --- 6. restarting the game process ------------------------------------------

print("6. game process")

import game_process  # noqa: E402

ok, why = game_process.mods_enabled()
check("STS2_Bridge is enabled in settings.save", ok, True)
check("...and says where it read that", "settings.save" in why, True)
check("the exe is where GAME_DIR says", game_process.GAME_EXE.exists(), True)


# --- 7. "pick exactly N cards" screens ---------------------------------------

print("7. card_select with an exact count")


def card_select_raw(n_cards, selected, want, *, can_confirm=False, preview=False):
    return {
        "state_type": "card_select",
        "context": {"act": 1, "floor": 12},
        "card_select": {
            "screen_type": "select",
            "prompt": "选择2张牌来移除。",
            "cards": [{"index": i, "name": f"card{i}"} for i in range(n_cards)],
            "selected_count": selected,
            "min_select": want,
            "max_select": want,
            "can_confirm": can_confirm,
            "preview_showing": preview,
        },
    }


# The live case, twice on 2026-09-14: 14 cards on offer, 2 wanted. The old rule
# picked all fourteen looking for `can_confirm` and then gave up the run.
s0 = state_adapter.normalize_state(card_select_raw(14, 0, 2))
payload, why = autoplay.choose(s0)
check("0/2 -> pick the first", payload, {"action": "select_card", "index": 0})
check("...and says the target", "1/2" in why, True)

s1 = state_adapter.normalize_state(card_select_raw(14, 1, 2))
payload, _ = autoplay.choose(s1)
check("1/2 -> pick the second", payload, {"action": "select_card", "index": 1})

# Enough picked, confirm still dark: wait for the screen rather than pick a
# third, which is over the limit and would toggle something back off.
s2 = state_adapter.normalize_state(card_select_raw(14, 2, 2))
payload, why = autoplay.choose(s2)
check("2/2 with no confirm -> WAIT", payload is autoplay.WAIT, True)
check("...rather than overshooting", "2/2 picked" in why, True)

ready = state_adapter.normalize_state(card_select_raw(14, 2, 2, can_confirm=True))
payload, _ = autoplay.choose(ready)
check("confirm available -> confirm", payload, {"action": "confirm_selection"})

# Fewer cards than the screen demands: stop and name it instead of looping.
short = state_adapter.normalize_state(card_select_raw(1, 1, 2))
payload, why = autoplay.choose(short)
check("impossible screen -> stop", payload, None)
check("...naming the mismatch", "cannot be satisfied" in why, True)

# An older bridge sends no counts, so the previous index-walking rule still runs
# - "cannot say" must not be read as "wants none".
old_cs = state_adapter.normalize_state(
    {"state_type": "card_select", "context": {},
     "card_select": {"screen_type": "select",
                     "cards": [{"index": 0, "name": "a"}, {"index": 1, "name": "b"}]}})
check("old bridge -> counts are None", old_cs["selected_count"], None)
payload, _ = autoplay.choose(old_cs)
check("...and it still picks", payload, {"action": "select_card", "index": 0})



# --- 8. the frozen deck leaves a card reward behind ---------------------------
#
# The first live run of --frozen-deck cycled combat_rewards -> card_reward
# -> combat_rewards for as long as it was left alone. Three things had to be
# true at once and each is checked below:
#
#   * `legal_verbs` offered `claim_reward` as the *only* verb, because a card
#     entry counted as claimable. The model said so in its own reason.
#   * claiming opens the card screen; the guard skips it; **skipping does not
#     consume the entry**, so the screen comes back unchanged.
#   * `read_loop`'s stuck detector keys on (floor, decision), and a cycle
#     between two decisions never repeats either one.
#
# The state below is not invented - it was read off the live game while the run
# was wedged, `can_proceed` and all.

print("8. frozen deck: a card reward is not claimable")

LIVE_WEDGED = {
    "decision": "combat_rewards",
    "items": [{"index": 0, "type": "card",
               "description": "把一张卡牌添加到你的牌组。"}],
    "is_complete": False,
    "can_proceed": True,
    "player": {"open_potion_slots": 2},
}

tally = {"skipped_rewards": 0, "refused": 0}


def frozen(inner_payload, inner_reason="policy wanted this"):
    return play.with_frozen_deck(lambda s, g=0: (inner_payload, inner_reason), tally)


# --frozen-deck is disabled, and the assertion is that it stays that way until
# the bridge can actually decline a card reward. Four attempts on 2026-09-16 were
# each accepted and changed nothing; meanwhile the rewards room cannot be left
# with a reward outstanding, so there is no way around it either. A flag that
# claims to freeze the deck and instead stalls on floor 2 is worse than no flag.
check("no skip_reward in the adapter",
      hasattr(action_adapter, "skip_reward"), False)
check("skip_card_reward survives (it is still how alternatives are clicked)",
      hasattr(action_adapter, "skip_card_reward"), True)

# The deck-changing refusals themselves are unchanged and still correct - they
# are what --frozen-deck would use once it works.
check("adding a card is refused",
      play.deck_change_reason(action_adapter.select_card_reward(0),
                              {"decision": "card_reward"}) is not None, True)
check("buying a card is refused",
      play.deck_change_reason(action_adapter.shop_purchase(0),
                              {"decision": "shop",
                               "items": [{"index": 0, "category": "card"}]}) is not None,
      True)
check("buying a potion is not",
      play.deck_change_reason(action_adapter.shop_purchase(0),
                              {"decision": "shop",
                               "items": [{"index": 0, "category": "potion"}]}), None)
check("a rest-site upgrade is refused",
      play.deck_change_reason(action_adapter.choose_rest_option(1),
                              {"decision": "rest_site",
                               "options": [{"name": "休息", "is_enabled": True},
                                           {"name": "强化", "is_enabled": True}]}) is not None,
      True)

# `legal_verbs` must stay free of frozen-deck clauses. Two versions of one were
# tried and both made things worse: hiding the card entry left `proceed` as the
# only verb, and `proceed` does not work while a reward is outstanding.
play.FROZEN_DECK = True
check("legal_verbs ignores the flag", play.legal_verbs(LIVE_WEDGED), ["claim_reward"])
play.FROZEN_DECK = False

# And the skip itself: the only stable handle is the game's own OptionId, because
# `_optionName` came back localised as 「跳过」 on 2026-09-14.
reward = {"decision": "card_reward", "cards": [{"index": 0, "name": "铁斩波"}],
          "alternatives": [{"index": 0, "option_id": "Sacrifice", "title": "献祭",
                            "is_enabled": True},
                           {"index": 1, "option_id": "Skip", "title": "跳过",
                            "is_enabled": True}]}
payload, reason = play.skip_this_reward(reward)
check("skip picks the Skip option, not index 0", payload,
      {"action": "skip_card_reward", "index": 1})
no_skip = dict(reward, alternatives=[{"index": 0, "option_id": "Reroll",
                                      "title": "换一组", "is_enabled": True}])
payload, reason = play.skip_this_reward(no_skip)
check("no Skip -> stops loudly rather than clicking Reroll", payload, None)
check("...and names what was there", "Reroll" in reason, True)

play.FROZEN_DECK = False

# The player's hp is the one quantity the adapter still converts, because
# `encode_coarse` divides by a module constant of 80 while the real max hp grows
# during a run - 91 by floor 2 on the first live run. Damage stays absolute.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))
import exp1_combat as exp1  # noqa: E402
import qtable_combat  # noqa: E402

def toy_hp(hp, max_hp):
    st = {"player": {"hp": hp, "max_hp": max_hp, "block": 0}, "energy": 3, "round": 1,
          "enemies": [{"entity_id": "E0", "hp": 30, "max_hp": 30,
                       "intents": [{"type": "Attack", "label": "12"}]}],
          "hand": []}
    toy, _ = qtable_combat.to_toy_state(st)
    return min(4, toy.player_hp * 5 // exp1.PLAYER_HP)

check("50/120 is 42%, so bucket 2", toy_hp(50, 120), 2)
check("...and the unscaled version got it wrong", min(4, 50 * 5 // exp1.PLAYER_HP), 3)
check("87/91 is 96%, so bucket 4", toy_hp(87, 91), 4)
check("40/80 is 50%, so bucket 2", toy_hp(40, 80), 2)
check("full hp clamps into the top bucket", toy_hp(80, 80), 4)


# --- 9. a declined card reward is remembered, not re-clicked -------------------
#
# The user's fix, and the right one: stop trying to make the game agree that a
# reward is declined, and just remember not to click it again. Seven bridge-side
# attempts on 2026-09-16 were each accepted and changed nothing - the card
# screen's own skip is declared `EndSelectionAndDoNotCompleteReward`, so the
# reward stays outstanding by design. What was missing was any memory of the
# decision, so the next poll saw a claimable card and clicked it again, forever.
#
# Not a --frozen-deck feature: `worth_taking` declines dilutive cards in ordinary
# play, and that decision led into the same loop.

print("9. declined card rewards")

CTX = {"act": 1, "floor": 6}
dtally = {"skipped_rewards": 0, "refused": 0, "declined": 0}


def _policy(state, g=0):
    """Worst case: skips on the card screen, then tries to claim it again."""
    if state.get("decision") == "card_reward":
        return action_adapter.skip_card_reward(1), "worth_taking says skip"
    return action_adapter.claim_reward(0), "claim [0] card"


wrapped = play.with_declined_rewards(_policy, dtally)
CARD_ONLY = {"decision": "combat_rewards", "context": CTX, "can_proceed": True,
             "items": [{"index": 0, "type": "card"}],
             "player": {"open_potion_slots": 3}}
REWARD = {"decision": "card_reward", "context": CTX, "cards": [{"index": 0}],
          "alternatives": [{"index": 1, "option_id": "Skip", "title": "skip",
                            "is_enabled": True}]}

play.DECLINED.clear()
# Before any decision the card MUST stay claimable: claiming it is the only way
# to see which three cards are offered, so hiding it earlier would decline every
# card reward sight unseen.
check("undecided -> the card is claimable", play.legal_verbs(CARD_ONLY), ["claim_reward"])

payload, why = wrapped(REWARD)
check("skipping on the card screen still happens", payload,
      {"action": "skip_card_reward", "index": 1})
check("...and latches the floor", (1, 6) in play.DECLINED, True)
check("...and says so in the reason", "declined floor 6" in why, True)

check("declined -> the card is no longer claimable",
      play.legal_verbs(CARD_ONLY), ["proceed"])
check("declined -> leave instead of re-claiming",
      wrapped(CARD_ONLY)[0], {"action": "proceed"})

MIXED = {"decision": "combat_rewards", "context": CTX, "can_proceed": True,
         "items": [{"index": 0, "type": "card"}, {"index": 1, "type": "gold"}],
         "player": {"open_potion_slots": 3}}
check("gold keeps claim_reward on offer", play.legal_verbs(MIXED), ["claim_reward"])
check("claiming the gold passes through",
      play.with_declined_rewards(
          lambda st, g=0: (action_adapter.claim_reward(1), "gold"), dtally)(MIXED)[0],
      {"action": "claim_reward", "index": 1})
check("claiming the declined card is vetoed", wrapped(MIXED)[0], {"action": "proceed"})

OTHER = dict(CARD_ONLY, context={"act": 1, "floor": 7})
check("another floor is unaffected", play.legal_verbs(OTHER), ["claim_reward"])
check("...and is claimed normally", wrapped(OTHER)[0],
      {"action": "claim_reward", "index": 0})

# The one thing still unmeasured: whether the room lets go with a card entry
# outstanding. If it does not, this returns None and read_loop stops loudly on a
# single repeated decision - which is the answer, not a bug.
NOPE = dict(CARD_ONLY, can_proceed=False)
payload, why = wrapped(NOPE)
check("no proceed -> stop loudly, never loop", payload, None)
check("...and names the consequence", "cannot be left" in why, True)
check("declined count is reported", dtally["declined"], 1)
play.DECLINED.clear()

# --- 10. what the bridge reports about a shop reaches the decision ------------
#
# The agent never bought anything, and the cause was not policy: `state_brief`'s
# whitelist dropped `card_description`, `relic_description` and
# `potion_description`, all three of which `BuildShopState` fills in. So a shop
# decision read "spend 150 gold on 燃烧之血, yes or no" with no way to find out
# what 燃烧之血 does, and declining was the only reasonable answer.
#
# Same family as the 2026-09-14 `screen_type` bug: a field the bridge reports
# that never reaches the thing making the decision. Asserted here because the
# whitelist and the builder are in different languages and nothing else keeps
# them in step.

print("10. shop: descriptions survive into the brief")

SHOP_RAW = {
    "state_type": "shop", "context": {"act": 1, "floor": 14},
    "shop": {
        "player": {"hp": 60, "max_hp": 80, "gold": 310}, "can_proceed": False,
        "items": [
            {"index": 0, "category": "card", "cost": 120, "is_stocked": True,
             "can_afford": True, "on_sale": False, "card_id": "CLEAVE",
             "card_name": "横扫", "card_type": "Attack", "card_rarity": "Common",
             "card_description": "对所有敌人造成8点伤害。"},
            {"index": 1, "category": "relic", "cost": 150, "is_stocked": True,
             "can_afford": True, "relic_id": "BURNING_BLOOD",
             "relic_name": "燃烧之血",
             "relic_description": "在战斗结束时，回复6点生命。"},
            {"index": 2, "category": "potion", "cost": 80, "is_stocked": True,
             "can_afford": True, "potion_id": "REGEN_POTION",
             "potion_name": "再生药水", "potion_description": "获得5层再生。"},
            {"index": 3, "category": "card_removal", "cost": 75,
             "is_stocked": True, "can_afford": True},
        ],
    },
}
shop = state_adapter.normalize_state(SHOP_RAW)
opts = brief(shop)["options"]
by_cat = {o["category"]: o for o in opts}

check("every entry survives", len(opts), 4)
for cat, key in (("card", "card_description"),
                 ("relic", "relic_description"),
                 ("potion", "potion_description")):
    check(f"{cat} keeps {key}", key in by_cat[cat], True)
check("relic description is the text, not the name",
      by_cat["relic"]["relic_description"], "在战斗结束时，回复6点生命。")
check("cost and affordability survive",
      (by_cat["relic"]["cost"], by_cat["relic"]["can_afford"]), (150, True))
check("shop_purchase is offered when something is affordable",
      play.legal_verbs(shop), ["shop_purchase", "proceed"])

# Nothing affordable -> no point offering the verb.
broke = {**SHOP_RAW, "shop": {**SHOP_RAW["shop"],
         "items": [{**i, "can_afford": False} for i in SHOP_RAW["shop"]["items"]]}}
check("nothing affordable -> proceed only",
      play.legal_verbs(state_adapter.normalize_state(broke)), ["proceed"])

# The prompt has to say what the program can actually do with a purchase. Nothing
# in this agent drinks a potion - `autoplay.pick_potion` returns None on purpose -
# so buying one is gold thrown away, and the model has to be told that.
check("pick_potion still declines", autoplay.pick_potion(shop), None)
sysmsg = play.build_system("greedy")
check("prompt recommends relics", "A relic is the best thing to spend gold on" in sysmsg, True)

# ⚠️ Whether a potion is worth buying depends on who plays combat, and a flat
# "never buy potions" would be a lie in the one mode where the model drinks them.
# `use_potion` is in legal_verbs for combat_play and the potions are in the brief
# with descriptions and target_type - but only the model ever calls it.
check("greedy mode forbids potions", "Do **not** buy potions" in sysmsg, True)
check("...and does not also recommend them",
      "Potions are worth buying" in sysmsg, False)
llmmsg = play.build_system("llm")
check("llm mode allows potions", "Potions are worth buying" in llmmsg, True)
check("...and does not also forbid them",
      "Do **not** buy potions" in llmmsg, False)
for mode in ("llm", "greedy", "fixed", "qtable"):
    txt = play.build_system(mode)
    check(f"{mode}: exactly one potion paragraph",
          ("Potions are worth buying" in txt) != ("Do **not** buy potions" in txt), True)
    check(f"{mode}: no placeholder left", "{{" in txt, False)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
