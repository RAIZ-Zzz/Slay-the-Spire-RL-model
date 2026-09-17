"""Drive a whole run automatically: read state, choose an action, send it, repeat.

This is the piece every later route needs. Re-running the baseline needs it,
because 39 minutes of hand-play per run does not scale. Collecting combat
trajectories needs it, because save files record meta decisions only. An LLM
agent is this loop with `choose` swapped for a model call.

And the random policy buys something the 392k community runs cannot: **unbiased
assignment**. A human never picks cards at random, so every human choice carries
their skill and their board state - which is why `card_value()` in
analyze_runs.py measured "how weak is the player who took this card" instead of
"how good is this card". A coin flip carries neither. Random play is not a
placeholder here; it is the only way to get a causal estimate.

Build it in levels. Each one runs against the real game before the next starts.

  A1  read loop         Poll `get_state()`, normalise, print decision + floor,
                        stop on game_over or menu. Send nothing. This alone
                        catches the state shapes the schemas never captured.
  A2  one action        Send exactly one kind of action - `choose_map_node` is
                        the easiest - and watch the state change. Proves the
                        write path inside a loop, which is different from
                        proving it once by hand.
  A3  all decisions     Cover the rest. 15 decisions exist; 8 have captured
                        schemas in `schemas/`, the other 6 you will meet live.
  A4  trajectory log    One JSONL line per decision: state, options, action.
                        The options matter as much as the choice - that is the
                        gap that makes save files useless for preference models.
  A5  swappable policy  Move `choose` behind an object so a greedy or LLM policy
                        drops in without touching the loop.

Two things learned the hard way that are worth knowing before you start:

  * `decision == "unknown"` is a **transient**, not an error. The bridge reports
    it while a room loads, with no legal actions attached. Sleep and re-poll;
    do not treat it as a failure and do not pick an action.
  * Actions need a gap between them (~0.5s). The game animates, and the state
    you read mid-animation may not be the one your action assumed.

"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

from cli_anything.slay_the_spire_ii.core import action_adapter
from cli_anything.slay_the_spire_ii.core.state_adapter import normalize_state
from cli_anything.slay_the_spire_ii.utils.sts2_backend import ApiError, Sts2RawClient

# --- A3 step 6: randomised choices ---------------------------------------------
#
# A fixed `index=0` is not a neutral default - it is one specific route walked
# over and over, so a hundred runs are one sample repeated. The value this agent
# has that a human does not is the ability to *randomise the treatment*: pick
# cards and routes by dice and the result is a clean causal estimate rather than
# the selection bias the 2026-09-08 card table ran into.
#
# A named `Random` rather than the `random` module, because the module's global
# stream is shared with anything else that imports `random` - one library call
# and the sequence shifts, so "step 37 did something stupid" stops reproducing.
#
# ⚠️ This is module state, and module state is what A5 exists to remove: the
# policy should carry its own rng. It is here rather than in the signature
# because `(state, grid_picks) -> (payload, reason)` is a contract with six
# implementations across two files, and widening it during a step whose whole
# point is "change one thing" is the wrong trade. `choose` takes an explicit
# `rng=` so A5 can hand one in without this ever being read.
RNG = random.Random()


def seed_choices(seed: int | None = None) -> int:
    """Seed the chooser. Returns the seed actually used - print it or lose it.

    `None` draws a fresh seed from the OS rather than leaving the rng unseeded,
    so that every run is both different *and* replayable. An unseeded rng gives
    the first of those and not the second, which is the worse half.
    """
    if seed is None:
        seed = random.SystemRandom().randrange(2 ** 31)
    RNG.seed(seed)
    return seed

# --- A1 -----------------------------------------------------------------------


def wait_key(state) -> tuple:
    """Everything a wait could be waiting on, as one comparable value.

    Used to tell "the game is slow" from "the game is stuck". If any of these
    moves, something is happening and waiting longer is correct no matter how
    long it takes. If none of them moves, no amount of further waiting will help.
    """
    return (
        state.get("decision"),
        state.get("round"),
        state.get("turn"),
        state.get("is_resolving"),
        state.get("energy"),
        len(state.get("hand") or []),
        state.get("draw_pile_count"),
        state.get("discard_pile_count"),
        tuple((e.get("entity_id"), e.get("hp")) for e in (state.get("enemies") or [])),
    )


def read_loop(
    client,
    max_steps: int = 50,
    act: bool = False,
    stall_polls: int = 400,
    poll: float = 0.3,
    give_up_after: int = 8,
    seed: int | None = None,
) -> str:
    """Poll and print until the run ends. Returns why it stopped.

    `max_steps` counts **decisions**, not polls. That distinction is the whole
    reason this is a while loop: waiting for an animation used to consume the
    budget, so on 2026-09-12 the user pointed out that a windowed game left
    unfocused drops into a low-power mode at a much lower frame rate - every
    animation takes several times longer in wall-clock, and a poll-counting
    budget would drain on waiting without ever reaching a decision. Exit reasons
    have to describe what the loop ran out of, and "ran out of patience while the
    game played an animation" is not the same event as "made 50 decisions".

    Nothing here decides anything from elapsed time. `poll` is only how often to
    look, and `stall_polls` only bounds waiting on a state that is **not
    changing** - a genuine hang, not a slow animation. Low frame rates make
    transients last *longer*, which makes them easier to observe, not harder; it
    is timeouts that a frame rate breaks, and there are none left in the guards.

    With `act=False` the chooser still runs and prints what it would do, but
    nothing is sent - so a new branch can be checked against the live game
    before it is allowed to touch it.
    """
    # Printed, not just used. A run whose seed was never written down is a run
    # that cannot be replayed, which is most of what seeding was for.
    print(f"  seed {seed_choices(seed)}")

    # The round our last combat action went out in. The only piece of history the
    # loop keeps, and it exists to tell an undealt hand apart from an emptied one
    # - see guard 3. None means "no action yet this combat", which correctly makes
    # the first hand of a fight wait to be dealt instead of being ended on sight.
    last_acted_round = None

    # Cards already picked on the card_select grid in front of us. Reset on
    # leaving that screen, for the same reason last_acted_round resets on
    # leaving combat: a stale count would skip the first card of the next one.
    grid_picks = 0

    step = 0          # decisions made, the budget `max_steps` applies to
    polls = 0         # reads performed, for the log only
    unchanged = 0     # consecutive waits with an identical wait_key
    prev_key = None
    cannot_act = 0    # consecutive "no branch / bridge cannot see it" answers
    refused = 0       # consecutive actions the bridge answered with status:error

    def hold(label: str, state, sleep_for: float) -> str | None:
        """Wait one poll. Returns a stop reason only if the state is frozen."""
        nonlocal unchanged, prev_key
        key = wait_key(state)
        unchanged = unchanged + 1 if key == prev_key else 0
        prev_key = key
        print(f"  [{step:3}/{polls:4}] {label}")
        if unchanged >= stall_polls:
            return (
                f"stalled: {unchanged} polls with no change at all while waiting "
                f"({label}). not slowness - nothing in the state moved."
            )
        time.sleep(sleep_for)
        return None

    while step < max_steps:
        raw = client.get_state()
        state = normalize_state(raw)
        polls += 1
        decision = state.get("decision")

        if decision == "game_over":
            return "Player's character is dead, game over"
        if decision == "menu":
            return f"Player is in {state.get('screen')}"

        # `unknown` means the room is still loading: no legal actions exist yet,
        # so waiting is the only correct response. Treating it as a failure
        # would abort every time the game changes rooms.
        if decision == "unknown":
            stop = hold("loading...", state, poll)
            if stop:
                return stop
            continue

        # Three transients that look exactly like decision points but are not.
        # All three were found by running with --act on 2026-09-12; observe mode
        # can never show them, because nothing advances the turn.

        # 1. The game is executing a GameAction: a card resolving, a relic
        # firing, a reshuffle. Measured covering the whole of a mid-turn
        # reshuffle, which is the case no timer could have covered - the pause
        # there scales with the size of the discard pile.
        #
        # `is_resolving is None` means the bridge predates the field. Treat that
        # as "cannot tell" and fall through rather than block forever.
        if state.get("is_resolving"):
            stop = hold(f"resolving {state.get('resolving_action') or ''}", state, poll)
            if stop:
                return stop
            continue

        # 2. The enemy's turn. 1.2-1.6s measured, which the old 1s sleep landed
        # inside almost every time - that is why end_turn went out twice after
        # every turn. Scoped to combat because `turn` only exists there.
        if decision == "combat_play" and state.get("turn") != "player":
            stop = hold(f"{state.get('turn')} turn, waiting", state, poll)
            if stop:
                return stop
            continue

        # 3. The turn has started but the hand has not been dealt yet. Measured
        # at 0.53s on five consecutive turns, and `is_resolving` is False for all
        # of it - the opening draw is apparently not a GameAction, so the guard
        # above does not see it. During that window `hand` is empty and nothing
        # is playable, which is indistinguishable by inspection from "I played
        # every card". That is the silent turn-throw: the loop ends a turn it
        # never got to play.
        #
        # `round` is what separates them, and it needs no timer. An empty hand in
        # a round we have not acted in yet means the deal is still coming; an
        # empty hand in the round we just acted in means we emptied it ourselves.
        # One integer of history, where a timeout would have had to guess at a
        # duration that varies with the work.
        if (
            decision == "combat_play"
            and not (state.get("hand") or [])
            and state.get("round") != last_acted_round
        ):
            stop = hold(
                f"round {state.get('round')} opening, hand not dealt", state, poll
            )
            if stop:
                return stop
            continue

        # Past every guard: this is a real decision point.
        unchanged = 0
        prev_key = None
        ctx = state.get("context") or {}
        print(
            f"  [{step:3}/{polls:4}] act {ctx.get('act')} "
            f"floor {ctx.get('floor')}  {decision}"
        )

        payload, reason = choose(state, grid_picks)

        # "Not yet" - a transient the chooser can see but the generic guards
        # cannot, e.g. a treasure chest still opening or a play blocked by a hook.
        # Costs a poll, not a decision.
        if payload is WAIT:
            stop = hold(f"wait: {reason}", state, poll)
            if stop:
                return stop
            continue

        # "I cannot act here." Stopping beats spinning: on 2026-09-13 the loop
        # printed the same `event with no unlocked option (0 shown)` fifteen times
        # on a DROWNING_BEACON event whose real interaction the bridge cannot see
        # at all. Fifteen identical lines read like a hang; one honest exit names
        # the gap and says a human is needed.
        if payload is None:
            cannot_act += 1
            print(f"  [{step:3}/{polls:4}] cannot act: {reason}")
            if cannot_act >= give_up_after:
                return (
                    f"cannot act on {decision!r} after {cannot_act} tries: {reason}. "
                    f"either `choose` has no branch for it or the bridge cannot "
                    f"see the real interaction - play this bit by hand."
                )
            time.sleep(poll)
            continue
        cannot_act = 0

        if not act:
            print(f"        would {reason}")
        else:
            try:
                response = send(client, payload)
            except ApiError as e:
                # Shaped like a refusal, not like nothing. An HTTP 500 used to
                # leave `response = {}`, which fell into the `else` below and
                # *reset* the counter - so a play the bridge could not execute
                # was retried until the step budget ran out. A failure at the
                # transport layer is still the bridge saying no.
                print(f"        rejected: {e}")
                response = {"status": "error", "error": str(e)}

            # An explicit refusal is a different animal from a stall: the bridge
            # looked at the state and said no. Repeating it cannot help, so stop
            # and quote the reason rather than spending the whole step budget -
            # which is what the wrong `skip_card_reward` verb would have done,
            # 400 times, before announcing "budget reached".
            if response.get("status") == "error":
                refused += 1
                if refused >= give_up_after:
                    return (
                        f"{refused} actions refused in a row on {decision!r}, last: "
                        f"{response.get('error')!r} (sent for: {reason}). the action "
                        f"is wrong for this state, not merely early - check the "
                        f"handler's preconditions in BridgeMod.Actions.cs."
                    )
            else:
                refused = 0

        # Recorded whether or not the action was sent, so that observe mode walks
        # the same path act mode does. Tracking it only on success would make the
        # two modes diverge exactly where a new branch is being checked.
        # Reached only when an action was chosen - WAIT and None both continued
        # above - so there is no `payload is not None` check to make here.
        # Counted for the same reason last_acted_round is: in both modes, so that
        # observe and act walk the same path through a branch being checked.
        if decision == "card_select" and payload.get("action") == "select_card":
            grid_picks += 1
        elif decision != "card_select":
            grid_picks = 0

        if decision == "combat_play":
            last_acted_round = state.get("round")
        else:
            # Leaving combat clears the history, because round numbers restart at
            # 1 in the next fight. Without this, a fight that ended during round 1
            # leaves last_acted_round == 1, and the *next* fight's opening hand
            # looks like one we emptied ourselves - guard 3 waves it through and
            # the first turn gets thrown away. Observed the transition at t=30-34
            # in the 2026-09-13 log: round went 5 -> (non-combat) -> 1.
            last_acted_round = None

        step += 1
        # A beat between actions, not a wait on anything. The game animates and a
        # state read mid-animation may not be the one the next action assumes -
        # but correctness does not rest here: if the animation is still running
        # the guards above catch it on the next poll, however slow the frame rate.
        time.sleep(1)

    return f"made {max_steps} decisions in {polls} polls (budget reached)"


# --- A2 / A3 ------------------------------------------------------------------

# Values of a card's `target_type`, read straight off the game's
# `MegaCrit.Sts2.Core.Entities.Cards.TargetType` enum with tools/api_probe
# (2026-09-12). The complete set is:
#
#     None  Self  AnyEnemy  AllEnemies  RandomEnemy
#     AnyPlayer  AnyAlly  AllAllies  TargetedNoCreature  Osty
#
# Only `AnyEnemy` asks the agent who to hit. `AllEnemies` and `RandomEnemy`
# resolve themselves, and `None`/`Self` need nobody - so all four are played
# bare. The other five are multiplayer or special cases that IRONCLAD
# singleplayer should never produce; they are left out on purpose, so that if
# one ever shows up the reason string names it rather than the loop inventing a
# target and misfiring in silence.
# `choose` has three answers, not two, and conflating the last two caused real
# bugs on 2026-09-13. Returning None for "the game is mid-animation" made the
# loop burn decisions on a screen that was about to change; returning an action
# for it was worse (a treasure room left without its relic). WAIT says "poll
# again, this is not my turn yet"; None says "I genuinely cannot act here", which
# is a gap worth stopping on rather than spinning through.
WAIT = object()

NO_TARGET = {"None", "Self", "AllEnemies", "RandomEnemy"}
ENEMY_TARGET = {"AnyEnemy"}

# Why a card cannot be played, from the game's `UnplayableReason` enum
# (api_probe, 2026-09-13). Complete set:
#
#   None  HasUnplayableKeyword  BlockedByHook  BlockedByCardLogic
#   EnergyCostTooHigh  StarCostTooHigh  NoLivingAllies
#
# Every one of them is settled for the rest of this turn, so a hand of nothing
# but these means the turn really is over and the only move left is to end it.
#
# `BlockedByHook` used to be treated as a transient instead, on the theory that a
# hook is something else taking over the play - the user's 2026-09-13 note about
# relics that play for you. Observed live on 2026-09-13 at act 1 / floor 8, and
# that reading is wrong:
#
#   player debuff  SMOGGY_POWER  烟雾弥漫  amount 1  (from LIVING_FOG_0)
#                  "每回合你只能打出1张技能牌。"
#   hand           [0] 余烬 cost 2  EnergyCostTooHigh   (energy was 1)
#                  [1] 防御 cost 1  BlockedByHook
#                  [2] 防御 cost 1  BlockedByHook
#   card keyword   "烟雾 - 你在本回合无法再打出更多技能牌。回合结束时消除。"
#
# Read that last line: the block is cleared *by ending the turn*. Waiting for it
# to clear waits on the one action the wait refuses to take, and the debuff fires
# again every turn, so the fight could not advance at all - 185 polls on one
# decision before it was interrupted.
#
# There may well be a real relic that plays cards for you, and it may well report
# the same value. It does not need its own branch here: something actively taking
# the play is a GameAction, so `is_resolving` is true and guard 1 in read_loop
# holds before this code is ever reached. Getting here means the game is idle,
# and an idle game with no playable card is a game whose turn is finished.
#
# The reason string below lists the actual set, which is what made this
# diagnosable from the log alone - `end turn (3 in hand, none playable:
# ['BlockedByHook', 'EnergyCostTooHigh'])` names both halves of the situation.


# What number each action wants is NOT uniform, and the difference is invisible
# until it picks the wrong thing. Read off BridgeMod.Actions.cs and
# BridgeMod.StateBuilder.cs on 2026-09-13 rather than guessed:
#
#   decision        state list                     the action's index is
#   combat_play     hand, incl. can_play=false     the item's own `index` field
#   map_select      choices (already filtered)      position == index, aligned
#   combat_rewards  items (builder applies the      position == index, aligned
#                   same IsEnabled filter)
#   card_reward     cards                           aligned
#   card_select     cards                           aligned
#   relic_select    relics                          aligned
#   hand_select     cards                           aligned
#   treasure        relics                          aligned
#   event_choice    options, INCL. is_locked        position among NOT locked
#   rest_site       options, INCL. disabled         position among is_enabled
#
# The last two are the trap. The builder numbers every option 0..n including the
# locked ones, while the handler does `.Where(b => !b.Option.IsLocked)` and
# indexes into that. So when option 0 is locked, passing the state's index 1
# selects the *second unlocked* option, not the one that was looked at - and the
# game accepts it without complaint, exactly like the card_index bug.
def legal_position(items, flag: str, want=True, rng: random.Random | None = None):
    """(position-within-legal, item) for a legal entry, or (None, None).

    The position is deliberately the index *into the filtered list*, because
    that is what these handlers take. Note this is the opposite convention from
    play_card, which wants the item's own `index`; there is no single rule, only
    what each handler was written to expect.

    With no `rng` this returns the first legal entry, which is what every caller
    did before A3 step 6 and what the offline tests still assert. With one it
    draws uniformly from all of them.
    """
    legal = list(enumerate(i for i in items if i.get(flag) == want))
    if not legal:
        return None, None
    return rng.choice(legal) if rng is not None else legal[0]


def pick_potion(state) -> tuple[int, str | None] | None:
    """Which potion to drink right now, as (slot, target), or None for "none".

    Returns None on purpose, and the seam is the deliverable. Everything else for
    potions is already in place - `action_adapter.use_potion(slot, target)` on
    one side, `ExecuteUsePotion` on the other - so what was missing was never the
    plumbing, it was a single agreed place for the decision to live. Without that
    place, adding potions later means re-deciding *where* as well as *what*, in
    the middle of a loop that has other things wrong with it.

    Drinking a potion is a policy decision, the same category as which card to
    buy in a shop or which map node to take, and a fixed default would be a bad
    policy wearing a placeholder's clothes. Stage 5 gives it one; A5 moves this
    behind the policy object along with `choose`.

    What a policy gets to look at, all of it already in the combat state
    (`BuildPlayerState` fills it, `_normalize_combat` passes it through):

        state["player"]["potions"][i] = {
            "slot":              the number use_potion wants
            "id" / "name":       e.g. "ASHWATER" / "灰水"
            "description":       current formatted text
            "can_use_in_combat": PotionUsage is CombatOnly or AnyTime
            "target_type":       "AnyEnemy" needs a target, exactly like a card
        }

    Two rules carried over from cards, both of which have already caused a bug
    in their card form:

      * `target` is required exactly when `target_type == "AnyEnemy"`, and it is
        an `entity_id` out of `state["enemies"]`, never a list position.
      * `slot` is the slot number, not the position in this list. Empty slots are
        skipped when building it, so the two diverge the moment slot 0 is free -
        the same trap as `card_index` versus the position within `playable`,
        which on 2026-09-12 was caught only because energy ran out.

    Not consulted outside combat. The one place potions have actually blocked a
    run is a full-slot potion reward, and there the bridge reports the slot
    *counts* but not which potions are held, so there is nothing to choose from
    yet - see the shadowed_overlay rule, which leaves that reward behind instead.
    """
    return None


def choose(state, grid_picks: int = 0, rng: random.Random | None = None):
    """Pick an action for this state. Return (payload, reason).

    `grid_picks` is how many cards have already been picked on the current
    card_select grid. It defaults to 0 so every other decision - and every test -
    can ignore it; only the multi-card grid needs it, and only because the game
    exposes no way to read back what is already selected. A5 folds it into the
    policy object along with the rng.

    `map_select` and `combat_play` are covered; everything else falls through
    to None so the human keeps playing it. Adding one decision at a time keeps
    the blame narrow: if the game stops moving, it is the branch just added.

    **A3 step 6 (2026-09-17): the fixed picks are now random.** Until a real
    combat had been played through, first-legal-every-time was worth more than
    variety: a stall and an unlucky draw look identical, and a fixed choice
    leaves one suspect instead of two. That gate was passed on 2026-09-12, so
    the picks below draw from `rng` - which is `RNG` unless a caller hands one
    in, and which `read_loop` seeds and prints.

    Nine sites are randomised: the map node, which playable card, which enemy,
    which card reward, the event option, the rest option, the treasure and
    select relics, and the hand-select card. `combat_rewards` is deliberately
    **not** - it claims one item at a time until the screen is empty, so the
    order carries no information and randomising it only adds noise.

    Still missing: the loop cannot yet tell an action that was accepted and
    ignored from one that simply takes a moment, so it would resend forever.
    Detecting that needs a stall check on (floor, decision) - and once this
    function actually sends things, "not handled" and "handled but ignored"
    look identical on screen. The reason strings below are the only record of
    which one happened, so they have to describe what really occurred.
    """
    rng = RNG if rng is None else rng
    decision = state.get("decision")

    if decision == "map_select":
        choices = state.get("choices") or []
        if not choices:
            return None, "map with no choices"

        index = rng.randrange(len(choices))
        node = choices[index]
        return (action_adapter.choose_map_node(index), f"node {index}: {node['type']}")

    if decision == "combat_play":
        hand = state.get("hand")
        if hand is None:
            # Not the same as an empty hand. A missing key means the field was
            # renamed or this is not really a combat state, and `or []` would
            # have flattened that into "no cards" - two unrelated bugs wearing
            # the same face. The first draft of this branch read `choices`,
            # which does not exist here, and spun for a whole run saying
            # nothing.
            return None, "combat_play state has no 'hand' key"

        # Asked before cards, because a potion can change what is worth playing -
        # energy, block, a dead enemy - and asking afterwards would mean deciding
        # on a board state the potion was about to change. Declines by default, so
        # this is behaviour-neutral until a policy exists. See pick_potion.
        potion = pick_potion(state)
        if potion is not None:
            slot, target = potion
            return (
                action_adapter.use_potion(slot, target=target),
                f"use potion slot {slot}" + (f" -> {target}" if target else ""),
            )

        # `can_play` is the game's own verdict and already accounts for energy,
        # so there is no arithmetic to do here. Good thing too: `cost` is a
        # *string* ("1", or "X" for X-cost cards), so `cost > energy` would
        # raise rather than answer.
        playable = [c for c in hand if c.get("can_play")]
        if not playable:
            # No second-guessing *why*. `can_play` is the game's own verdict and
            # it already folds in energy, star cost, curses and debuffs; the
            # reasons are reported because they are worth reading in the log, not
            # because the decision turns on them. The one branch that did turn on
            # them - waiting out `BlockedByHook` - is what wedged floor 8.
            reasons = {c.get("unplayable_reason") for c in hand}
            return (
                action_adapter.end_turn(),
                f"end turn ({len(hand)} in hand, none playable: "
                f"{sorted(r for r in reasons if r)})",
            )

        skipped = []
        # A copy: `playable` is used again below to build the reason string, and
        # shuffling it in place would make the message describe an order that is
        # not the one the loop walked.
        order = list(playable)
        rng.shuffle(order)
        for card in order:
            # The card's index in `hand`, not its position in `playable`.
            # Filtering shifted everything: once hand[0] is unplayable,
            # playable[0] *is* hand[2]. play_card takes the hand index, so
            # passing the list position plays a different card than the one
            # chosen - and the game accepts it without complaint. This is the
            # bug that cannot happen on turn 1, because with full energy every
            # card is playable and the two numbers agree.
            card_index = card["index"]
            name = card.get("name")
            target_type = card.get("target_type")

            if target_type in NO_TARGET:
                return (
                    action_adapter.play_card(card_index),
                    f"play [{card_index}] {name} ({target_type})",
                )

            if target_type in ENEMY_TARGET:
                alive = [
                    e for e in (state.get("enemies") or []) if (e.get("hp") or 0) > 0
                ]
                if not alive:
                    skipped.append(f"[{card_index}] {name}: no living enemy")
                    continue
                target = rng.choice(alive)["entity_id"]
                return (
                    action_adapter.play_card(card_index, target=target),
                    f"play [{card_index}] {name} -> {target}",
                )

            skipped.append(f"[{card_index}] {name}: target_type {target_type!r}")

        # Every playable card needed something this branch cannot supply. End
        # the turn rather than stall - but say exactly what was refused, since
        # an unhandled `target_type` showing up here is news.
        return (
            action_adapter.end_turn(),
            "end turn, nothing sendable: " + "; ".join(skipped),
        )

    # --- the meta decisions ----------------------------------------------------
    # Roughly 120 of these per run against ~700 combat decisions, and they are
    # the half an LLM is supposed to take over later. Fixed choices for now, for
    # the same reason combat started on `index=0`: an unproven write path plus a
    # random choice gives two suspects for every failure. Randomising all of them
    # together is step 6.

    if decision == "card_reward":
        cards = state.get("cards") or []
        if cards:
            card = rng.choice(cards)
            return (
                action_adapter.select_card_reward(card["index"]),
                f"take card [{card['index']}] {card.get('name')} "
                f"({card.get('rarity')}) of {len(cards)}",
            )
        # No cards left to take, so the row of alternatives is the only way off
        # this screen. Which one matters: the row is not all "skip". Pael's Wing
        # adds a sacrifice, and a reroll does not even close the screen
        # (PostAlternateCardRewardAction.DoNothing), so picking it would leave
        # the loop resending forever. Prefer an id that says skip; otherwise say
        # what is being clicked instead of calling it a skip.
        alts = [a for a in (state.get("alternatives") or []) if a.get("is_enabled")]
        if alts:
            # Matched on the stable id, never on the title. Two wrong guesses
            # preceded this, both caught by running the game rather than by
            # reading it: the button's `_optionName` turned out to be the
            # localised label ("跳过", so `"SKIP" in ...` could not fire in a
            # Chinese client), and the id lookup turned out to cover the ordinary
            # skip as well as the relic-added options, so "is it an extra" did not
            # separate them either. What is left is the one thing that is actually
            # stable: OptionId == "Skip".
            by_id = [a for a in alts
                     if (a.get("option_id") or "").strip().upper() == "SKIP"]
            pick, how = ((by_id[0], "option_id is Skip") if by_id
                         else (alts[0], "NO Skip id - taking position 0 blind"))
            label = pick.get("title") or pick.get("option_id") or "unknown"
            return (
                action_adapter.skip_card_reward(pick["index"]),
                f"card reward alternative [{pick['index']}] {label} "
                f"of {len(alts)} ({how})",
            )
        # An older bridge reports no `alternatives` at all; `can_skip` is all
        # there is, and index 0 is the guess it always silently was.
        if state.get("alternatives") is None and state.get("can_skip"):
            return (
                action_adapter.skip_card_reward(),
                "skip card reward (old bridge: no `alternatives`, index 0 is a guess)",
            )
        return None, "card_reward with no cards and no enabled alternative"

    if decision == "combat_rewards":
        # Gold, potions and relics all arrive here.
        #
        # "The builder drops anything not claimable, so every listed item can be
        # taken" is what this comment used to say, and it is wrong. The builder
        # drops buttons with `!IsEnabled`, and a potion reward offered with no
        # free slot stays enabled: the game is willing to be clicked, it just
        # cannot give you the potion. Observed at act 1 / floor 9 with
        # `open_potion_slots: 0` - claiming index 0 did nothing, the state came
        # back identical, and the loop resent it until the step budget ran out
        # and reported "budget reached" as though it had finished a run.
        #
        # So the filter has to happen on this side too. Being *in* `items` means
        # the game will accept the click, not that anything will come of it.
        items = state.get("items") or []
        player = state.get("player") or {}

        # `== 0`, not `or 0`: if this field ever goes missing the answer is None,
        # which is not 0, so the potion gets claimed and the old loud stall comes
        # back. The alternative reading would skip every potion for ever and
        # report nothing, and a quiet loss is the worse of the two failures.
        potions_full = player.get("open_potion_slots") == 0
        takeable = [i for i in items if not (potions_full and i.get("type") == "potion")]
        left_behind = len(items) - len(takeable)

        if takeable:
            item = takeable[0]
            return (
                action_adapter.claim_reward(item["index"]),
                f"claim [{item['index']}] {item.get('type')} "
                f"{item.get('gold_amount') or item.get('potion_name') or ''} "
                f"({len(takeable)} takeable of {len(items)})",
            )
        if state.get("can_proceed"):
            # Walking away from a potion on purpose. Freeing a slot means
            # drinking one, and which potion to drink when is a policy - the same
            # reason the shop buys nothing. Said out loud so it never looks like
            # the reward was simply missed.
            if left_behind:
                # `discard_potion` exists now, so this is a choice rather than a
                # limitation - and still not one a fixed rule should make. Which
                # potion is worth throwing away is the same kind of judgement as
                # what to buy in the shop, which is why that spends nothing
                # either. The LLM policy is offered the verb; this one declines.
                return action_adapter.proceed(), (
                    f"proceed, leaving {left_behind} potion(s): all "
                    f"{player.get('potion_slots')} slots full. discard_potion could "
                    f"free one, but which to throw away is a policy decision"
                )
            return action_adapter.proceed(), "proceed (all rewards claimed)"
        # Nothing to take and no way out. Until 2026-09-13 this was a dead end
        # caused by the bridge looking for the proceed button inside the rewards
        # overlay, where it never is - `can_proceed` was permanently false. If it
        # happens again, `is_complete` says whether the screen itself thinks it is
        # done, which separates "the button is still missing" from "the screen is
        # genuinely waiting on something else".
        return None, (
            f"combat_rewards: nothing takeable ({len(items)} listed, "
            f"{left_behind} unclaimable), can_proceed=False, "
            f"is_complete={state.get('is_complete')}, "
            f"open_potion_slots={player.get('open_potion_slots')}"
        )

    if decision == "event_choice":
        # Dialogue first: an event that is still talking has no options to pick.
        if state.get("in_dialogue"):
            return action_adapter.advance_dialogue(), "advance event dialogue"
        options = state.get("options") or []
        position, option = legal_position(options, "is_locked", want=False, rng=rng)
        if option is None:
            # Name the widget the bridge could not read. An event with no options
            # and no dialogue means something is on screen that BuildEventState
            # does not look for - the reward screens were that, until 2026-09-13.
            overlay = state.get("top_overlay_type")
            return None, (
                f"event {state.get('event_id')} has no unlocked option "
                f"({len(options)} shown)"
                + (
                    f"; top overlay is {overlay} - the bridge has no branch for it"
                    if overlay
                    else "; no overlay on top either, so the widget is inside the "
                    "event room and BuildEventState does not look for it"
                )
            )
        return (
            action_adapter.choose_event_option(position),
            f"event option @{position} (state index {option.get('index')}) "
            f"{option.get('title')}",
        )

    if decision == "rest_site":
        options = state.get("options") or []
        position, option = legal_position(options, "is_enabled", rng=rng)
        if option is not None:
            return (
                action_adapter.choose_rest_option(position),
                f"rest option @{position} (state index {option.get('index')}) "
                f"{option.get('name')}",
            )
        if state.get("can_proceed"):
            return action_adapter.proceed(), "leave rest site (no enabled option)"
        return None, "rest_site with no enabled option and no proceed"

    if decision == "shop":
        # Buying nothing, on purpose, and for two separate reasons.
        #
        # One: the index mapping here is the only one that could not be confirmed
        # from source. The builder numbers CardEntries then RelicEntries in one
        # sequence, while the handler indexes GetLocalInventory().AllEntries -
        # nothing proves those orders agree, and `is_stocked` says sold-out
        # entries stay listed. Sending an unverified index would buy a random
        # item and look like it worked.
        #
        # Two: what to spend gold on is a strategy, not a default. Picking item 0
        # is not a neutral placeholder - it is a bad policy wearing one.
        # Sent without checking `can_proceed`, which for a shop is not a usable
        # precondition. It reports `NMerchantRoom.ProceedButton.IsEnabled`, and
        # the state builder opens the shop inventory on *every* read
        # (`if (!merchUI.Inventory.IsOpen) merchUI.OpenInventory()`), which
        # disables that button - so it is permanently false. Guarding on it
        # blocked the one action that clears it: ExecuteProceed's merchant branch
        # closes the inventory with the back button first and only then presses
        # proceed (BridgeMod.Actions.cs:692-706).
        #
        # Wedged a run at act 1 / floor 14 for 8 tries saying "shop with no
        # proceed button"; one manual `proceed` against that exact state returned
        # "Proceeding from shop" and the state went straight to `map`. One call
        # does both halves.
        #
        # Third instance of the same shape - a state builder whose side effect
        # creates the condition that blocks the exit. The others: the treasure
        # chest opened by reading it, and the rewards screen that stayed on the
        # stack. Worth expecting a fourth.
        #
        # A real failure is still loud: the loop prints `rejected: No proceed
        # button available or enabled` on every attempt, naming the error.
        return action_adapter.proceed(), (
            f"leave shop, buying nothing ({len(state.get('items') or [])} items, "
            f"index mapping unverified; can_proceed={state.get('can_proceed')} "
            f"is an artifact of the builder auto-opening the inventory)"
        )

    if decision == "treasure":
        relics = state.get("relics") or []
        if relics:
            relic = rng.choice(relics)
            return (
                action_adapter.claim_treasure_relic(relic["index"]),
                f"take relic [{relic['index']}] {relic.get('name')}",
            )
        # No relics yet does not mean there are none. BuildTreasureState clicks
        # the chest itself - reading the state is what opens it - so the first
        # read always comes back empty with a "Opening chest..." message while
        # the relics appear a read or two later. Proceeding here walks out of the
        # room without the relic, and nothing would report that it happened.
        if state.get("message"):
            return WAIT, f"treasure: {state.get('message')}"
        if state.get("can_proceed"):
            return action_adapter.proceed(), "leave treasure room (relic taken)"
        return WAIT, "treasure with no relic yet, waiting for the chest"

    if decision == "relic_select":
        relics = state.get("relics") or []
        if relics:
            relic = rng.choice(relics)
            return (
                action_adapter.select_relic(relic["index"]),
                f"select relic [{relic['index']}] {relic.get('name')} of {len(relics)}",
            )
        if state.get("can_skip"):
            return action_adapter.skip_relic_selection(), "skip relic select"
        return None, "relic_select with no relics and no skip"

    # Two-step screens: pick, then confirm. Confirm is checked first because
    # after a pick lands, `can_confirm` is the only thing that changed - looking
    # at the card list first would re-pick forever.
    if decision == "card_select":
        if state.get("can_confirm"):
            return action_adapter.confirm_selection(), "confirm card selection"
        cards = state.get("cards") or []

        # The screen now reports how many are picked and how many it wants, so
        # stop guessing when both are there. The guess below - keep picking fresh
        # indices until `can_confirm` lights up - is wrong for any screen asking
        # for an exact number: 选择2张牌来移除 offered 14 cards, wanted 2, and the
        # loop picked all fourteen and gave up. Twice, on 2026-09-14.
        #
        # `is not None`, not truthiness: `selected_count` of 0 is the normal
        # starting state and `min_select` of 0 is a legal screen.
        picked = state.get("selected_count")
        want = state.get("min_select")
        if picked is not None and want is not None:
            if picked < want:
                # Pick a card the screen does not already have. `select_card`
                # toggles, so re-sending a selected index would undo it - and the
                # set of selected indices is still not reported, only its size.
                # Walking forward by the count is enough: each step adds one.
                if picked < len(cards):
                    card = cards[picked]
                    return (
                        action_adapter.select_card(card["index"]),
                        f"select card {picked + 1}/{want} [{card['index']}] "
                        f"{card.get('name')} ({state.get('screen_type')}, "
                        f"{len(cards)} shown)",
                    )
                return None, (
                    f"card_select wants {want} cards but only {len(cards)} are "
                    f"shown and {picked} are picked - the screen cannot be satisfied"
                )
            # Enough picked and confirm is still dark: the screen is between its
            # two confirmation stages, or it confirms itself. Wait rather than
            # picking more, which would go over the limit.
            return WAIT, (
                f"card_select has {picked}/{want} picked, waiting for confirm "
                f"(preview_showing={state.get('preview_showing')})"
            )
        # `grid_picks`, not a fixed 0. Some screens want more than one card - an
        # event handing out two, a multi-upgrade - and on this screen
        # `select_card` **toggles**: ExecuteSelectCard emits HolderPressed and the
        # bridge's own message is "Toggling card selection". Selected cards stay
        # in the grid, and the state has no flag saying which ones they are -
        # `NGridCardHolder` has no Select member at all (api_probe, 2026-09-13)
        # and `NCardGridSelectionScreen` exposes only `Task CardsSelected()`, so
        # there is nothing to read.
        #
        # Re-sending index 0 would therefore select, deselect, select, ... while
        # `can_confirm` stayed false, for the whole step budget. Advancing the
        # index instead needs no selection flag: each pick is a card not yet
        # picked, so the count only goes up, and `can_confirm` ends it whenever
        # the screen is satisfied. One counter, the same shape as
        # `last_acted_round`.
        #
        # Contrast `hand_select`, which needs none of this: its `cards` comes
        # from `hand.ActiveHolders` and a chosen card *moves* to
        # %SelectedHandCardContainer, so it leaves the list and index 0 is always
        # the next unpicked card.
        if grid_picks < len(cards):
            card = cards[grid_picks]
            return (
                action_adapter.select_card(card["index"]),
                f"select card #{grid_picks + 1} [{card['index']}] {card.get('name')} "
                f"({state.get('screen_type')}, {len(cards)} shown)",
            )
        if cards:
            # Every card picked and still no confirm. Stopping beats toggling:
            # either the screen wants a different action, or `can_confirm` is
            # being read from the wrong node.
            return None, (
                f"card_select ({state.get('screen_type')}): picked all "
                f"{len(cards)} cards and can_confirm never became true"
            )
        if state.get("can_skip"):
            # `cancel_selection`, not `skip_card_reward`. The latter starts with
            # `if (overlay is not NCardRewardSelectionScreen) return Error(...)`,
            # and a card_select state is by definition one of the *other* two
            # screens - so it could only ever have failed. ExecuteCancelSelection
            # is the one that handles NChooseACardSelectionScreen's "SkipButton"
            # (BridgeMod.Actions.cs:841). The old call would have come back as an
            # ApiError, printed one `rejected:` line, and been retried until the
            # step budget ran out and announced "budget reached".
            return action_adapter.cancel_selection(), (
                f"skip card select ({state.get('screen_type')})"
            )
        return None, f"card_select ({state.get('screen_type')}) with no cards"

    if decision == "hand_select":
        if state.get("can_confirm"):
            return action_adapter.combat_confirm_selection(), "confirm hand selection"
        cards = state.get("cards") or []
        if cards:
            card = rng.choice(cards)
            return (
                action_adapter.combat_select_card(card["index"]),
                f"hand-select [{card['index']}] {card.get('name')} "
                f"({state.get('mode')}, {state.get('prompt')})",
            )
        return None, f"hand_select ({state.get('mode')}) with no cards"

    if decision == "overlay":
        # A screen with no branch in the bridge. Of the 14 IOverlayScreen
        # implementers, 13 are matched by name; the one that is not is the
        # crystal sphere minigame, and it owns a proceed button. Leaving probably
        # forfeits whatever it was offering, which is worth saying out loud - but
        # the alternative is the run stopping here, and the loop can only report
        # a class name it has no branch for.
        screen = state.get("screen_type") or "unknown screen"
        if state.get("can_proceed"):
            return (
                action_adapter.proceed(),
                f"proceed past unhandled overlay {screen} "
                f"(leaving it - this may forfeit what it offers)",
            )
        return None, (
            f"overlay {screen} has no branch and no live proceed button - "
            f"play this bit by hand, then add a branch for it"
        )

    return None, f"not handled yet: {decision}"


def send(client, payload) -> dict:
    """Post one action and return what the bridge said about it.

    The adapter puts the verb under payload["action"] and the rest are keyword
    arguments, so it has to come back out before the call:

        body = dict(payload)
        verb = body.pop("action")
        client.post_action(verb, **body)

    Transmission only: what a rejection means is the caller's decision, and
    `read_loop` makes it. Keeping the policy out of here means changing it
    later - to retry, say - touches the loop and not this function.

    The response used to be discarded, which was the single worst line in this
    file. A refused action comes back as a *return value*, not an exception -
    `{"status": "error", "error": "Item is sold out"}` with a perfectly normal
    HTTP 200 - so `except ApiError` in the loop never saw it, and a rejected
    action printed exactly like an accepted one. Every index-mapping question in
    this project is answered by a string that was being thrown away:

        "Selecting rest site option: 强化"      <- which option really got clicked
        "Traveling to Treasure at (2,9)"       <- which node really got taken
        "Not enough gold (need 151, have 84)"  <- the cost the *handler* resolved
        "Claiming reward: potion (灰水)"        <- status ok, and nothing happened

    That last one is the case to keep in mind: `status: ok` is the bridge saying
    "I clicked it", not "it worked". Printing the message does not close that gap
    - only comparing states does - but it at least stops the loop from hiding
    the half it does know.
    """
    body = dict(payload)
    verb = body.pop("action")
    print(f"    -> post_action({verb!r}, **{body})")
    response = client.post_action(verb, **body) or {}
    status = response.get("status")
    detail = response.get("error") or response.get("message")
    if status == "error":
        print(f"       !! refused: {detail}")
    elif detail:
        print(f"       bridge: {detail}")
    return response


class Watcher(NamedTuple):
    """A spawned watch_resolving.py and the two files it talks through."""

    proc: subprocess.Popen
    workdir: Path
    stop_file: Path      # its existence is the "keep going" signal
    summary_file: Path   # where it leaves the measurement on the way out


def spawn_watcher(base_url: str) -> Watcher | None:
    """Open tools/watch_resolving.py in its own console. None if it did not start.

    Tied to --act because the window it hunts for only opens when turns actually
    advance. It has to be a separate process, not a branch in this loop: the
    watcher samples at 10Hz to catch something sub-second, while this loop waits
    a beat between actions on purpose. One cannot poll at the other's rate.

    It is still never killed. Its summary ("danger windows that refilled: N") is
    the measurement, and terminating the process to tidy up would throw away the
    result the run was for - so instead it is *asked* to stop, by removing the
    file it watches, and it writes the summary down before it goes. See
    finish_watcher.
    """
    watcher = Path(__file__).with_name("watch_resolving.py")
    if not watcher.exists():
        print(f"  (no watcher at {watcher}, skipping)")
        return None
    # CREATE_NEW_CONSOLE is Windows-only. Elsewhere, say so rather than quietly
    # running the experiment without its instrument.
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    if not flags:
        print("  (new-console spawn is Windows-only; run the watcher yourself)")
        return None

    workdir = Path(tempfile.mkdtemp(prefix="autoplay-watch-"))
    stop_file = workdir / "running"
    summary_file = workdir / "summary.txt"
    # Created *before* the spawn. The other order is a race: the watcher would
    # check a file that does not exist yet and exit immediately, reporting a
    # clean run that never happened.
    stop_file.write_text("", encoding="utf-8")

    # No --hold: this console is meant to close itself. The watcher still holds
    # the window open if it crashes, because a traceback in a window that
    # vanishes is the reason --hold was written in the first place.
    proc = subprocess.Popen(
        [
            sys.executable, str(watcher),
            "--base-url", base_url,
            "--stop-file", str(stop_file),
            "--summary-file", str(summary_file),
        ],
        creationflags=flags,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    return Watcher(proc, workdir, stop_file, summary_file)


def finish_watcher(w: Watcher, timeout: float = 5.0) -> None:
    """Ask the watcher to stop, then print what it measured.

    Called from a `finally`, so the numbers come back whether the loop stopped
    on its own, raised, or was interrupted - the three ways a real session ends.

    The summary is printed *here*, in the terminal the run was started from.
    Before this, reading it meant noticing a second console and pressing Ctrl-C
    in it, which is an easy thing to forget and silently loses the measurement.
    """
    w.stop_file.unlink(missing_ok=True)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if w.summary_file.exists():
            print("watcher:")
            for line in w.summary_file.read_text(encoding="utf-8").splitlines():
                print(f"  {line}")
            break
        time.sleep(0.1)
    else:
        # Said out loud rather than passed over. No summary means the instrument
        # did not report, and a run with no measurement should not look like a
        # run with a clean one.
        print(
            f"watcher: no summary after {timeout:.0f}s - its console should still "
            f"be open with the reason (files in {w.workdir})"
        )
        return

    try:
        w.proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    shutil.rmtree(w.workdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument(
        "--act",
        action="store_true",
        help="send actions; without it the loop only watches",
    )
    ap.add_argument(
        "--no-watch",
        action="store_true",
        help="with --act, do not open the watch_resolving.py console",
    )
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the agent's own random choices (map node, card reward, ...). Omitted = a fresh one, printed either way. Note this does NOT seed the game itself, so it replays the decision sequence, not the run.")
    ap.add_argument("--base-url", default="http://localhost:15526")
    args = ap.parse_args()

    client = Sts2RawClient(base_url=args.base_url, timeout=10.0)
    print(f"mode: {'act' if args.act else 'observe'}")

    watcher = None
    if args.act and not args.no_watch:
        watcher = spawn_watcher(args.base_url)
        if watcher:
            print("  watcher opened in a second console (10Hz, read-only)")
            # It has to be sampling before the first action lands, or the first
            # turn transition goes unobserved.
            time.sleep(1.5)

    # The watcher is shut down in a `finally` because the interesting endings are
    # the abnormal ones. A bridge that drops mid-run or a Ctrl-C used to leave
    # its console orphaned, still polling a game nobody was driving.
    try:
        try:
            reason = read_loop(client, args.max_steps, args.act,
                               seed=args.seed)
        except ApiError as e:
            print(f"bridge unreachable: {e}")
            return 1
        print(f"stopped: {reason}")
        return 0
    finally:
        if watcher:
            finish_watcher(watcher)


if __name__ == "__main__":
    raise SystemExit(main())
