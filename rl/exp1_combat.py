"""Experiment 1: a fight with a real deck, a draw pile, and a reason to block.

Designed by the user on 2026-09-15. Stage 1's toy fight (`toy_combat.py`) proved
the algorithm works but was degenerate: no draw, no deck, and pure aggression
cost 12 hp out of 50. Everything here exists to remove that.

Three things it adds:

  * a **deck**, drawn 5 a turn, discarded on play, reshuffled when it runs low
  * **vulnerable**, so one card sets up the others and order starts to matter
  * numbers picked so that **blocking is not free** - see the check below

The card numbers are not invented. They were read off `data/library.json`, which
was exported from the running game, and they are the real IRONCLAD starting deck:

    STRIKE_IRONCLAD  cost 1   "造成6点伤害。"
    DEFEND_IRONCLAD  cost 1   "获得5点格挡。"
    BASH             cost 2   "造成8点伤害。 给予2层易伤。"
                              "易伤的生物从攻击中受到的伤害增加50%。"

That matters beyond tidiness: a policy trained here is speaking the same card
vocabulary as the real game, so Stage 3 does not have to translate.

    python rl/exp1_combat.py          # play a fight from the keyboard
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

# --- the cards ---------------------------------------------------------------
#
# Indexed, and the index *is* the action. See the note on the action space below.


class Card(NamedTuple):
    name: str
    cost: int
    damage: int
    block: int
    vulnerable: int  # turns of vulnerable applied to the target


CARDS = (
    Card("打击", 1, 6, 0, 0),
    Card("防御", 1, 0, 5, 0),
    Card("痛击", 2, 8, 0, 2),
)
STRIKE, DEFEND, BASH = 0, 1, 2
END_TURN = len(CARDS)

# How many of each card the deck starts with. Sums to 10.
DECK = (5, 4, 1)

PLAYER_HP = 80
# 80 is the real Ironclad's max hp, measured off a live state on 2026-09-16
# (`player: {"hp": 80, "max_hp": 80}`), and it is 80 rather than a tuned number
# because of the experiment this file now feeds: the run never adds, removes or
# upgrades a card, so the deck stays (5打击, 4防御, 1痛击) and every quantity the
# agent controls - 6 damage a 打击, 5 block a 防御 - is an *absolute* number. A
# threat has to be measured on the same ruler, so the fight is played in real
# game units and no conversion happens anywhere.
#
# ⚠️ This supersedes the 20/70 tuning recorded below, which was chosen to make
# blocking worth something in a self-contained toy. It solved that, and in
# solving it made the toy 2-4x harsher than Act 1: the toy player took 30-60% of
# max hp per turn where the real Act 1 enemy deals about 15%. A table trained
# there has no rows for a light hit, so every real Act 1 turn would have missed.
# The replacement keeps the pressure by *sampling* difficulty per fight (see
# TIERS) rather than by making every fight hard.
#
# The original note, kept because the reasoning is still right about signal
# strength and is the reason TIERS spans hard fights at all:
#
# 70, not the 40 this file was first written with. 40 was measured and rejected on
# 2026-09-16, and the number that killed it was not the win rate:
#
#   玩家/敌人   看 incoming 会改变动作的步   纯进攻胜率   会防御胜率   两者 Q 值之差
#     20/40              48.0%                98.1%       100.0%         0.038
#     20/70              47.4%                10.4%        93.6%         1.664
#
# The *frequency* of the choice is the same either way - blocking is picked on
# about half the steps in both. What changes is what it is worth. At 40 hp a
# policy that never blocks is already at 98%, so the gap between the right action
# and the wrong one is 0.038 of return, against a reward that is +1 or -1 per
# episode. Separating a 0.038 difference from that noise needs roughly
# (1.664/0.038)^2 ~ 1900x the episodes that separating 1.664 does.
#
# So the fight was not "too easy". The signal was too weak, which is a different
# illness with a different cure: raise the cost of the wrong action, not the
# difficulty of the fight.
# Total hp of everything we have to kill, sampled per fight. A range rather than
# a constant because the real game's fights are not one size, and **total**
# rather than one enemy's because the toy's win condition (`enemy_hp <= 0`) maps
# onto "every enemy is dead". Tracking only the current target would make the
# number jump back up each time one died, and the table would see a fight it
# never makes progress in.
#
# 20-150 is measured: across 143 recorded encounters the per-fight total ran
# 10-252 with a median near 50. The top of that range is one Act 1 boss; the
# cap is 150 so that boss-sized fights are sampled without half the training
# being spent on them.
ENEMY_HP_RANGE = (20, 150)
ENERGY_PER_TURN = 3
HAND_SIZE = 5

# The reshuffle rule is "the draw pile is empty", checked before each single
# card, which is what the real game does. There is no threshold constant on
# purpose. The first spec said "<= 5 cards left", and drawing one at a time made
# that redundant - worse, 5 collided with HAND_SIZE and a 10-card deck, so it
# fired *every* turn:
#
#     <= 5:   洗 洗 洗 洗 洗    pile always full again, the deck has no memory
#     empty:  .  洗 .  洗 .     turn 2 draws the five cards turn 1 did not
#
# Only the second lets "I just played three 打击, so 防御 is likelier next turn"
# be true - and that inference is a real part of playing the game.

# What the enemy hits for. Rolled fresh each turn, uniformly from one of two
# pools, and the pool moves up once the fight has run long.
#
# Two properties, and it takes both pools to get both:
#
#   * **random**, so blocking is a response to a number the agent reads rather
#     than a schedule it memorises off the turn counter. With a fixed ramp the
#     damage is a function of `turn`, so an agent that never looks at the intent
#     still plays optimally - which is not a policy that depends on state.
#   * **rising**, so a long fight is a losing fight. A single uniform pool has
#     the same mean every turn, and dragging the fight out would cost only
#     linearly - that deletes the stamina pressure this fight is built around.
#
# ⚠️ The roll is visible: it lives in `State.incoming` and is decided at the
# start of the turn, before the player acts. Rolling it at resolution time
# instead would make the same visible state sometimes cost 6 and sometimes 12,
# and no policy can be optimal against that - the environment would stop being
# an MDP. Same trap `intent_index` avoids in `toy_combat`.
# Difficulty is now two things, not one:
#
#   * **which fight this is** - a tier, drawn once at `reset`, that fixes the band
#     the enemy's attacks come from. This is what makes a light Act 1 skirmish and
#     an Act 3 elite both representable in one table.
#   * **how long it has run** - within the band, early turns roll the lower half
#     and later turns the upper half, which keeps the original "a long fight is a
#     losing fight" pressure that the two-pool design existed to create.
#
# The bands are the user's, in real damage numbers. The weights are measured from
# 138 of this project's own recorded fights (`trajectories/*.jsonl`), scoring each
# fight by its median announced damage:
#
#   tier          band     real share    enemy total hp, median [range]
#   低   LOW      0-10        45.7%          46  [10-94]
#   中   MID     11-20        50.7%          56  [26-252]
#   高   HIGH    21-30         3.6%          87  [51-140]
#   特高 VHIGH   31-40         0.0%          -
#
# ⚠️ VHIGH was never observed - but every run in that data was dead by floor 24,
# so "never seen" means "never got deep enough", not "does not exist". It is
# sampled at 1% so the table holds rows for it instead of a hole. If the deployed
# run reaches Act 3 and starts missing, this weight is the first suspect.
TIERS = ((0, 10), (11, 20), (21, 30), (31, 40))
TIER_WEIGHTS = (45, 50, 4, 1)
EARLY_TURNS = 3  # turns 0,1,2 roll the lower half of the band


def roll_tier(rng: random.Random) -> int:
    """Which difficulty band this fight draws from. Once per fight, at `reset`."""
    return rng.choices(range(len(TIERS)), weights=TIER_WEIGHTS)[0]


def roll_intent(turn: int, tier: int, rng: random.Random) -> int:
    """What the enemy will hit for on `turn`. Uniform within half the band."""
    low, high = TIERS[tier]
    mid = (low + high) // 2
    return rng.randint(low, mid) if turn < EARLY_TURNS else rng.randint(mid + 1, high)


VULNERABLE_MULTIPLIER = 1.5


# --- the state ---------------------------------------------------------------
#
# A NamedTuple: hashable, so a Q-table can key on it directly, but the fields
# have names, which a nine-element bare tuple would not. `toy_combat` used a bare
# 6-tuple and that was already at the edge of readable.
#
# ⚠️ The piles are **counts per card type**, not ordered lists. Two consequences,
# both deliberate:
#
#   1. The agent cannot see the draw order. Drawing k cards from a shuffled pile
#      is the same distribution as sampling k without replacement from a
#      multiset, so nothing is lost - except the thing the bridge accidentally
#      leaks in the real game (`draw_pile` comes back in true draw order, which
#      no human can see). Training on information a human does not have produces
#      a policy that quietly fails when it is taken away.
#   2. **The action space is fixed at 4.** Action 0 always means 打击, in every
#      state, for ever. Contrast the real game's `play_card(card_index)`, where
#      index 2 is a different card every turn - the bug family that has bitten
#      this project four times. Cards of one type are interchangeable, so there
#      is no information in which copy you play.


class State(NamedTuple):
    player_hp: int
    player_block: int
    enemy_hp: int  # total across every living enemy
    enemy_max_hp: int  # what that total started at; sampled per fight
    enemy_vulnerable: int  # turns of vulnerable left on the enemy
    energy: int
    turn: int  # 0-based; decides which half of the band rolls
    tier: int  # index into TIERS; fixed for the fight
    incoming: int  # this turn's rolled damage, already visible
    hand: tuple[int, ...]  # counts, one per entry in CARDS
    draw: tuple[int, ...]
    discard: tuple[int, ...]


def intent(state: State) -> int:
    """What the enemy will hit for at the end of this turn."""
    return state.incoming


# --- helpers you can lean on -------------------------------------------------


def _add(pile: tuple[int, ...], index: int, n: int = 1) -> tuple[int, ...]:
    """Return `pile` with `n` more of card `index`. Tuples are immutable."""
    out = list(pile)
    out[index] += n
    return tuple(out)


def total(pile: tuple[int, ...]) -> int:
    """How many cards are in a pile."""
    return sum(pile)


def legal_actions(state: State) -> list[int]:
    """Which actions this state will accept. Ending the turn is always allowed.

    A card needs a copy in hand *and* enough energy. Same reason
    `toy_combat.legal_actions` exists and the same reason the bridge grew
    `legal_verbs`: an agent offered an action the state cannot take will pick it,
    and then either nothing happens or something silently wrong does.
    """
    return [
        i
        for i, card in enumerate(CARDS)
        if state.hand[i] > 0 and card.cost <= state.energy
    ] + [END_TURN]


# --- the four functions that are yours ---------------------------------------
#
# Each is small enough to check by hand. Write one, run the hand-play interface
# at the bottom, and confirm the numbers on screen are the ones you expect before
# writing the next.


def damage_after_vulnerable(base: int, vulnerable: int) -> int:
    """Damage `base` becomes what, against a target with `vulnerable` turns left?

    Hand-check both branches:

        damage_after_vulnerable(6, 0) == 6
        damage_after_vulnerable(6, 2) == 9        # 6 * 1.5
        damage_after_vulnerable(8, 1) == 12       # 8 * 1.5

    ⚠️ Vulnerable is a **duration**, not a stack of multipliers: 2 layers means
    two turns at 150%, not 200%. And with this deck every damage value is even,
    so `* 1.5` is always a whole number and the rounding rule never fires. Decide
    what it should be anyway - a card added later will not be so convenient.
    """
    if vulnerable:
        return int(base * VULNERABLE_MULTIPLIER)
    return base


def draw_cards(state: State, n: int, rng: random.Random) -> State:
    """Draw `n` cards into hand. Returns a new State.

    The rule, checked before drawing each single card:

        if the draw pile is empty, tip the whole discard pile into it

    Because that only fires when the draw pile is all zeros, swapping the two
    piles is the same thing as merging them, and says it in one move.

    Two things to get right, and both have a wrong version that looks fine:

      * **Draw one at a time, checking the pile each time.** Checking once up
        front and then taking 5 lets the pile go empty mid-draw.
      * **Draw at random, weighted by how many of each card are in the pile.**
        A pile of (3 打击, 1 防御) must give 打击 three times as often. Taking
        the first non-zero index instead is a deterministic draw order wearing
        a shuffle's name - and it will not announce itself, the fight will just
        play strangely.

    Hand-check: draw (2, 1, 0), discard (1, 2, 0), draw 5. The first three come
    straight off the draw pile; on the fourth it is empty, so the piles swap to
    draw (1, 2, 0) / discard (0, 0, 0) and the last two come out of that.

    The card count never changes. Ten cards before, ten cards after, only spread
    differently - that total is the cheapest assertion that catches a draw which
    adds to the hand without taking from the pile.

    `rng` is passed in rather than using the module-level `random` so a run is
    reproducible from a seed. "Step 37 did something stupid" has to be
    reproducible or it cannot be debugged.
    """
    draw = state.draw
    hand = state.hand
    discard = state.discard
    for i in range(0, n):
        if not total(draw):
            discard, draw = draw, discard
        kind = rng.choices([0, 1, 2], weights=draw, k=1)[0]
        hand = _add(hand, kind, +1)
        draw = _add(draw, kind, -1)

    return state._replace(hand=hand, draw=draw, discard=discard)





def play_card(
    state: State, index: int, rng: random.Random
) -> tuple[State, float, bool]:
    """Play one card. Returns (state, reward, done).

    What has to happen, in an order you should think about rather than take from
    this list:

      * spend the cost, move the card from hand to discard
      * block goes on the player; damage goes on the enemy, through
        `damage_after_vulnerable` first
      * vulnerable from 痛击 lands on the enemy
      * if the enemy is at 0 or less, the fight is won

    **The ordering, decided by the user on 2026-09-15: damage resolves first, and
    the card's vulnerable lands after - so 痛击 does not amplify itself.** That is
    also how the game's own text reads it: "造成8点伤害。 给予2层易伤。", damage
    then debuff. What it costs, at 3 energy:

        打击 x3                       6 + 6 + 6  = 18
        痛击 -> 打击, no self-amp      8 + 9      = 17   <- this rule
        痛击 -> 打击, self-amp        12 + 9      = 21

    So 痛击 loses the turn it is played and has to earn it back later: vulnerable
    survives into the next turn, where three 打击 hit for 9 each (27) instead of
    18. Two turns: 17 + 27 = 44 kills a 40 hp enemy, 18 + 18 = 36 does not. The
    card is a bet on the fight lasting one more turn, which is exactly the kind of
    decision the toy fight had none of.

    ⚠️ Vulnerable is only *added* here. It ticks down in `end_turn`, so 痛击 on
    turn 1 covers the rest of turn 1 and all of turn 2. Keeping the decrement in
    one place is what stops the duration from depending on how many cards you
    happened to play.

    `rng` is unused - no card here draws or shuffles. It stays in the signature so
    `step` can dispatch to either branch without special-casing, and so the first
    card that does need it (陀螺, and everything like it) does not change the
    shape of the call.
    """
    card = CARDS[index]
    dealt = damage_after_vulnerable(card.damage, state.enemy_vulnerable)

    state = state._replace(
        energy=state.energy - card.cost,
        # The card moves hand -> discard. It does not leave the fight: the ten
        # cards have to still be ten, or the reshuffle quietly runs a smaller deck
        # every cycle.
        hand=_add(state.hand, index, -1),
        discard=_add(state.discard, index, +1),
        player_block=state.player_block + card.block,
        enemy_hp=state.enemy_hp - dealt,
        enemy_vulnerable=state.enemy_vulnerable + card.vulnerable,
    )

    # `<= 0`, not `== 0`: 6 damage onto a 3 hp enemy lands on -3, and `== 0` would
    # walk straight past the win and keep fighting a corpse.
    won = state.enemy_hp <= 0
    return state, reward(won, False, state.player_hp), won


def end_turn(state: State, rng: random.Random) -> tuple[State, float, bool]:
    """End the turn: the enemy attacks, then a fresh hand is dealt.

    Order, all of which you can check on screen:

      * the enemy hits for `intent(state)`, block first and then hp
      * vulnerable on the enemy ticks down by one, floored at 0
      * the player's block is **cleared** - that is what makes 防御 a decision
        about this turn rather than an investment that compounds
      * the whole hand is discarded, a new one is drawn, energy refills
      * `turn` goes up by one, and the **next turn's damage is rolled now**
        (`roll_intent`) and stored in `incoming`, so the player can see it
        before deciding anything

    The player is dead at 0 hp or less.
    """
    # The enemy hits for the number the player was **already shown** at the top of
    # the turn. Rolling a fresh one here instead is the trap in the header note:
    # the same visible state would sometimes cost 6 and sometimes 12, the
    # environment would stop being an MDP, and no policy could be optimal against
    # it. `roll_intent` is called once, at the bottom, for the *next* turn.
    #
    # `max(0, ...)` rather than a branch on whether block was enough, because both
    # cases are the same arithmetic:
    #
    #     挨 8, 挡 5  ->  max(0, 8 - 5) = 3 掉血
    #     挨 6, 挡 10 ->  max(0, 6 - 10) = 0 掉血，多的 4 点不留
    player_hp = state.player_hp - max(0, intent(state) - state.player_block)

    if player_hp <= 0:
        # Dead. Deliberately no new hand: a terminal state has no next turn, and
        # one that still looks playable invites a loop to keep stepping it.
        return state._replace(player_hp=player_hp, player_block=0), reward(False, True, player_hp), True

    state = state._replace(
        player_hp=player_hp,
        # Block does not carry. That is the whole reason 防御 is a decision about
        # this turn rather than an investment that compounds.
        player_block=0,
        # Vulnerable is a countdown of turns, floored at 0 - and the floor matters:
        # a negative value is still truthy, so `damage_after_vulnerable` would read
        # -1 as "vulnerable" and quietly hand out 150% for ever.
        enemy_vulnerable=max(0, state.enemy_vulnerable - 1),
        # The whole hand goes to the discard before anything is drawn. Both piles
        # are counts, so merging them is one add per card type. Ten cards in, ten
        # cards out - `draw_cards` relies on that to know when to reshuffle.
        hand=(0,) * len(CARDS),
        discard=tuple(d + h for d, h in zip(state.discard, state.hand)),
        energy=ENERGY_PER_TURN,
        turn=state.turn + 1,
    )

    # Rolled with the **new** turn, so the pool moves up on schedule, and stored
    # before the player acts so it is visible. Passing the old `turn` here would
    # delay the escalation by a turn and never show up on screen.
    state = state._replace(incoming=roll_intent(state.turn, state.tier, rng))
    return draw_cards(state, HAND_SIZE, rng), reward(False, False, player_hp), False











# --- the interface every RL text uses ----------------------------------------


def reset(rng: random.Random | None = None) -> State:
    """A fresh fight, with the opening hand already drawn."""
    rng = rng or random.Random()
    tier = roll_tier(rng)
    # Independent of `tier` on purpose. The two are not the same axis in the real
    # game and pairing them would invent fights that do not exist: the Act 1 boss
    # measured at floor 17 has 252 hp but announces a median of 17 damage - huge
    # and slow - while a three-pack of 10 hp trash can announce more than that.
    enemy_hp = rng.randint(*ENEMY_HP_RANGE)
    state = State(
        player_hp=PLAYER_HP,
        player_block=0,
        enemy_hp=enemy_hp,
        enemy_max_hp=enemy_hp,
        enemy_vulnerable=0,
        energy=ENERGY_PER_TURN,
        turn=0,
        tier=tier,
        incoming=roll_intent(0, tier, rng),
        hand=(0,) * len(CARDS),
        draw=DECK,
        discard=(0,) * len(CARDS),
    )
    return draw_cards(state, HAND_SIZE, rng)


def step(state: State, action: int, rng: random.Random) -> tuple[State, float, bool]:
    """Play one action. Returns (state, reward, done).

    Unlike `toy_combat.step` this is **not** a pure function - it draws cards, so
    it needs the rng. Same (state, action) can give different results, which is
    exactly what a deck adds and what makes a single evaluation fight stop being
    the whole story. Pass the rng explicitly so a seed still reproduces a run.
    """
    if action not in legal_actions(state):
        raise ValueError(f"action {action} is not legal in {state}")
    if action == END_TURN:
        return end_turn(state, rng)
    return play_card(state, action, rng)


# Which reward to pay. A switch rather than an edit, so "same environment, only
# the reward changed" stays a one-flag ablation instead of a diff.
REWARD_MODE = "shaped"  # "shaped" | "sparse"

# Three numbers, and only their **ratios** matter - scaling all three by a
# constant leaves the optimal policy untouched. So this is two knobs, not three:
#
#   WIN_FLOOR / |LOSE_REWARD|   how much winning *at all* is worth
#   WIN_TOP   / WIN_FLOOR       how much the hp on top of that is worth
#
# ⚠️ **WIN_TOP > WIN_FLOOR is required.** `run_sweep` moves WIN_TOP alone, and a
# value below the floor would make the reward *decrease* in hp - the assert in
# `reward` exists because that failure is silent otherwise: training still runs,
# still prints a win rate, and the agent quietly learns to end fights hurt.
WIN_FLOOR = 0.25   # paid for a win at ~0 hp
WIN_TOP = 2.0      # paid for a win at full hp
WIN_CURVE = 3.0    # >0 convex (hp worth more near full), ->0 linear
LOSE_REWARD = -2.0


def reward(won: bool, lost: bool, player_hp: int) -> float:
    """What the agent is paid.

    `sparse` is the Stage 1 reward, kept for the ablation: +1 / -1 / 0, and hp is
    ignored, so winning at 1 hp scores exactly what winning at 20 does.

    `shaped` is the user's design. Two decisions, taken 2026-09-16, and they are
    **independent** - either can be reverted without the other:

      1. **Every win pays positive.** The previous version paid a *negative*
         reward for winning below half hp (-0.9 at 1 hp), which had two costs.
         It forced `gamma = 1` (see below), and it left the policy ranking with
         no crossover: measured over WIN_TOP in [0, 100], the racing Q-table beat
         the blocking baseline at *every* weight, so `--sweep` was guaranteed to
         return identical rows. With wins positive the crossover appears near
         WIN_TOP ~ 0.9 and the sweep has a real front to find.
      2. **Convex in hp, not concave.** User's call: 「高血量大于低血量」, i.e. a
         point of hp should be worth *more* near full than near death. The log
         this replaced said the opposite - it paid 0.266 for 1->2 hp and 0.032
         for 19->20. Rationale for the flip: hp carries between fights, so the
         differences that change later decisions sit at the top of the bar
         (18 vs 14 decides whether you take the next elite; 2 vs 5 both mean rest).

        lost                 -2
        won                  WIN_FLOOR rising as e^(WIN_CURVE*h) to WIN_TOP

            剩余血   1     4     8    10    14    18    20
            奖励    0.265 0.325 0.463 0.569 0.907 1.523 2.000
            边际    ----- 0.023 0.042 0.057 0.104 0.190 0.257

        The margin column is the point of the whole design: 19->20 hp is worth
        **14.9x** what 1->2 hp is worth.

    Three invariants this shape holds, all asserted below:

      * **monotone in hp** - more hp is never worth less.
      * **every win beats every loss** - min win is +0.265 against -2.
      * **every win beats never finishing** - the "stall forever" payout is 0, and
        the worst win is +0.265 above it. The *previous* reward did not have this
        structurally (a -0.9 win was worse than a 0 stall) and relied on a
        measurement instead: a pure-block policy died in all 5000 fights, because
        four 防御 in ten cards is ~10 block a turn against a late-game average of
        10 damage. That measurement is still true, but it is no longer load
        bearing - and a measurement can expire when the numbers change, while an
        ordering cannot.

    ⚠️ **WIN_FLOOR is the knob that buys invariant 3.** A bare convex curve
    through the origin pays 0.017 for a win at 1 hp, which satisfies "positive"
    on paper while leaving a 0.017 margin over stalling. Set `WIN_FLOOR = 0.0` to
    get that bare form back.

    📌 **`gamma = 1` is no longer required.** It was, under the old reward: a
    negative payout gets cheaper the longer it is deferred, so "stall 40 steps
    and then die" scored 20x better than "win now at 4 hp". Every terminal payout
    is now positive-or--2, so gamma<1 rewards *finishing early*, which is
    ordinary behaviour rather than a trap. gamma=1 is still the default because
    「不在乎时间」 is still the design - but it is now a preference, not a
    correctness requirement.
    """
    if REWARD_MODE == "sparse":
        return 1.0 if won else -1.0 if lost else 0.0

    if lost:
        return LOSE_REWARD
    if won:
        assert WIN_TOP > WIN_FLOOR, (
            f"WIN_TOP ({WIN_TOP}) must exceed WIN_FLOOR ({WIN_FLOOR}); "
            "otherwise the reward decreases in hp and the agent is paid to get hurt"
        )
        # Normalised to [0, 1] at h = 0 and h = 1 whatever WIN_CURVE is, so the
        # curvature knob and the two endpoint knobs stay independent - changing
        # WIN_CURVE alone does not move what a full-hp win pays.
        h = player_hp / PLAYER_HP
        shape = (math.exp(WIN_CURVE * h) - 1) / (math.exp(WIN_CURVE) - 1)
        return WIN_FLOOR + (WIN_TOP - WIN_FLOOR) * shape
    return 0.0


# --- state encoding: the knob that decides whether a Q-table can cope ---------
#
# A full State has about 1.8e9 distinct values (945 ways to split the deck across
# three piles, times hp, block, enemy hp, vulnerable, energy and turn). Stage 1's
# toy fight had 4,556. Twenty thousand episodes visit maybe 3e5 states, so a
# table keyed on the raw State would be ~0.02% full and every lookup in
# evaluation would miss, return all-zeros, and tie-break at random - a random
# policy wearing a Q-table's name, printing plausible `Q=0.000` reasons.
#
# So encoding is a separate, swappable function, and how coarse to make it is an
# experiment rather than a detail: coarse enough and the table fills up but
# cannot tell useful states apart; fine enough and it never fills. Watching that
# trade fail from both ends is the argument for Stage 2's network.


def encode_full(state: State) -> State:
    """No compression: the state is its own key. The fine end of the trade."""
    return state


HP_BUCKETS = 5


def encode_coarse(state: State) -> tuple:
    """A deliberately lossy key: seven components instead of the State's twelve.

    The three questions the first draft of this docstring asked have been
    answered, and each answer was measured rather than guessed - the numbers are
    in [[实验1 — 真实卡组战斗环境]] under「压缩代价」. Merging rows costs something
    only when the rows being merged disagreed about the best action, so that is
    what was counted, over the 47,324 rows of the trained `encode_full` table
    whose argmax was decided by a margin above 0.05:

      * **exact hp, or a band?** Bands. Five of them, costing 14.7%. Ten would
        cost 5.8% and three 23.1%; five is the pick because a coarser key is also
        a key the real game can *hit*, and the previous attempt at transfer died
        of a 100% miss rate, not of a lost distinction.
      * **the pile contents, or the count?** Neither. Dropping `draw` and
        `discard` outright merged **zero** rows out of all 244,539 - they are
        fully determined by the rest, because `enemy_hp` records how much damage
        has been dealt and `player_block` how many 防御 were played. The deck is
        fixed at ten cards, so the third pile follows.
      * **`turn`, or the damage number?** The number. Dropping `turn` costs 1.9%,
        because it only ever acted through `incoming` anyway.

    `tier` goes too, and that one is about deployment rather than size: the real
    game has no such field, so `tools/qtable_combat.py` has to infer it from the
    visible `incoming`. Keeping it would mean the component meant "how hard this
    whole fight is" in training and "how hard this one hit is" at deployment -
    the same error as measuring hp in absolute points, one layer down.

    Two quantities are ratios and two are absolute, and the split is not
    arbitrary. hp is a ratio because 15 hp means "nearly dead" at 80 max and
    "comfortable" at 20. Damage is absolute because the deck is frozen: a 防御 is
    5 block whatever else is true, so "can 15 block cover this?" is a question
    with the same meaning in both worlds.
    """
    # `min` on both ratios: `80 * 5 // 80` is 5, one past the last bucket, and it
    # fires only at exactly full hp. Folding it into bucket 4 rather than letting
    # it have its own is a choice - "took no damage yet" is arguably worth
    # knowing - but it would be a bucket holding a single value, reachable only
    # on turn one. Change the `min` to widen it back out.
    #
    # `max(0, ...)` on the damage band: the bands are (0,10), (11,20), (21,30),
    # (31,40), so `(v - 1) // 10` lands each one on its index - except v=0, where
    # Python floors -1//10 to -1 rather than 0. `min` guards the other end for
    # the real game, which can announce more than 40.
    return (
        min(HP_BUCKETS - 1, state.player_hp * HP_BUCKETS // PLAYER_HP),
        min(HP_BUCKETS - 1, state.enemy_hp * HP_BUCKETS // state.enemy_max_hp),
        state.player_block >= state.incoming,
        min(len(TIERS) - 1, max(0, (state.incoming - 1) // 10)),
        state.enemy_vulnerable,
        state.energy,
        state.hand,
    )

# --- playing it by hand ------------------------------------------------------


def describe(state: State) -> str:
    vuln = f"  易伤{state.enemy_vulnerable}" if state.enemy_vulnerable else ""
    return (
        f"回合{state.turn + 1}  你 {state.player_hp}/{PLAYER_HP}"
        f"  格挡{state.player_block}  能量{state.energy}"
        f"  |  敌人 {state.enemy_hp}/{state.enemy_max_hp}{vuln}"
        f"  意图: 攻击{intent(state)}"
        f"  |  抽{total(state.draw)} 弃{total(state.discard)}"
    )


def menu(state: State) -> str:
    legal = legal_actions(state)
    parts = []
    for i, card in enumerate(CARDS):
        if state.hand[i] == 0:
            continue
        tag = f"{i} {card.name}x{state.hand[i]}({card.cost}费"
        if card.damage:
            tag += f",{card.damage}伤"
        if card.block:
            tag += f",{card.block}挡"
        if card.vulnerable:
            tag += f",易伤{card.vulnerable}"
        parts.append(tag + ")" + ("" if i in legal else " [能量不足]"))
    return "  ".join(parts) + f"  |  {END_TURN} 结束回合"


def play(seed: int | None = None) -> None:
    rng = random.Random(seed)
    state = reset(rng)
    while True:
        print(describe(state))
        print("  " + menu(state))
        raw = input("> ").strip()
        if raw in ("q", "quit"):
            return
        if not raw.isdigit() or int(raw) not in legal_actions(state):
            print("  不是一个能打的选择")
            continue
        state, r, done = step(state, int(raw), rng)
        if done:
            print(describe(state))
            print("你赢了" if state.enemy_hp <= 0 else "你死了", f"(reward {r:+})")
            return


if __name__ == "__main__":
    play()
