"""A tiny hand-written fight. Stage 1's environment, playable from the keyboard.

Nothing here touches the game or the bridge. That is the whole point: RL needs
10^5-10^6 steps and the real game runs at about one step per second, so the brain
has to be grown somewhere that runs millions of fights a second and then carried
over. The interface is deliberately the one every RL text uses:

    state = reset()
    state, reward, done = step(state, action)

`step` is a pure function - given the same (state, action) it returns the same
thing, with no module-level state to forget to reset. Same shape as
`wilson_interval` in tools/analyze_runs.py, and it is what lets a Q-table key on
the state directly.

Run it and play a fight yourself:

    python rl/stage1_tabular/toy_combat.py

Play two or three before writing any learning code. The numbers below are the
ones from the roadmap and they are **degenerate** - see the note on them.
"""
from __future__ import annotations

import random


# --- the numbers, which are yours to set -------------------------------------
#
# ⚠️ As written, straight from the roadmap, this fight has nothing to learn:
#
#     best damage per turn = 旁劈 x3   = 24
#     turns to kill        = ceil(40/24) = 2
#     damage taken in 2 turns          = 12 + 12 = 24
#     player 50 - 24                   = 26 left, never close to dying
#
# So 防御 is never worth a card, 打击 is strictly worse than 旁劈 at the same
# cost, and the fight ends before the third intent ever happens. The optimal
# policy is the constant "always 旁劈", and a constant policy is one a Q-table
# learns in about four updates while teaching nothing about states.
#
# The test for a candidate set of numbers: over `ceil(enemy_hp / best damage per
# turn)` turns, does the damage you take come close to your hp? If not, pure
# aggression is free and blocking is dead.
PLAYER_HP = 50
ENEMY_HP = 40
ENERGY_PER_TURN = 3

# The enemy repeats this forever. ("attack", n) hits you for n; ("block", n)
# gives the enemy n block, which absorbs damage before its hp.
INTENTS = [("attack", 12), ("attack", 12), ("block", 8)]

# name: (cost, damage, block)
CARDS = [
    ("打击", 1, 6, 0),
    ("防御", 1, 0, 5),
    ("重击", 2, 14, 0),
    ("旁劈", 1, 8, 0),
]
END_TURN = len(CARDS)

# --- the state ---------------------------------------------------------------
#
# A tuple, because a Q-table is a dict keyed on the state and dicts need
# something hashable. Six numbers:
#
#     (player_hp, player_block, enemy_hp, enemy_block, intent_index, energy)
#
# ⚠️ Two of these are design decisions, not details, and both are yours:
#
#   * `player_block` is in here so that playing 防御 changes the state. Leave it
#     out and the card's effect is invisible to the learner - it would see the
#     same state before and after, and could never learn when to use it.
#   * `intent_index` is in here so the agent can see what is coming. Leave it out
#     and the environment stops being an MDP: the same visible state would
#     sometimes be followed by 12 damage and sometimes by none, and no policy can
#     be optimal against that.


def reset() -> tuple:
    return (PLAYER_HP, 0, ENEMY_HP, 0, 0, ENERGY_PER_TURN)


def legal_actions(state: tuple) -> list[int]:
    """Which actions the state will accept. Ending the turn is always allowed.

    Separated out for the same reason the bridge work needed `legal_verbs`: an
    agent offered an action the state cannot take will pick it, and then either
    nothing happens or something silently wrong does.
    """
    *_, energy = state
    return [i for i, (_, cost, _, _) in enumerate(CARDS) if cost <= energy] + [END_TURN]


def _deal(amount: int, hp: int, block: int) -> tuple[int, int]:
    """Damage eats block first, then hp. Returns the new (hp, block)."""
    absorbed = min(amount, block)
    return hp - (amount - absorbed), block - absorbed


def step(state: tuple, action: int) -> tuple[tuple, float, bool]:
    """Play one card, or end the turn. Returns (state, reward, done)."""
    php, pblock, ehp, eblock, intent_i, energy = state

    if action not in legal_actions(state):
        raise ValueError(f"action {action} is not legal in {state}")

    if action != END_TURN:
        _, cost, damage, block = CARDS[action]
        energy -= cost
        pblock += block
        if damage:
            ehp, eblock = _deal(damage, ehp, eblock)
        if ehp <= 0:
            return (php, pblock, ehp, eblock, intent_i, energy), reward(True, False), True
        return (php, pblock, ehp, eblock, intent_i, energy), reward(False, False), False

    # Ending the turn: the enemy acts, then a fresh turn starts.
    #
    # Block is cleared at the start of your next turn, not kept - that is what
    # makes 防御 a decision about *this* turn rather than a stacking investment.
    kind, amount = INTENTS[intent_i]
    if kind == "attack":
        php, pblock = _deal(amount, php, pblock)
    else:
        eblock += amount

    intent_i = (intent_i + 1) % len(INTENTS)
    done = php <= 0
    return (php, 0, ehp, eblock, intent_i, ENERGY_PER_TURN), reward(False, done), done


def reward(won: bool, lost: bool) -> float:
    """What the agent is paid. **This is the exercise, and it is yours.**

    Two designs the roadmap names, which produce different policies:

      * +1 for winning, -1 for losing, 0 otherwise. Sparse: most steps say
        nothing, so learning is slower, but nothing can be gamed.
      * something every turn, e.g. the damage dealt. Dense and faster - and the
        trap: pay for damage without paying for *ending* the fight and the best
        policy is to keep the enemy barely alive and farm it forever. That is
        reward hacking, and building one on purpose is worth more than reading
        about ten.

    Left as the sparse version so the file runs. Change it, run both, and look at
    the policies rather than only at the learning curves.
    """
    if won:
        return 1.0
    if lost:
        return -1.0
    return 0.0


# --- playing it by hand ------------------------------------------------------

def describe(state: tuple) -> str:
    php, pblock, ehp, eblock, intent_i, energy = state
    kind, amount = INTENTS[intent_i]
    intent = f"攻击{amount}" if kind == "attack" else f"防御+{amount}"
    return (f"你 {php}/{PLAYER_HP}  格挡{pblock}  能量{energy}"
            f"  |  敌人 {ehp}/{ENEMY_HP}  格挡{eblock}  意图: {intent}")


def menu(state: tuple) -> str:
    parts = []
    for i, (name, cost, damage, block) in enumerate(CARDS):
        tag = f"{i} {name}({cost}费"
        if damage:
            tag += f",{damage}伤"
        if block:
            tag += f",{block}挡"
        parts.append(tag + ")" + ("" if i in legal_actions(state) else " [能量不足]"))
    return "  ".join(parts) + f"  |  {END_TURN} 结束回合"


def play(seed: int | None = None) -> None:
    random.seed(seed)
    state = reset()
    turn = 1
    print(f"--- 第 {turn} 回合 ---")
    while True:
        print(describe(state))
        print("  " + menu(state))
        raw = input("> ").strip()
        if raw in ("q", "quit"):
            return
        if not raw.isdigit() or int(raw) not in legal_actions(state):
            print("  不是一个能打的选择")
            continue

        action = int(raw)
        state, r, done = step(state, action)
        if done:
            print(describe(state))
            print("你赢了" if state[2] <= 0 else "你死了", f"(reward {r:+})")
            return
        if action == END_TURN:
            turn += 1
            print(f"--- 第 {turn} 回合 ---")


if __name__ == "__main__":
    play()
