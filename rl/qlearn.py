"""Tabular Q-learning on the toy fight. Stage 1's actual content.

    python rl/qlearn.py                 # train, print the curve and the table
    python rl/qlearn.py --gamma 1.0     # watch the values blow up

Everything here is scaffolding except one line in `train`, marked TODO. That line
is Q-learning; the rest is bookkeeping around it.

No neural network, no library, no GPU. A dict from state to a list of four
numbers, updated in place. The table does not generalise at all - one cell per
state, memorising each one - which is why nothing here can overfit, and also why
it is worthless against any enemy but this one. Making that trade differently is
Stage 2's job.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

import toy_combat as env


def greedy(q: dict, state: tuple, legal: list[int]) -> int:
    """The action the table currently thinks is best. Ties broken at random.

    Randomly, not by taking the first: with a table that starts all zeros, every
    action ties on the first visit, and always picking index 0 would explore
    exactly one action per state for ever.
    """
    values = q[state]
    best = max(values[a] for a in legal)
    return random.choice([a for a in legal if values[a] == best])


def epsilon_greedy(q: dict, state: tuple, legal: list[int], epsilon: float) -> int:
    """Explore with probability epsilon, otherwise take the current best.

    Without this the agent commits to whatever it stumbled on first: the first
    action to get a positive value is the only one that ever gets picked again,
    so the others keep their initial 0 for ever and are never compared.
    """
    if random.random() < epsilon:
        return random.choice(legal)
    return greedy(q, state, legal)


def train(episodes: int = 20000, alpha: float = 0.1, gamma: float = 0.9,
          eps_start: float = 1.0, eps_end: float = 0.05,
          seed: int = 0) -> tuple[dict, list[float]]:
    """Play `episodes` fights, updating the table after every action.

    Returns the table and the win rate measured in blocks of 500 fights, which
    is the learning curve.
    """
    random.seed(seed)
    q = defaultdict(lambda: [0.0] * (len(env.CARDS) + 1))
    curve, wins, block = [], 0, 500

    for ep in range(episodes):
        # Exploration decays from eps_start to eps_end across the whole run.
        # Held at eps_end rather than 0 on purpose - the roadmap's second pitfall
        # is an epsilon that never decays, and the first is one that hits zero
        # early and locks in whatever it believed at the time.
        epsilon = eps_start + (eps_end - eps_start) * (ep / episodes)

        state = env.reset()
        while True:
            legal = env.legal_actions(state)
            action = epsilon_greedy(q, state, legal, epsilon)
            next_state, r, done = env.step(state, action)

            # The value of the state we land in: the best we think we can do from
            # there. Zero when the fight is over, because there is no "from
            # there" - and getting that wrong is how a value function ends up
            # crediting a terminal state with an imaginary future.
            future = 0.0 if done else max(q[next_state][a]
                                          for a in env.legal_actions(next_state))

            # ------------------------------------------------------------------
            # TODO(you): one line. This is Q-learning.
            #
            #     Q[s][a]  <-  Q[s][a] + alpha * ( r + gamma * future - Q[s][a] )
            #
            # In this code: `q[state][action]`, `alpha`, `r`, `gamma`, `future`.
            #
            # The bracket is the error - "how much better things turned out than
            # I thought" - and alpha is how much of that error to believe. Write
            # it, then check one update by hand against what the table prints.
            # ------------------------------------------------------------------
            q[state][action] += alpha * (r + gamma * future - q[state][action])

            state = next_state
            if done:
                wins += 1 if state[2] <= 0 else 0
                break

        if (ep + 1) % block == 0:
            curve.append(wins / block)
            wins = 0

    return q, curve


def evaluate(q: dict, fights: int = 1000, seed: int = 999) -> tuple[float, int]:
    """Win rate with no exploration at all, plus the average fight length.

    Separate from training because the training win rate is dragged down by the
    exploration: a curve that plateaus at 0.7 with epsilon=0.05 can still be a
    policy that wins every time when it stops rolling dice.
    """
    random.seed(seed)
    wins, turns = 0, 0
    for _ in range(fights):
        state = env.reset()
        steps = 0
        while True:
            action = greedy(q, state, env.legal_actions(state))
            state, _, done = env.step(state, action)
            steps += 1
            if done:
                wins += 1 if state[2] <= 0 else 0
                turns += steps
                break
            if steps > 500:      # a policy that never ends the fight
                turns += steps
                break
    return wins / fights, turns // fights


def show_policy(q: dict, limit: int = 12) -> None:
    """The table, for the states the greedy policy actually walks through.

    Printing all of it is useless - most cells were visited once by an
    exploration roll and mean nothing. These are the ones the policy uses.
    """
    names = [c[0] for c in env.CARDS] + ["结束回合"]
    state = env.reset()
    print(f"\n{'state':38} " + "  ".join(f"{n:>6}" for n in names))
    for _ in range(limit):
        values = q[state]
        legal = env.legal_actions(state)
        row = "  ".join(f"{values[a]:6.3f}" if a in legal else "     ."
                        for a in range(len(names)))
        action = greedy(q, state, legal)
        print(f"{str(state):38} {row}   -> {names[action]}")
        state, _, done = env.step(state, action)
        if done:
            print(f"{str(state):38} {'(打完了)' if state[2] <= 0 else '(死了)'}")
            break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    q, curve = train(args.episodes, args.alpha, args.gamma, seed=args.seed)

    print(f"episodes={args.episodes}  alpha={args.alpha}  gamma={args.gamma}")
    print("\n学习曲线（每 500 局的胜率）")
    for i, rate in enumerate(curve):
        bar = "#" * int(rate * 40)
        print(f"  {(i + 1) * 500:6}  {rate:5.2f}  {bar}")

    rate, length = evaluate(q)
    print(f"\n不探索时：胜率 {rate:.3f}，平均 {length} 步")
    print(f"表里有 {len(q)} 个状态")
    show_policy(q)


if __name__ == "__main__":
    main()
