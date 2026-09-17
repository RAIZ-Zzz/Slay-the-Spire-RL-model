"""Tabular Q-learning on experiment 1's fight. The same algorithm as `qlearn.py`.

    python rl/stage1_tabular/qlearn_exp1.py                          # the pre-registered run
    python rl/stage1_tabular/qlearn_exp1.py --reward sparse          # the ablation: only the reward changes
    python rl/stage1_tabular/qlearn_exp1.py --encode coarse          # once `encode_coarse` is written

The learning rule is one line and it is the same one, copied from `qlearn.py:91`.
Everything else here exists because Stage 1 taught us that **Q-learning never
reports failure**: it always returns a table, a curve and a win rate, whether it
learned anything or not. `toy_combat` came back with 1.000 and a table whose rows
read `打击 0.656  重击 0.656  旁劈 0.656` - three identical numbers, so `max` tied
and every "decision" was `random.choice`. A constant policy wearing a Q-table's
name.

So four things are measured, and three of them are not the win rate:

  * **baselines**, hand-written and non-learning, to give the win rate a scale.
    94% means nothing until you know random gets 0.3% and four lines of `if` get
    95.4%.
  * **table hit rate**, counted with `in`. On a miss the row is all zeros, `max`
    ties across every legal action, and the step is a coin flip. At 8% hit rate
    92% of the "policy" is dice - and the win rate would still print.
  * **the incoming pair**, which is the pre-registered success criterion: two
    states differing in `incoming` alone, with different argmax. That is direct
    evidence of a state-dependent policy rather than an inference from a number.
  * **average reward**, because the shaped reward ranks fights the win rate
    cannot: winning at 2 hp and winning at 18 hp are both "a win".

📌 `gamma` defaults to **1.0**, not 0.9. This *used* to be load bearing: under
the old reward a win could pay a negative number, and a discount makes a negative
payout cheaper the longer it is deferred, so gamma<1 paid the agent to avoid
winning. Since 2026-09-16 every win pays positive (see `exp1_combat.reward`), so
gamma<1 merely rewards finishing early. 1.0 is still the default because
「不在乎时间」 is still the design - but it is a preference now, not a
correctness requirement, and `--gamma 0.99` is a legitimate thing to sweep.
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

# `exp1_combat` is the shared environment and stays in `rl/`. This file moved into
# `rl/stage1_tabular/` on 2026-09-17, so the parent has to go on the path first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import exp1_combat as env  # noqa: E402

STEP_CAP = 400  # a fight that has not ended by here is a bug, not a long fight


# --- policies ----------------------------------------------------------------
#
# A policy is a function (state, rng) -> action. The Q-table version and the
# hand-written ones have the same shape on purpose: they go through the same
# evaluation loop, so their numbers are comparable.


def p_random(state: env.State, rng: random.Random) -> int:
    return rng.choice(env.legal_actions(state))


def p_strike_only(state: env.State, rng: random.Random) -> int:
    """Never blocks, never sets up. The weakest thing that is not random."""
    legal = env.legal_actions(state)
    return env.STRIKE if env.STRIKE in legal else env.END_TURN


def p_attack(state: env.State, rng: random.Random) -> int:
    """Maximum aggression: 痛击 to set up, then 打击. Still never looks at hp."""
    legal = env.legal_actions(state)
    if env.BASH in legal and state.enemy_vulnerable == 0:
        return env.BASH
    if env.STRIKE in legal:
        return env.STRIKE
    return env.END_TURN


def p_rule(state: env.State, rng: random.Random) -> int:
    """Block until this turn's hit is covered, then attack. Looks at one field.

    This is the bar the Q-table has to clear: it is four lines and it reads
    exactly one number out of the state. Measured stronger than the "block if
    incoming > block" version that was pre-registered (95.4% vs 93.6%), so it is
    the one used - a baseline that is weaker than necessary flatters the agent.
    """
    legal = env.legal_actions(state)
    if state.player_block < state.incoming and env.DEFEND in legal:
        return env.DEFEND
    return p_attack(state, rng)


BASELINES = (
    ("随机", p_random),
    ("只打打击", p_strike_only),
    ("痛击优先", p_attack),
    ("挡够了才打", p_rule),
)


# --- the table ---------------------------------------------------------------


def greedy(q: dict, key, legal: list[int], rng: random.Random) -> tuple[int, bool]:
    """Best action according to the table, plus whether the row existed.

    `q.get`, never `q[key]`: `q` is a defaultdict, so a subscript **creates** the
    row. Measuring the hit rate with a subscript would turn every miss into a hit
    on the way past, and the metre would read 100% by construction.

    Ties break at random, which is also what a missed row does - all four zeros
    tie. The returned flag is the only thing that tells those two apart.
    """
    values = q.get(key)
    if values is None:
        return rng.choice(legal), False
    best = max(values[a] for a in legal)
    return rng.choice([a for a in legal if values[a] == best]), True


def epsilon_greedy(q, key, legal, epsilon, rng) -> int:
    if rng.random() < epsilon:
        return rng.choice(legal)
    return greedy(q, key, legal, rng)[0]


def train(encode, episodes=20000, alpha=0.1, gamma=1.0,
          eps_start=1.0, eps_end=0.05, seed=0):
    rng = random.Random(seed)
    q = defaultdict(lambda: [0.0] * (len(env.CARDS) + 1))
    curve, wins, block = [], 0, max(1, episodes // 40)

    for ep in range(episodes):
        epsilon = eps_start + (eps_end - eps_start) * (ep / episodes)
        state = env.reset(rng)

        for _ in range(STEP_CAP):
            key = encode(state)
            legal = env.legal_actions(state)
            action = epsilon_greedy(q, key, legal, epsilon, rng)
            next_state, r, done = env.step(state, action, rng)

            # Zero when the fight is over, because there is no "from there".
            # Getting this wrong credits a terminal state with an imaginary
            # future - and with a coarse encoding a dead enemy can share a row
            # with a live one, so `done` cannot be recovered from the state.
            future = 0.0 if done else max(
                q[encode(next_state)][a] for a in env.legal_actions(next_state)
            )

            q[key][action] += alpha * (r + gamma * future - q[key][action])

            state = next_state
            if done:
                wins += state.enemy_hp <= 0
                break

        if (ep + 1) % block == 0:
            curve.append(wins / block)
            wins = 0

    return q, curve


# --- measuring ---------------------------------------------------------------


def evaluate(choose, fights=2000, seed=999):
    """Run `choose` for `fights` fights. Returns the four numbers that matter.

    `choose(state, rng) -> (action, hit)`. Baselines report `hit=None`, which is
    honest: they have no table, so they have no hit rate, and reporting 0 or 1
    would invite comparing it with the agent's.
    """
    rng = random.Random(seed)
    wins = turns = hp_on_win = steps = hits = 0
    total_reward = 0.0

    for _ in range(fights):
        state = env.reset(rng)
        for _ in range(STEP_CAP):
            action, hit = choose(state, rng)
            steps += 1
            hits += bool(hit)
            state, r, done = env.step(state, action, rng)
            total_reward += r
            if done:
                break
        turns += state.turn + 1
        if state.enemy_hp <= 0:
            wins += 1
            hp_on_win += state.player_hp

    return {
        "win": wins / fights,
        "reward": total_reward / fights,
        "turns": turns / fights,
        "hp": hp_on_win / wins if wins else 0.0,
        "hit": hits / steps,
    }


def incoming_pairs(q, encode, limit=4):
    """The pre-registered criterion: same state, different `incoming`, different argmax.

    Only defined when the key *is* a State - a coarse encoding is opaque, and if
    it drops `incoming` entirely then no such pair can exist by construction,
    which is itself the finding.
    """
    if encode is not env.encode_full:
        return None

    # Every damage number the enemy can announce, across every tier. Was the two
    # fixed pools; since 2026-09-16 the bands are contiguous, so it is the range.
    values = list(range(env.TIERS[0][0], env.TIERS[-1][1] + 1))
    found, checked = [], 0
    for key, row in q.items():
        for v in values:
            if v <= key.incoming:
                continue
            other = key._replace(incoming=v)
            row2 = q.get(other)
            if row2 is None:
                continue
            checked += 1
            a1, m1 = _sole_argmax(row, env.legal_actions(key))
            a2, m2 = _sole_argmax(row2, env.legal_actions(other))
            # `is not None` on both: a tie is not a decision. An all-zero row -
            # which is what an unvisited or never-updated row looks like - ties
            # across every legal action, and `max` would hand back whichever
            # happens to be first. The first version of this check counted those,
            # and "found" four pairs in a table that had learned nothing: one row
            # was [0, 0, 0, 0] and the other [-0.02, 0, 0, 0]. The criterion has
            # to require a strict winner or it is satisfiable by noise.
            if a1 is not None and a2 is not None and a1 != a2:
                found.append((key, v, a1, a2, row, row2, min(m1, m2)))
    found.sort(key=lambda f: -f[-1])
    return found, checked


def _sole_argmax(row, legal):
    """The single best legal action and its margin, or (None, 0) on a tie.

    The margin is best-minus-runner-up, and it is not decoration. A unique argmax
    with a margin of 0.0004 is not a decision, it is the residue of two updates
    that happened to land differently - and printed at three decimals it looks
    exactly like a confident 0.0 vs 0.0. Report the number, then judge it.
    """
    if len(legal) < 2:
        # Only one thing is legal, so nothing was decided. Counting these as
        # evidence of a state-dependent policy would credit the agent for the
        # environment's constraints.
        return None, 0.0
    ranked = sorted((row[a] for a in legal), reverse=True)
    best = ranked[0]
    winners = [a for a in legal if row[a] == best]
    if len(winners) != 1:
        return None, 0.0
    return winners[0], best - ranked[1]


# --- reporting ---------------------------------------------------------------

NAMES = [c.name for c in env.CARDS] + ["结束回合"]


def run_sweep(values, args, encode) -> None:
    """Train once per WIN_TOP and lay the results out as a Pareto front.

    Roijers et al.'s survey is blunt about this: a scalarisation weight encodes a
    *preference*, so there is no correct value to derive - but there is a correct
    way to choose one. Solve the scalarised problem at several weights, plot the
    objectives you actually care about, and pick a point off the front. That is
    what this does, with 胜率 and 赢时剩血 as the two objectives.

    Two things it deliberately does not do:

      * **compare 平均奖励 across rows.** Each row is scored by its own reward
        function, so those numbers live on different scales and ranking them
        would be meaningless. The two objective columns are the comparable ones.
      * **re-measure the baselines per row.** The hand-written policies never read
        the reward, so their 胜率 and 剩血 are identical at every WIN_TOP. They are
        measured once and printed as fixed reference points on the same front.
    """
    print(f"\n扫 WIN_TOP = {values}，每个训 {args.episodes:,} 局（seed 固定 {args.seed}，"
          f"所以行与行的差别只来自 WIN_TOP）\n")

    print(f"{'参考点（不随奖励变）':>22} {'胜率':>7} {'赢时剩血':>9} {'回合':>6}")
    fixed = []
    for name, policy in BASELINES:
        s = evaluate(lambda st, rng, p=policy: (p(st, rng), None), args.fights)
        fixed.append((name, s))
        print(f"{name:>22} {s['win']:7.1%} {s['hp']:9.1f} {s['turns']:6.1f}")

    rows = []
    for top in values:
        env.WIN_TOP = top
        q, curve = train(encode, args.episodes, args.alpha, args.gamma, seed=args.seed)
        s = evaluate(
            lambda st, rng: greedy(q, encode(st), env.legal_actions(st), rng),
            args.fights,
        )
        rows.append((top, s, len(q), curve[-1] if curve else float("nan")))
        if args.save:
            stem = args.save[:-4] if args.save.endswith(".pkl") else args.save
            with open(f"{stem}_top{top:g}.pkl", "wb") as fh:
                pickle.dump((dict(q), curve), fh)
        print(f"  WIN_TOP={top:<5g} 训完：胜率 {s['win']:.1%}  剩血 {s['hp']:.1f}")

    print(f"\n{'WIN_TOP':>8} {'胜率':>7} {'赢时剩血':>9} {'战损':>6} {'回合':>6} "
          f"{'命中率':>8} {'表行数':>10} {'曲线末值':>9} {'被支配':>7}")
    for top, s, nrows, tail in rows:
        # Pareto-dominated = some other row is at least as good on both objectives
        # and strictly better on one. Those points are never the right pick, no
        # matter how you weigh 胜率 against 剩血.
        dominated = any(
            (o["win"] >= s["win"] and o["hp"] >= s["hp"])
            and (o["win"] > s["win"] or o["hp"] > s["hp"])
            for t2, o, _, _ in rows if t2 != top
        )
        print(f"{top:8g} {s['win']:7.1%} {s['hp']:9.1f} "
              f"{env.PLAYER_HP - s['hp']:6.1f} {s['turns']:6.1f} {s['hit']:8.1%} "
              f"{nrows:10,} {tail:9.2f} {'是' if dominated else '':>7}")

    print("\n⚠️ 曲线末值明显低于胜率那一列 = 还没收敛，这一行的位置不可信，"
          "不要拿它去比 —— 先加局数。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--encode", choices=("full", "coarse"), default="full")
    ap.add_argument("--reward", choices=("shaped", "sparse"), default="shaped")
    ap.add_argument("--fights", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="训练完把表 pickle 到这个路径，省得重训")
    ap.add_argument("--load", help="读一张训练好的表，跳过训练")
    ap.add_argument("--sweep", help="逗号分隔的 WIN_TOP 列表，每个值训一次，"
                                    "最后并排成 Pareto 前沿。例：--sweep 0.5,1,2,4")
    args = ap.parse_args()

    env.REWARD_MODE = args.reward
    encode = env.encode_full if args.encode == "full" else env.encode_coarse

    print(f"环境 玩家{env.PLAYER_HP} / 敌人{env.ENEMY_HP_RANGE}   奖励 {args.reward}   "
          f"编码 {args.encode}   gamma {args.gamma}   alpha {args.alpha}   "
          f"{args.episodes} 局训练   输 {env.LOSE_REWARD:+g}")

    if args.sweep:
        run_sweep([float(v) for v in args.sweep.split(",")], args, encode)
        return

    if args.load:
        with open(args.load, "rb") as fh:
            q, curve = pickle.load(fh)
        print(f"（读的是 {args.load}，没有重新训练）")
    else:
        q, curve = train(encode, args.episodes, args.alpha, args.gamma, seed=args.seed)
        if args.save:
            with open(args.save, "wb") as fh:
                pickle.dump((dict(q), curve), fh)

    print("\n学习曲线（训练中的胜率，带探索）")
    for i, rate in enumerate(curve):
        print(f"  {(i + 1) * max(1, args.episodes // 40):7} {rate:5.2f}  "
              + "#" * int(rate * 40))

    print(f"\n{'策略':>12} {'胜率':>7} {'平均奖励':>9} {'赢时剩血':>9} "
          f"{'回合':>6} {'查表命中率':>11}")
    for name, policy in BASELINES:
        s = evaluate(lambda st, rng, p=policy: (p(st, rng), None), args.fights)
        print(f"{name:>12} {s['win']:7.1%} {s['reward']:9.3f} {s['hp']:9.1f} "
              f"{s['turns']:6.1f} {'—':>11}")

    stats = evaluate(
        lambda st, rng: greedy(q, encode(st), env.legal_actions(st), rng),
        args.fights,
    )
    print(f"{'Q-learning':>12} {stats['win']:7.1%} {stats['reward']:9.3f} "
          f"{stats['hp']:9.1f} {stats['turns']:6.1f} {stats['hit']:11.1%}")
    print(f"\n表里有 {len(q):,} 行")

    pairs = incoming_pairs(q, encode)
    if pairs is None:
        print("成功判据：编码不是 State，无法逐字段构造配对（这本身是个结果）")
    else:
        found, checked = pairs
        print(f"\n成功判据：只差 incoming 的状态对 {checked:,} 组，"
              f"argmax 不同的 {len(found):,} 组")
        for thr in (0.01, 0.05, 0.10):
            n = sum(1 for f in found if f[-1] > thr)
            print(f"    其中两边的胜出边际都 > {thr:.2f} 的：{n:,}")
        for key, v, a1, a2, row, row2, margin in found[:4]:
            print(f"  边际 {margin:.4f}   血{key.player_hp} 挡{key.player_block} "
                  f"敌{key.enemy_hp} 能量{key.energy} 手牌{key.hand}")
            print(f"    incoming={key.incoming:>2} -> {NAMES[a1]:<5}"
                  f" {['%.4f' % x for x in row]}")
            print(f"    incoming={v:>2} -> {NAMES[a2]:<5}"
                  f" {['%.4f' % x for x in row2]}")


if __name__ == "__main__":
    main()
