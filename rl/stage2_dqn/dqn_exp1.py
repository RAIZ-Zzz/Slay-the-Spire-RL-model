"""Stage 2: the same fight, the same evaluation, the Q-table replaced by a network.

    pip install torch --index-url https://download.pytorch.org/whl/cpu

    python rl/stage2_dqn/dqn_exp1.py --check                   # no training: does the wiring hold up
    python rl/stage2_dqn/dqn_exp1.py --variant naive           # no replay, no target net - watch it thrash
    python rl/stage2_dqn/dqn_exp1.py --variant replay          # + replay buffer
    python rl/stage2_dqn/dqn_exp1.py --variant dqn             # + target network
    python rl/stage2_dqn/dqn_exp1.py --variant dqn --raw-obs   # the observation-scale pit, on purpose
    python rl/stage2_dqn/dqn_exp1.py --plot                    # every saved curve on one figure

**The environment does not change.** That is the whole design of this stage: the
fight, the reward, the baselines and the evaluation loop are imported from Stage
1 rather than rewritten, so anything that goes wrong is the algorithm's fault and
not the environment's. `evaluate` and `BASELINES` come straight out of
`qlearn_exp1`, which means the numbers printed here sit on the same scale as the
ones already measured:

    随机          36.9%
    挡够了才打     64.8%
    只打打击      71.4%
    表格 Q        78.8%   <- 1.6M episodes, sparse reward, 100% table hit rate
    痛击优先      82.2%   <- four lines of `if`, and the ceiling of this fight

So this stage has a **known answer**. A DQN landing near 78% is working; one
landing at 45% has a bug, and there is no need to wonder whether the environment
is at fault. Stage 1 did not have that luxury and it cost a day.

⚠️ Do not expect the network to beat 82.2%. Measured on 2026-09-16: this fight
has almost no decision content - across 20 (tier, enemy hp) cells and three
hand-written policies, there is **no cell where blocking wins**, because the deck
is frozen so damage output is constant and a 打击 saves more than a 防御 blocks.
The point of Stage 2 is not a better score. It is PyTorch fluency and the three
curves below, on an environment whose answer is already pinned down.

--- what is deliberately left undone -----------------------------------------

Two functions raise `NotImplementedError` and they are the two that matter:

  * `build_net`   - the 2-layer MLP
  * `td_target`   - the one line that is the whole algorithm, and the same line
                    you hand-computed for the table on 2026-09-14

Everything else here is plumbing: the replay buffer, the epsilon schedule, the
evaluation, the curve bookkeeping, the CLI. Those are worth having written for
you; those two are not.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

# Both of these live outside this folder since the 2026-09-17 reshuffle:
#   exp1_combat  -> rl/                 the shared environment, unchanged from Stage 1
#   qlearn_exp1  -> rl/stage1_tabular/  the five baselines this run is measured against
_RL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RL))
sys.path.insert(0, str(_RL / "stage1_tabular"))

import exp1_combat as env  # noqa: E402
import qlearn_exp1 as tabular  # noqa: E402

CURVE_DIR = Path(__file__).resolve().parent / "curves"
STEP_CAP = tabular.STEP_CAP


def _require_torch():
    """Import torch with a message that says what to run, not just what failed."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise SystemExit(
            "PyTorch is not installed. For this environment the CPU wheel is the "
            "right one - the state is 18 floats and the net is two layers, so a "
            "GPU spends more time on transfer than on arithmetic:\n\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        ) from exc
    import torch
    return torch


# --- observation ---------------------------------------------------------------
#
# The table keyed on a tuple and looked it up; a network takes a fixed-length
# vector of floats and computes. That difference is the whole stage, and two of
# its consequences show up before any training happens:
#
#   * every component is shared by every state, so an update for one state moves
#     the answer for states never visited. That is the generalisation a table
#     cannot do, and also why a naive DQN thrashes.
#   * the size of a component decides how big a step it takes. Raw hp is 0-80 and
#     energy is 0-3, so the hp component's gradient is ~27x larger and the net
#     follows hp and ignores energy. `--raw-obs` turns normalisation off so that
#     can be watched rather than believed.

OBS_NAMES = (
    "player_hp_frac",     # 0-1   血 / 最大血
    "block_frac",         # 0-1   格挡 / 最大血
    "block_covers",       # 0/1   挡得住这一下吗 - the one feature the tabular
                          #       encoding proved useful, kept explicit because a
                          #       2-layer net learning `a >= b` from two inputs is
                          #       a waste of capacity
    "enemy_hp_frac",      # 0-1   敌人血 / 敌人最大血
    "fight_size",         # 0-1   敌人最大血 / 150, i.e. how long this fight is
    "vulnerable",         # 0-1   易伤层数 / 3
    "energy",             # 0-1   能量 / 3
    "incoming",           # 0-1   这一下的伤害 / 40
    "incoming_vs_hp",     # 0-1   这一下占当前血的比例, capped - "会不会打死我"
    "hand_strike",        # 0-1   手牌张数 / 5
    "hand_defend",
    "hand_bash",
    "draw_strike",        # 0-1   抽牌堆张数 / 5, capped
    "draw_defend",
    "draw_bash",
    "discard_strike",
    "discard_defend",
    "discard_bash",
    "turn",               # 0-1   回合 / 20, capped
)
OBS_DIM = len(OBS_NAMES)
N_ACTIONS = len(env.CARDS) + 1


def encode_vector(state: env.State, normalize: bool = True,
                  cap: bool = True) -> list[float]:
    """The State as a fixed-length vector of floats.

    `draw` and `discard` are included even though the tabular version measured
    them as fully redundant - dropping them merged zero rows out of 244,539,
    because enemy hp and block already pin down what has been played. They stay
    because the redundancy was a property of a frozen ten-card deck and this
    encoding is meant to survive Stage 3, and because a network can learn to
    ignore a feature while a table cannot ignore a key. If the trained net turns
    out to put ~0 weight on those six, that is a result worth reporting.
    """
    hp_max = env.PLAYER_HP
    e_max = max(1, state.enemy_max_hp)
    raw = [
        state.player_hp,
        state.player_block,
        float(state.player_block >= state.incoming),
        state.enemy_hp,
        state.enemy_max_hp,
        state.enemy_vulnerable,
        state.energy,
        state.incoming,
        state.incoming / max(1, state.player_hp),
        *state.hand,
        *state.draw,
        *state.discard,
        state.turn,
    ]
    if not normalize:
        return [float(x) for x in raw]

    scales = [hp_max, hp_max, 1.0, e_max, env.ENEMY_HP_RANGE[1], 3.0, 3.0,
              env.TIERS[-1][1], 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
              20.0]
    assert len(raw) == len(scales) == OBS_DIM, (len(raw), len(scales), OBS_DIM)
    # `cap=False` is only for the wiring check, which needs to see how often a
    # component would have gone over 1 - a capped value and a value that happens
    # to land on 1.0 are indistinguishable afterwards.
    if not cap:
        return [x / s for x, s in zip(raw, scales)]
    return [min(1.0, x / s) for x, s in zip(raw, scales)]


# --- replay --------------------------------------------------------------------


class Replay:
    """A ring buffer of transitions, sampled uniformly.

    It exists to break the correlation between consecutive samples. Without it a
    network is trained on a walk through one fight, so every batch looks like the
    last one and the weights chase whatever is happening right now - which the
    linear demo on 2026-09-16 showed as an oscillation between 0.17 and 13.44
    with the targets held constant.

    Size is a knob worth abusing: too small and it is barely a buffer, too large
    and it is still learning from a policy it has long since abandoned.
    """

    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done, next_legal):
        self.buf.append((obs, action, reward, next_obs, done, next_legal))

    def sample(self, batch: int, rng: random.Random):
        return [self.buf[rng.randrange(len(self.buf))] for _ in range(batch)]

    def __len__(self):
        return len(self.buf)


# --- ⬜ the two pieces that are yours -------------------------------------------


def build_net(torch, obs_dim: int, n_actions: int, hidden: int = 64):
    return torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, n_actions))


def td_target(torch, reward, next_q_row, done, gamma, next_legal):
    if done:
        return reward
    else:
        return reward + gamma * max(next_q_row[i] for i in next_legal)


# --- training ------------------------------------------------------------------

VARIANTS = {
    # name -> (use_replay, use_target_net). Run all three and plot them together:
    # the shape of the three curves *is* the answer to "why does DQN look like
    # this", and it is the one part of this stage the roadmap calls out as the
    # most valuable.
    "naive": (False, False),
    "replay": (True, False),
    "dqn": (True, True),
}


def train(args) -> tuple[list[float], object]:
    torch = _require_torch()
    use_replay, use_target = VARIANTS[args.variant]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    net = build_net(torch, OBS_DIM, N_ACTIONS, args.hidden)
    target_net = build_net(torch, OBS_DIM, N_ACTIONS, args.hidden) if use_target else net
    if use_target:
        target_net.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    replay = Replay(args.replay_size) if use_replay else None
    curve, wins, block = [], 0, max(1, args.episodes // 40)
    updates = 0

    def obs_of(s):
        return encode_vector(s, normalize=not args.raw_obs)

    for ep in range(args.episodes):
        epsilon = args.eps_start + (args.eps_end - args.eps_start) * (ep / args.episodes)
        state = env.reset(rng)

        for _ in range(STEP_CAP):
            legal = env.legal_actions(state)
            obs = obs_of(state)
            if rng.random() < epsilon:
                action = rng.choice(legal)
            else:
                with torch.no_grad():
                    row = net(torch.tensor([obs], dtype=torch.float32))[0]
                action = max(legal, key=lambda a: row[a].item())

            next_state, reward, done = env.step(state, action, rng)
            next_legal = env.legal_actions(next_state)
            next_obs = obs_of(next_state)

            batch = [(obs, action, reward, next_obs, done, next_legal)]
            if replay is not None:
                replay.push(*batch[0])
                if len(replay) < args.batch:
                    state = next_state
                    if done:
                        wins += state.enemy_hp <= 0
                        break
                    continue
                batch = replay.sample(args.batch, rng)

            obs_b = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            next_b = torch.tensor([b[3] for b in batch], dtype=torch.float32)
            with torch.no_grad():
                next_rows = target_net(next_b)
            targets = torch.tensor(
                [td_target(torch, b[2], next_rows[i], b[4], args.gamma, b[5])
                 for i, b in enumerate(batch)],
                dtype=torch.float32)

            q = net(obs_b)
            chosen = q[range(len(batch)), [b[1] for b in batch]]
            loss = torch.nn.functional.mse_loss(chosen, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1

            if use_target and updates % args.target_sync == 0:
                target_net.load_state_dict(net.state_dict())

            state = next_state
            if done:
                wins += state.enemy_hp <= 0
                break

        if (ep + 1) % block == 0:
            curve.append(wins / block)
            wins = 0

    return curve, net


# --- measuring -----------------------------------------------------------------


def greedy_of(torch, net, raw_obs: bool):
    """Wrap the net as the `(state, rng) -> (action, hit)` shape `evaluate` wants.

    `hit` is always None, not True. A table can miss - that was the number that
    decided whether Stage 1's policy was a policy at all - but a network always
    answers, so a hit rate here would be 100% by construction and would invite
    comparing it with the table's. The absence of that metric *is* the difference
    between the two approaches, and printing a fake one would hide it.
    """
    def choose(state, rng):
        legal = env.legal_actions(state)
        with torch.no_grad():
            row = net(torch.tensor([encode_vector(state, not raw_obs)],
                                   dtype=torch.float32))[0]
        best = max(row[a].item() for a in legal)
        return rng.choice([a for a in legal if row[a].item() == best]), None
    return choose


def report(args, curve, net) -> dict:
    """Print the comparison table. Returns the measured win rates for the plot.

    The return value is what lets `plot_all` draw the no-learning reference: the
    training curve is measured *with* exploration, so it climbs on its own as
    epsilon decays even when the policy never improves. Separating those two
    needs the random win rate (where epsilon=1 lands) and the greedy win rate
    (where epsilon=0 lands), and both are measured right here.
    """
    torch = _require_torch()
    print(f"\n学习曲线（训练中的胜率，带探索）  variant={args.variant}")
    for i, rate in enumerate(curve):
        print(f"  {(i + 1) * max(1, args.episodes // 40):8} {rate:5.2f}  "
              + "#" * int(rate * 40))

    measured = {}
    print(f"\n{'策略':>12} {'胜率':>8} {'平均奖励':>10} {'赢时剩血':>10} {'回合':>7}")
    for name, policy in tabular.BASELINES:
        s = tabular.evaluate(lambda st, rng, p=policy: (p(st, rng), None), args.fights)
        measured[name] = s["win"]
        print(f"{name:>12} {s['win']:8.1%} {s['reward']:10.3f} {s['hp']:10.1f} "
              f"{s['turns']:7.1f}")

    s = tabular.evaluate(greedy_of(torch, net, args.raw_obs), args.fights)
    measured["DQN"] = s["win"]
    print(f"{'DQN':>12} {s['win']:8.1%} {s['reward']:10.3f} {s['hp']:10.1f} "
          f"{s['turns']:7.1f}")
    print("\n参考（2026-09-16 实测）：表格 Q-learning 78.8%，"
          "本环境上限约 82.2%。差太多就是算法的锅，环境这次没有嫌疑。")
    return measured


def save_curve(args, curve, measured=None) -> Path:
    CURVE_DIR.mkdir(exist_ok=True)
    tag = args.variant + ("_rawobs" if args.raw_obs else "")
    path = CURVE_DIR / f"{tag}.json"
    path.write_text(json.dumps({
        "variant": args.variant, "raw_obs": args.raw_obs, "episodes": args.episodes,
        "lr": args.lr, "hidden": args.hidden, "batch": args.batch,
        "replay_size": args.replay_size, "target_sync": args.target_sync,
        "seed": args.seed,
        # The epsilon schedule is saved because the curve cannot be read without
        # it - see `report`. Older files predate these keys; `plot_all` treats
        # them as "no reference line available" rather than guessing.
        "eps_start": args.eps_start, "eps_end": args.eps_end,
        "measured": measured or {},
        "curve": curve,
    }, indent=1), encoding="utf-8")
    return path


# Colour is assigned by entity, in this fixed order, so that a missing run never
# repaints the others. Three validated categorical slots; a fourth variant would
# have to fold in rather than invent a hue.
VARIANT_ORDER = ("naive", "replay", "dqn")
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _no_learning_reference(d):
    """The curve a policy that never improved would still have drawn.

    The training curve is measured **with** exploration: a fraction `epsilon` of
    the actions are uniform-random, and epsilon decays over training. So the line
    rises on its own even when nothing is learned - at the start most moves are
    random (win rate near the random baseline), at the end almost none are (win
    rate near the greedy one). Plotting the curve without this reference invites
    reading epsilon decay as learning, which is exactly what happened on the
    first `naive` run: it looked like a breakthrough at episode 15,000.

    Mixing the two rates linearly is an approximation - one random action early in
    a fight drags the rest of that fight with it - so this is a reference, not a
    prediction. The part that matters is robust to that: where the real curve sits
    **below** this line, the policy was genuinely worse than it ended up, and
    closing that gap is the learning.
    """
    m = d.get("measured") or {}
    rand, greedy = m.get("随机"), m.get("DQN")
    if rand is None or greedy is None or "eps_start" not in d:
        return None
    n, eps0, eps1 = d["episodes"], d["eps_start"], d["eps_end"]
    block = max(1, n // 40)
    out = []
    for i in range(len(d["curve"])):
        ep = (i + 1) * block
        eps = eps0 + (eps1 - eps0) * (ep / n)
        out.append(eps * rand + (1 - eps) * greedy)
    return out


def plot_all() -> None:
    """Every saved curve on one figure - the deliverable of this stage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(CURVE_DIR.glob("*.json"))
    if not files:
        print(f"no curves in {CURVE_DIR} yet - run the three variants first")
        return

    runs = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        v = d.get("variant", f.stem)
        rank = VARIANT_ORDER.index(v) if v in VARIANT_ORDER else len(VARIANT_ORDER)
        runs.append((rank, f.stem, d))
    runs.sort(key=lambda r: (r[0], r[1]))

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Benchmarks first, so the data sits on top of them.
    for y, text, style in ((0.788, "tabular Q  78.8%", (0, (5, 3))),
                           (0.822, "four lines of `if`  82.2%", (0, (1, 2)))):
        ax.axhline(y, ls=style, lw=1, color=MUTED, zorder=1)
        # Right-aligned: the legend lives top-left and the two collided there.
        ax.text(0.996, y + 0.008, text, fontsize=8, color=MUTED, ha="right",
                transform=ax.get_yaxis_transform(), zorder=1)

    labelled_reference = False
    for i, (_, stem, d) in enumerate(runs):
        colour = SERIES_COLOURS[min(i, len(SERIES_COLOURS) - 1)]
        block = max(1, d["episodes"] // 40)
        xs = [(j + 1) * block for j in range(len(d["curve"]))]

        ref = _no_learning_reference(d)
        if ref is not None:
            ax.plot(xs, ref, ls=(0, (4, 3)), lw=1.4, color=colour, alpha=0.55,
                    zorder=2,
                    label="no learning (epsilon decay alone)"
                    if not labelled_reference else None)
            labelled_reference = True

        ax.plot(xs, d["curve"], lw=2, color=colour, label=stem, zorder=3)
        # A coloured dot carries identity; the text stays ink. Direct labels are
        # also the relief the palette check asks for on the low-contrast slot.
        ax.plot(xs[-1], d["curve"][-1], "o", ms=6, color=colour,
                mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f" {stem} {d['curve'][-1]:.0%}", (xs[-1], d["curve"][-1]),
                    fontsize=9, color=INK, va="center", zorder=4)

    ax.set_xlabel("episodes", fontsize=9, color=MUTED)
    ax.set_ylabel("win rate during training (with exploration)",
                  fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_ylim(0.25, 0.88)
    ax.margins(x=0.13)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        leg = ax.legend(handles, labels, fontsize=8, frameon=False,
                        loc="upper left", labelcolor=INK)
        leg.set_zorder(5)

    fig.tight_layout()
    out = CURVE_DIR / "curves.png"
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"wrote {out}")


def check() -> None:
    """Everything that can be verified without training. Run this first."""
    rng = random.Random(0)
    ok = True

    def c(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {name}" + ("" if good else f"  got {got!r} want {want!r}"))

    print("observation")
    s = env.reset(rng)
    v = encode_vector(s)
    c("length matches OBS_NAMES", len(v), OBS_DIM)
    c("normalised into [0,1]", all(0.0 <= x <= 1.0 for x in v), True)
    c("raw version is the same length", len(encode_vector(s, normalize=False)), OBS_DIM)

    # The pit the roadmap asks you to step in on purpose, made visible without
    # training: without normalisation the components differ by two orders of
    # magnitude, and gradient size is proportional to component size.
    raw = encode_vector(s, normalize=False)
    print(f"  ..  normalised range {min(v):.3f}-{max(v):.3f}   "
          f"raw range {min(raw):.1f}-{max(raw):.1f}  <- the scale pit")

    print("coverage: walk 2000 random fights, every component stays in range")
    lo = [9e9] * OBS_DIM
    hi = [-9e9] * OBS_DIM
    # ⚠️ Reaching 1.000 and being *clipped* are different things, and the first
    # version of this check conflated them - it reported twelve components as
    # "hitting the cap" when most of them simply have a scale equal to their true
    # maximum (full hp is 1.0, 3 of 3 energy is 1.0, and `block_covers` is a
    # boolean). Only a component whose raw value exceeds its scale is losing
    # information, so count that instead. Same distinction as `or []` flattening
    # "missing" into "empty".
    clipped = [0] * OBS_DIM
    steps = 0
    for _ in range(2000):
        st = env.reset(rng)
        for _ in range(STEP_CAP):
            vv = encode_vector(st)
            uncapped = encode_vector(st, normalize=True, cap=False)
            lo = [min(a, b) for a, b in zip(lo, vv)]
            hi = [max(a, b) for a, b in zip(hi, vv)]
            for i, u in enumerate(uncapped):
                if u > 1.0:
                    clipped[i] += 1
            steps += 1
            st, _, done = env.step(st, rng.choice(env.legal_actions(st)), rng)
            if done:
                break
    c("all components within [0,1]", all(0 <= a and b <= 1 for a, b in zip(lo, hi)), True)
    dead = [OBS_NAMES[i] for i in range(OBS_DIM) if hi[i] == lo[i]]
    print(f"  ..  components that never varied: {dead or 'none'}"
          + ("  <- constant input, contributes nothing but a bias" if dead else ""))
    losing = [(OBS_NAMES[i], clipped[i]) for i in range(OBS_DIM) if clipped[i]]
    if losing:
        print(f"  ..  components actually clipped (raw > scale), out of {steps:,} steps:")
        for name, n in sorted(losing, key=lambda kv: -kv[1]):
            print(f"        {name:<18} {n:6,}  {n / steps:5.1%}")
        print("        ^ information thrown away. Fine if deliberate - "
              "`incoming_vs_hp` above 1 just means lethal - and a bug if not.")
    else:
        print("  ..  nothing is clipped: every scale is at or above its true max")

    print("the two functions that are yours")
    for fn, name in ((build_net, "build_net"), (td_target, "td_target")):
        try:
            fn(None, 1, 1) if name == "build_net" else fn(None, 0.0, None, True, 1.0, [0])
            print(f"  ok   {name} is written")
        except NotImplementedError:
            print(f"  ..   {name} still raises NotImplementedError - that one is yours")
        except Exception as exc:
            print(f"  ok   {name} is written (called with dummies: {type(exc).__name__})")

    print("\nall good" if ok else "\nsomething above is wrong")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="dqn")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--target-sync", type=int, default=500,
                    help="gradient steps between copying net -> target_net")
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--fights", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw-obs", action="store_true",
                    help="skip normalisation, on purpose, to watch hp dominate")
    ap.add_argument("--check", action="store_true", help="wiring checks, no training")
    ap.add_argument("--plot", action="store_true", help="draw every saved curve")
    args = ap.parse_args()

    if args.check:
        return check()
    if args.plot:
        return plot_all()

    print(f"variant {args.variant}  obs {OBS_DIM}维"
          f"{' (未归一化)' if args.raw_obs else ''}  "
          f"episodes {args.episodes:,}  lr {args.lr}  hidden {args.hidden}  "
          f"batch {args.batch}  gamma {args.gamma}")
    curve, net = train(args)
    measured = report(args, curve, net)
    print(f"curve -> {save_curve(args, curve, measured)}")


if __name__ == "__main__":
    main()
