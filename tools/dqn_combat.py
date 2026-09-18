"""Point the Stage 2 network at the real game's combat - as an instrument, not a policy.

    python tools/dqn_combat.py --selftest --net rl/stage2_dqn/target13.pt
    python tools/play.py --policy heuristic --frozen-deck --act \
        --dqn-watch rl/stage2_dqn/target13.pt      # observe, sends nothing

⚠️ `--selftest` is a smoke test, not an evaluation. `schemas.json` holds exactly
one normalized `combat_play` sample, and in it the hand is empty and energy is 0,
so the only legal action is END_TURN and the margin comes back `inf`. It proves
the net loads and the translation runs; it measures nothing about transfer. The
numbers that do are the ones `--dqn-watch` collects over a live run.

**This is not expected to play better than what is already there.** In the one
environment where both have been measured, four lines of `if` score 82.2% and the
network scores 79.6% (19-dim) or 72.8% (13-dim); and `play.greedy_combat` has a
rule the toy cannot even express - "if a card kills the weakest living enemy,
play it" - because the toy fight has one enemy. Deploying this would most likely
be a downgrade, and no measurement exists that says otherwise.

What it *is* for: the tabular policy has a miss rate, and a network has no
equivalent. A table says "I have never seen this state". A network always
answers, with the same confident four numbers, whether the state is one it was
trained on or one from a run three acts deep with relics and cards the toy has
never heard of. That gap is unmeasured, and it is the thing that decides whether
any of Stage 2 transfers. So this module reports three numbers per decision and
does not pretend to be a strategy:

  * **margin** - best Q minus second-best over the legal actions. The analogue of
    a table hit: near zero means the net has no opinion and the pick is close to
    a coin flip, and it is the only thing separating "the net decided" from "the
    net shrugged". Written into the reason string for the same reason
    `qtable_combat` writes the key there.
  * **out-of-range** - how many observation components the real state pushes past
    the [0, 1] the encoder normalises into. `encode_vector` caps them, so a
    capped 1.0 and a true 1.0 are indistinguishable afterwards; passing
    `cap=False` is the only way to see it happened.
  * **agreement** - whether `greedy_combat` would have played the same card. Two
    policies on the same real state is a comparison that costs nothing and needs
    no ground truth.

⚠️ **Use a `--no-piles` net.** `qtable_combat.to_toy_state` fills draw and
discard with `(0, 0, 0)` markers - `encode_coarse` provably ignores them, so it
never had to lie truthfully. A 19-dim network does not ignore them: measured on
`target.pt`, zeroing those six moves the argmax in **26.1%** of states. Loading a
19-dim net here is refused rather than silently wrong.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "rl"))
sys.path.insert(0, str(_ROOT / "rl" / "stage1_tabular"))
sys.path.insert(0, str(_ROOT / "rl" / "stage2_dqn"))
sys.path.insert(0, str(_ROOT / "tools"))

import dqn_exp1 as dqn  # noqa: E402
import exp1_combat as env  # noqa: E402
import qtable_combat as qt  # noqa: E402

# The translation is not re-implemented. Sharing `to_toy_state` with the tabular
# adapter is what makes the two comparable at all: a second copy would drift, and
# then a difference in their picks would be a difference in their translations.
Miss = qt.Miss
to_toy_state = qt.to_toy_state


def load(path: str):
    """Load a net and refuse a 19-dim one, loudly."""
    net, meta = dqn.load_net(path)
    if meta.get("piles", True):
        raise SystemExit(
            f"{path} was trained with the draw/discard components (19-dim).\n"
            "`to_toy_state` cannot supply them - it fills (0, 0, 0) markers - and a\n"
            "net that reads them picks a different action in 26.1% of states when\n"
            "they are faked. Retrain with --no-piles:\n\n"
            "    python rl/stage2_dqn/dqn_exp1.py --variant target --no-piles "
            "--save rl/stage2_dqn/target13.pt\n"
        )
    return net, meta


def diagnose(toy: env.State) -> dict:
    """What the encoder had to distort to express this state. Empty when nothing."""
    uncapped = dqn.encode_vector(toy, cap=False, piles=False)
    names = dqn.obs_names(piles=False)
    return {n: round(v, 3) for n, v in zip(names, uncapped) if v > 1.0 or v < 0.0}


def choose(state: dict, net, rng: random.Random, adapter, torch):
    """Pick a combat action from the network. Returns `(action, reason)`.

    Falls back to `(None, reason)` only when the state cannot be translated at
    all - a network, unlike a table, has no other way to decline.
    """
    try:
        toy, target = to_toy_state(state)
    except Miss as exc:
        return None, f"dqn miss: {exc}"

    # The real game is the authority on what can be played: `can_play` folds in
    # energy and every card-specific restriction. Intersecting with the toy's
    # vocabulary keeps us from proposing a card the game will ignore - the
    # silent no-op that has bitten this project four times.
    playable = {}
    for card in state.get("hand") or []:
        if card.get("can_play") and card.get("id") in qt.CARD_IDS:
            playable.setdefault(qt.CARD_IDS.index(card["id"]), card)
    legal = sorted(playable) + [env.END_TURN]

    with torch.no_grad():
        row = net(torch.tensor([dqn.encode_vector(toy, piles=False)],
                               dtype=torch.float32))[0]
    values = sorted(((row[a].item(), a) for a in legal), reverse=True)
    best_q, action = values[0]
    margin = best_q - values[1][0] if len(values) > 1 else float("inf")

    out = diagnose(toy)
    note = f"margin {margin:.3f}"
    if out:
        note += f" OUT-OF-RANGE {out}"

    if action == env.END_TURN:
        return adapter.end_turn(), f"dqn end turn ({note})"
    card = playable[action]
    extra = ({"target": target["entity_id"]}
             if card.get("target_type") == "AnyEnemy" else {})
    return (adapter.play_card(card["index"], **extra),
            f"dqn {card.get('name')} ({note})")


# --- selftest -----------------------------------------------------------------


def selftest(path: str) -> int:
    """Run the net over the captured combat samples. Needs no game."""
    torch = dqn._require_torch()
    net, meta = load(path)
    print(f"net: {path}  variant={meta.get('variant')} "
          f"obs_dim={meta['obs_dim']} hidden={meta['hidden']} "
          f"measured={ {k: round(v, 3) for k, v in (meta.get('measured') or {}).items()} }")

    store = json.loads((_ROOT / "schemas" / "schemas.json").read_text(encoding="utf-8"))
    samples = [(n, v["layers"]["normalized"]["sample"])
               for n, v in store["variants"].items()
               if n.startswith("combat_play") and "normalized" in v["layers"]]
    if not samples:
        print("no combat_play sample in schemas.json")
        return 1

    from cli_anything.slay_the_spire_ii.core import action_adapter

    rng = random.Random(0)
    for name, sample in samples:
        print(f"--- {name}")
        try:
            toy, _ = to_toy_state(sample)
        except Miss as exc:
            print(f"    Miss: {exc}")
            continue
        print(f"    toy : {toy}")
        payload, reason = choose(sample, net, rng, action_adapter, torch)
        print(f"    dqn : {reason}")
        print(f"          -> {payload}")
        greedy = _greedy_says(sample)
        print(f"    greedy: {greedy}")
    return 0


def _greedy_says(state: dict) -> str:
    """What `play.greedy_combat` would do with the same state, for comparison."""
    try:
        import play
        cards = {}
        payload, reason = play.greedy_combat(state, cards)
        return reason or str(payload)
    except Exception as exc:  # pragma: no cover - comparison only, never fatal
        return f"(greedy unavailable: {type(exc).__name__}: {exc})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="rl/stage2_dqn/target13.pt")
    ap.add_argument("--selftest", action="store_true",
                    help="run the net over schemas/schemas.json and print what it picks")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest(args.net))
    ap.print_help()


if __name__ == "__main__":
    main()
