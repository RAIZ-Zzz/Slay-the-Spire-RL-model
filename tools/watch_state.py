"""Live terminal monitor for the STS2 bridge, with automatic schema capture.

Two jobs in one tool:

1. Render the current game state in a readable panel, refreshed on a timer.
2. The first time each decision type is seen, dump both the raw bridge JSON and
   the CLI-normalized JSON to `schemas/`. Walking one run therefore produces the
   Stage 0 deliverable (the state/action schema) without any extra bookkeeping.

Reuses the installed harness (`cli_anything.slay_the_spire_ii`) rather than
re-implementing the HTTP client or the normalizer, so the captured schema is
exactly what the agent will see.

Usage:
    python tools/watch_state.py                 # poll forever, 1s
    python tools/watch_state.py --interval 0.5
    python tools/watch_state.py --once          # print once and exit
    python tools/watch_state.py --once --raw    # dump full JSON instead
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from cli_anything.slay_the_spire_ii.core.state_adapter import normalize_state
from cli_anything.slay_the_spire_ii.utils.sts2_backend import ApiError, Sts2RawClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "schemas"

import schema_store  # noqa: E402  (after REPO_ROOT, same package dir)
TRAJECTORY_DIR = REPO_ROOT / "trajectories"

# Every decision the normalizer can emit. Used for the "n/15 captured" progress
# line, so it is obvious what is still missing from the schema corpus.
ALL_DECISIONS = [
    "menu", "map_select", "combat_play", "hand_select", "combat_rewards",
    "card_reward", "event_choice", "rest_site", "shop", "card_select",
    "relic_select", "treasure", "overlay", "game_over", "unknown",
]

CLEAR = "\033[2J\033[H"
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


# --------------------------------------------------------------------------
# schema capture
# --------------------------------------------------------------------------

# One decision can cover several genuinely different screens - `menu` is both the
# main menu and character select, `combat_play` is monster/elite/boss, an event
# can be ancient or ordinary. Capturing only per decision would silently keep
# just whichever variant happened to come first, so key on the discriminators.
# VARIANT_FIELDS and the key function live in schema_store now. Two modules with
# their own idea of "the same screen" would split the store down the middle.
capture_key = schema_store.variant_key


def capture(raw: dict, state: dict) -> str | None:
    """Fold this state into the consolidated store. Returns the variant, or None.

    Every sighting, not just the first. A first-sight-only capture cannot tell an
    optional key from an absent one - `unplayable_reason` is missing whenever the
    card is playable - and it is why the old `schemas/` directory froze on
    2026-09-07 while the real coverage kept growing in `trajectories/`.

    Returns the variant only when it is new, so the watcher keeps announcing
    first sightings and nothing else.
    """
    before = set(schema_store.load()["variants"])
    key = schema_store.merge_into_store(
        dt.datetime.now().isoformat(timespec="seconds"), raw=raw, normalized=state)
    return key if key not in before else None


class TrajectoryLog:
    """Append every state change to JSONL - one file per watcher session.

    The `.run` save files record only the final deck and outcome, so a human
    playing normally leaves no per-decision trace. Polling and writing down each
    distinct state recovers that: the action taken at each point is implicit in
    the transition to the next state. Consecutive identical states are dropped,
    otherwise idling would bury the real decisions.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.last_fingerprint: str | None = None
        self.count = 0
        self.path: Path | None = None
        if enabled:
            TRAJECTORY_DIR.mkdir(exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = TRAJECTORY_DIR / f"session_{stamp}.jsonl"

    def append(self, raw: dict, state: dict) -> None:
        if not self.enabled or self.path is None:
            return
        fingerprint = json.dumps(raw, sort_keys=True, ensure_ascii=False)
        if fingerprint == self.last_fingerprint:
            return
        self.last_fingerprint = fingerprint
        record = {
            "t": dt.datetime.now().isoformat(timespec="milliseconds"),
            "decision": state.get("decision"),
            "raw_state_type": raw.get("state_type"),
            "raw": raw,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1


def captured_decisions() -> set[str]:
    """Which decisions the store has ever seen. Read from the store, not the dir.

    Counting files under `schemas/` was how the dashboard came to claim treasure
    and shop were captured when no such file existed: those states were seen, but
    by `play.py`, which writes to `trajectories/`. One store, one answer.
    """
    return {e["decision"] for e in schema_store.load()["variants"].values()
            if e.get("decision")}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def fmt_player(p: dict) -> str:
    hp, mx = p.get("hp"), p.get("max_hp")
    bits = [f"hp {hp}/{mx}"]
    if p.get("block"):
        bits.append(f"block {p['block']}")
    if p.get("gold") is not None:
        bits.append(f"gold {p['gold']}")
    for key in ("powers", "buffs", "statuses"):
        if p.get(key):
            bits.append(f"{key}={p[key]}")
    return "  ".join(str(b) for b in bits)


def render_combat(s: dict) -> list[str]:
    p = s.get("player") or {}
    lines = [
        f"{BOLD}PLAYER{RESET}   {fmt_player(p)}   "
        f"energy {s.get('energy')}/{s.get('max_energy')}",
        f"{BOLD}PILES{RESET}    draw {s.get('draw_pile_count')}  "
        f"discard {s.get('discard_pile_count')}  exhaust {s.get('exhaust_pile_count')}"
        f"   round {s.get('round')}  play_phase={s.get('is_play_phase')}",
        f"{BOLD}HAND{RESET}",
    ]
    for i, card in enumerate(s.get("hand") or []):
        name = card.get("name") or card.get("id") or "?"
        playable = card.get("can_play", True)
        mark = f"{GREEN}o{RESET}" if playable else f"{RED}x{RESET}"
        note = "" if playable else f"  {RED}({card.get('unplayable_reason')}){RESET}"
        target = f"  ->{card['target_type']}" if card.get("target_type") else ""
        lines.append(
            f"  {mark} [{i}] {name:<14} cost={str(card.get('cost')):<3}"
            f"{DIM}{card.get('type', '')}{target}{RESET}{note}"
        )
        if card.get("description"):
            lines.append(f"        {DIM}{card['description']}{RESET}")
    lines.append(f"{BOLD}ENEMIES{RESET}")
    for i, e in enumerate(s.get("enemies") or []):
        eid = e.get("entity_id") or e.get("name") or "?"
        # `intents` is a list: an enemy can telegraph several things at once.
        intents = e.get("intents") or []
        parts = []
        for it in intents:
            label = it.get("title") or it.get("type") or "?"
            if it.get("label"):
                label += f" {it['label']}"
            parts.append(label)
        intent_s = " | ".join(parts) if parts else "(none telegraphed)"
        status = f"  {e['status']}" if e.get("status") else ""
        lines.append(
            f"  [{i}] {str(eid):<22} hp {e.get('hp')}/{e.get('max_hp')}"
            f"  block {e.get('block', 0)}{status}"
        )
        lines.append(f"       {YELLOW}intent: {intent_s}{RESET}")
    return lines


def render_choices(s: dict, key: str, label: str) -> list[str]:
    lines = [f"{BOLD}{label}{RESET}"]
    for i, item in enumerate(s.get(key) or []):
        if isinstance(item, dict):
            # Field names vary by state: events use title/description, cards use
            # name, map nodes use room_type. Take the first that is present.
            title_keys = ("title", "name", "text", "label", "room_type", "id")
            desc = next((item[k] for k in title_keys if item.get(k)), "?")
            body = item.get("description") or item.get("body") or ""
            skip = set(title_keys) | {"description", "body"}
            extra = " ".join(
                f"{k}={v}" for k, v in item.items()
                if k not in skip and not isinstance(v, (dict, list))
            )
            lines.append(f"  [{i}] {str(desc):<20} {body}")
            if extra:
                lines.append(f"       {DIM}{extra}{RESET}")
        else:
            lines.append(f"  [{i}] {item}")
    if len(lines) == 1:
        lines.append(f"  {DIM}(none){RESET}")
    return lines


def render_body(s: dict) -> list[str]:
    d = s.get("decision")
    if d == "combat_play":
        return render_combat(s)
    if d == "hand_select":
        return render_choices(s, "hand", "HAND (select)")
    if d == "map_select":
        out = render_choices(s, "choices", "MAP CHOICES")
        boss = s.get("boss")
        if boss:
            out.append(f"  {DIM}boss: {boss}{RESET}")
        return out
    if d == "event_choice":
        return [
            f"{BOLD}EVENT{RESET}  {s.get('event_name')}  "
            f"{DIM}(ancient={s.get('is_ancient')} dialogue={s.get('in_dialogue')}){RESET}",
            f"  {DIM}{str(s.get('description') or '')[:200]}{RESET}",
            *render_choices(s, "options", "OPTIONS"),
        ]
    if d == "rest_site":
        return render_choices(s, "options", "REST OPTIONS")
    if d == "shop":
        return [
            *render_choices(s, "cards", "SHOP CARDS"),
            *render_choices(s, "relics", "SHOP RELICS"),
            *render_choices(s, "potions", "SHOP POTIONS"),
        ]
    if d in {"card_reward", "card_select"}:
        return render_choices(s, "cards" if s.get("cards") else "choices", "CARDS")
    if d == "combat_rewards":
        return render_choices(s, "rewards", "REWARDS")
    if d in {"relic_select", "treasure"}:
        return render_choices(s, "relics", "RELICS")
    if d == "menu":
        return [
            f"{BOLD}MENU{RESET}  screen={s.get('screen')}",
            f"  can_start_new_game={s.get('can_start_new_game')}  "
            f"can_continue_game={s.get('can_continue_game')}  "
            f"can_abandon_game={s.get('can_abandon_game')}",
            f"  characters={s.get('characters')}",
        ]
    if d == "game_over":
        return [f"{BOLD}GAME OVER{RESET}  " + json.dumps(
            {k: v for k, v in s.items() if not isinstance(v, (dict, list))},
            ensure_ascii=False)]
    # unknown / overlay / anything new: show the scalars, they are the clue
    return [
        f"{BOLD}{str(d).upper()}{RESET}",
        *[f"  {k} = {v}" for k, v in s.items() if not isinstance(v, (dict, list))],
    ]


def render(raw: dict, state: dict, interval: float, new_file: str | None,
           log: "TrajectoryLog") -> str:
    ctx = state.get("context") or {}
    run = state.get("run") or {}
    now = dt.datetime.now().strftime("%H:%M:%S")
    head = (
        f"{CYAN}== STS2 live state =={RESET}  {now}  {DIM}poll {interval}s  "
        f"Ctrl+C to stop{RESET}"
    )
    ident = (
        f"{BOLD}decision{RESET} {GREEN}{state.get('decision')}{RESET}"
        f"   {DIM}raw_state_type={raw.get('state_type')}{RESET}"
        f"   act {ctx.get('act')}  floor {ctx.get('floor')}  asc {ctx.get('ascension')}"
    )
    if run.get("character"):
        ident += f"   {run.get('character')}"

    seen = captured_decisions()
    missing = [d for d in ALL_DECISIONS if d not in seen]
    progress = f"{DIM}schema {len(seen)}/{len(ALL_DECISIONS)} captured{RESET}"
    if new_file:
        progress += f"   {GREEN}+ new: {new_file}{RESET}"
    miss_line = f"{DIM}missing: {' '.join(missing) if missing else '(none - all captured)'}{RESET}"
    if log.enabled and log.path is not None:
        log_line = f"{DIM}trajectory {log.count} states -> {log.path.name}{RESET}"
    else:
        log_line = f"{DIM}trajectory logging off{RESET}"

    return "\n".join([head, ident, "", *render_body(state), "",
                      progress, miss_line, log_line])


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interval", type=float, default=1.0, help="poll seconds")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--raw", action="store_true", help="dump JSON instead of the panel")
    ap.add_argument("--base-url", default="http://localhost:15526")
    ap.add_argument("--no-capture", action="store_true", help="do not write schemas/")
    ap.add_argument("--no-log", action="store_true",
                    help="do not record the trajectory to trajectories/")
    args = ap.parse_args()

    try:  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.system("")  # enable ANSI escapes on Windows terminals

    client = Sts2RawClient(base_url=args.base_url, timeout=5.0)
    log = TrajectoryLog(enabled=not (args.no_log or args.once))

    while True:
        try:
            raw = client.get_state()
            if not isinstance(raw, dict):
                raw = {"state_type": "unknown", "payload": raw}
            state = normalize_state(raw)
            new_file = None if args.no_capture else capture(raw, state)
            log.append(raw, state)

            if args.raw:
                print(json.dumps({"raw": raw, "normalized": state},
                                 ensure_ascii=False, indent=2))
            else:
                print(CLEAR + render(raw, state, args.interval, new_file, log))
        except ApiError as exc:
            # HTTP 500 means the bridge is alive but could not build the state -
            # usually a game-API mismatch. That is a very different problem from
            # the game not running, so do not report them the same way.
            text = str(exc)
            if text.startswith("HTTP 5"):
                reason = text
                for line in text.splitlines():
                    if '"error"' in line:
                        reason = line.strip().rstrip(",")
                        break
                msg = (f"{RED}bridge reachable but state unreadable{RESET}\n"
                       f"  {DIM}{reason}{RESET}\n"
                       f"  {YELLOW}likely a game-API mismatch - the mod needs "
                       f"rebuilding against this game version{RESET}")
            else:
                msg = f"{RED}bridge unreachable{RESET}  {DIM}{text}{RESET}"
            print((CLEAR + msg) if not args.once else msg)
            if args.once:
                return 1
        except KeyboardInterrupt:
            return 0

        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
