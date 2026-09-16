"""One consolidated schema file, merged from every source that ever recorded a state.

Why one file. The captures used to be one JSON per variant in `schemas/`, written
by `watch_state.py` on first sight and never touched again. That had three
problems at once, all visible on 2026-09-14:

  * **Stale.** The directory stopped changing on 2026-09-07, because everything
    after that ran through `play.py`, which logs to `trajectories/` instead. The
    dashboard said "schema 11/15 captured, treasure and shop are in", and the
    files for treasure and shop did not exist - the data was in a jsonl nobody
    was reading as a schema source.
  * **First sight only.** A single sample cannot say which keys are optional.
    `unplayable_reason` is absent when a card is playable; from one capture that
    is indistinguishable from a key that does not exist.
  * **Split across layers.** `raw` is what the bridge posts, `normalized` is what
    `choose()` reads, `brief` is what a policy is shown. Three different key sets
    for the same screen, and the older files carried only two of them.

So this keeps one `schemas/schemas.json`, keyed by (decision, screen variant),
holding a key census per layer plus one real sample. Merging is additive and
idempotent: re-running over the same logs changes nothing, and a new observation
only ever adds keys and bumps counts.

Usage:
    python tools/schema_store.py            # rebuild from all sources, report
    python tools/schema_store.py --check    # rebuild in memory, fail if stale
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "schemas"
STORE = SCHEMA_DIR / "schemas.json"
# The original per-variant captures, moved out of `schemas/` so the directory
# presents one readable file. They stay as *inputs*: they hold the only copy of
# some raw states, so deleting them would quietly shrink what a rebuild can see.
CAPTURE_DIR = SCHEMA_DIR / "captures"
TRAJECTORY_DIR = REPO_ROOT / "trajectories"

# Every decision `normalize_state` can emit, so the report can say what is still
# unseen rather than only what is present.
ALL_DECISIONS = [
    "menu", "map_select", "combat_play", "combat_rewards", "card_reward",
    "card_select", "hand_select", "event_choice", "rest_site", "shop",
    "treasure", "relic_select", "game_over", "overlay", "unknown",
]

# What distinguishes two states that share a `decision`. Learned the hard way:
# `decision=menu` covers both the main menu (characters `[]`) and character
# select (a full list), and a policy that keys only on `decision` reads one as
# the other. Kept identical to watch_state.VARIANT_FIELDS on purpose - two
# different ideas of "the same screen" would split the store in half.
VARIANT_FIELDS = {
    "menu": ("screen",),
    "combat_play": ("room_type",),
    "event_choice": ("is_ancient", "in_dialogue"),
    "card_select": ("screen_type",),
    "overlay": ("screen_type",),
}


def variant_key(state: dict) -> str:
    """`decision`, plus whatever distinguishes two screens that share it.

    A missing variant field becomes "?" rather than being dropped. Dropping it
    silently merged two different questions into one bucket name: an older brief
    that never carried `is_ancient` produced `event_choice__False`, which reads
    like a third kind of event rather than "this record cannot say". The "?" is
    the honest version, and it sorts next to its siblings.
    """
    decision = str(state.get("decision") or "unknown")
    fields = VARIANT_FIELDS.get(decision, ())
    if not fields:
        return decision
    parts = ["?" if state.get(f) is None else str(state.get(f)) for f in fields]
    return "__".join([decision, *parts])


def _census(entry: dict, layer: str, payload: dict, when: str) -> None:
    """Fold one observation of one layer into the entry.

    Counts per key rather than a set of keys: a key seen in 3 of 40 observations
    is optional, and that is the question a single sample could never answer.
    """
    slot = entry["layers"].setdefault(
        layer, {"observations": 0, "keys": {}, "sample": None, "sample_at": None})
    slot["observations"] += 1
    for k in payload:
        slot["keys"][k] = slot["keys"].get(k, 0) + 1
    # Keep the richest sample rather than the first: a combat state caught
    # mid-animation has an empty hand and teaches nothing about a hand's shape.
    if slot["sample"] is None or len(payload) > len(slot["sample"]):
        slot["sample"] = payload
        slot["sample_at"] = when


def observe(store: dict, when: str, *, raw: dict | None = None,
            normalized: dict | None = None, brief: dict | None = None) -> str:
    """Record one observation. Every layer is optional; at least one is required.

    Each layer is counted only if it was actually supplied. An earlier version
    took the brief positionally as "the state" and recorded it as the normalized
    layer too, which made the two look identical for every policy log
    (`combat_play brief:20k/382 normalized:20k/382`) - a census of one layer
    reported as a census of two. The brief is a deliberately lossy subset (no
    `draw_pile`, among others); saying it is the normalized state would defeat
    the point of tracking layers at all.
    """
    for_key = normalized or brief or raw or {}
    key = variant_key(for_key)
    entry = store.setdefault(key, {
        "decision": for_key.get("decision"),
        "variant": key,
        "first_seen": when,
        "last_seen": when,
        "observations": 0,
        "layers": {},
    })
    entry["observations"] += 1
    entry["first_seen"] = min(entry["first_seen"], when)
    entry["last_seen"] = max(entry["last_seen"], when)
    for name, payload in (("raw", raw), ("normalized", normalized), ("brief", brief)):
        if payload is not None:
            _census(entry, name, payload, when)
    return key


# --- sources -----------------------------------------------------------------

def _load_normalizer():
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))
    from cli_anything.slay_the_spire_ii.core.state_adapter import normalize_state
    return normalize_state


def rebuild() -> tuple[dict, dict]:
    """Scan every source. Returns (store, per-source counts)."""
    normalize_state = _load_normalizer()
    store: dict = {}
    counts = {"legacy": 0, "sessions": 0, "policy": 0, "skipped": 0}

    # 1. the archived per-variant captures, which carry raw *and* normalized.
    for path in sorted(CAPTURE_DIR.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counts["skipped"] += 1
            continue
        raw, norm = doc.get("raw"), doc.get("normalized")
        if not isinstance(norm, dict):
            counts["skipped"] += 1
            continue
        observe(store, doc.get("captured_at") or "", raw=raw, normalized=norm)
        counts["legacy"] += 1

    # 2. watcher sessions: raw only, so the normalized layer is recomputed here.
    #    Labelled honestly - this says what today's adapter makes of a 2026-09-07
    #    state, which is what the loop would see, not what it saw back then.
    for path in sorted(TRAJECTORY_DIR.glob("session_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                raw = rec["raw"]
                observe(store, rec.get("t") or "",
                        raw=raw, normalized=normalize_state(raw))
                counts["sessions"] += 1
            except Exception:
                counts["skipped"] += 1

    # 3. policy logs: `brief` only. A lossy subset by design (no draw_pile), but
    #    the only source that is current, and the only one covering rest_site,
    #    shop, treasure and overlay at all.
    for path in sorted(TRAJECTORY_DIR.glob("*.jsonl")):
        if path.name.startswith("session_") or path.name == "runs.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                b = rec.get("brief")
                if not isinstance(b, dict):
                    counts["skipped"] += 1
                    continue
                observe(store, rec.get("t") or "", brief=b)
                counts["policy"] += 1
            except Exception:
                counts["skipped"] += 1

    return store, counts


def merge_into_store(when: str, *, raw=None, normalized=None, brief=None) -> str:
    """Fold one live observation into the on-disk store. Used by watch_state."""
    doc = load()
    key = observe(doc["variants"], when, raw=raw, normalized=normalized, brief=brief)
    save(doc["variants"])
    return key


def load() -> dict:
    if STORE.exists():
        return json.loads(STORE.read_text(encoding="utf-8"))
    return {"generated_at": None, "variants": {}}


def save(variants: dict) -> None:
    SCHEMA_DIR.mkdir(exist_ok=True)
    STORE.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "note": ("One entry per (decision, variant). `layers` holds a key "
                         "census per layer - raw is what the bridge posts, "
                         "normalized is what choose() reads, brief is what a "
                         "policy is shown. A key's count below `observations` "
                         "means it is optional."),
                "all_decisions": ALL_DECISIONS,
                "variants": variants,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def report(store: dict, counts: dict) -> None:
    print(f"sources: {counts['legacy']} legacy files, {counts['sessions']} watcher "
          f"states, {counts['policy']} policy briefs, {counts['skipped']} skipped")
    print(f"{len(store)} variants\n")
    print(f"  {'variant':34} {'obs':>5}  layers (keys/observations)")
    for key in sorted(store):
        e = store[key]
        layers = "  ".join(
            f"{name}:{len(s['keys'])}k/{s['observations']}"
            for name, s in sorted(e["layers"].items()))
        print(f"  {key:34} {e['observations']:>5}  {layers}")

    seen = {e["decision"] for e in store.values()}
    missing = [d for d in ALL_DECISIONS if d not in seen]
    print(f"\ndecisions {len(ALL_DECISIONS) - len(missing)}/{len(ALL_DECISIONS)}"
          f"   still unseen: {' '.join(missing) if missing else '(none)'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the store is out of date")
    args = ap.parse_args()

    store, counts = rebuild()
    report(store, counts)

    if args.check:
        old = load().get("variants", {})
        if old != store:
            print("\nstore is out of date - run without --check")
            return 1
        print("\nstore is up to date")
        return 0

    save(store)
    print(f"\nwrote {STORE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
