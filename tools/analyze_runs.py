"""Clean, attribute and stratify the local run history.

Successor to `survey_runs.py` (deleted 2026-09-13), which produced wrong per-character win rates
because it read `players[0]` - in 82 of 179 runs that is a co-op teammate, not
the save's owner.

Data source: `%APPDATA%\\SlayTheSpire2\\steam\\<steamid>\\...\\saves\\history\\*.run`
(plain JSON, one file per run). Read-only - never write to the save directory.

Mind the profile split. A recursive glob over that path matches two save
profiles, and only the first one counts here:

    profile1\\saves\\history          120 runs   vanilla        <- the data
    modded\\profile1\\saves\\history    59 runs   mods enabled   <- exclude

The modded profile was played with card-pool-altering mods, so its runs are not
comparable to vanilla ones and not to the live game either.

Build it in five levels; each one should run and print something before the next
is started.

  L1  load + validate      do all files parse? is `schema_version` the same
                           across all of them? if not, which fields moved?
  L2  attribution          find *my* entry: the player whose `id` equals the
                           steamid directory the file sits in - not players[0].
  L3  cleaning             keep the vanilla profile only; split solo
                           (len(players)==1) from co-op; drop game_mode !=
                           "standard"; drop modded characters (id not starting
                           with "CHARACTER."); drop test runs (file mtime on or
                           after 2026-09-07). Print how many each filter
                           removed - never drop rows silently.
  L4  stratify + CI        group by (build_id, character, ascension). Every rate
                           gets n and a confidence interval. Small n needs a
                           Wilson interval, not the normal approximation - work
                           out why before using it.
  L5  deck timeline        `deck[i].floor_added_to_deck` says which floor each
                           card joined the deck, so the whole build order can be
                           reconstructed. Do wins pick up their key cards
                           earlier than losses?

Known baseline to check against (measured 2026-09-07; vanilla profile + solo +
standard + vanilla characters): 37 runs, 16 wins, 43.2%. If your L3 output
disagrees, one of us is wrong - find out which before moving on.

That 37 is what is left of the 178 runs the plan was written around, and it is
too thin to be a control group: 29 of them are Silent, every other character has
3 or fewer, and 35 of 37 are on v0.99.1 - four game versions old. Establishing
that is a real result, not a setback.
"""

import collections
import glob
import gzip
import json
import os
import sys
import time
from pathlib import Path
from math import sqrt

BASE = os.path.expandvars(r"%APPDATA%/SlayTheSpire2/steam")
# `*` does not cross a path separator, so `*/profile1` matches the vanilla
# profile but not `*/modded/profile1` - the modded runs are excluded by the
# pattern itself, no extra filter needed.
VANILLA_GLOB = os.path.join(BASE, "*", "profile1", "saves", "history", "*.run")
REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DUMP = str(REPO_ROOT / "external_data" / "runs-all-before-2026-06.json")

# Runs from this date on were driven by the agent, not played by hand. Nothing
# in a run records who was at the controls, so the clock is the only way to tell
# them apart. Version is a separate question with a separate answer: `build_id`
# is right there in the data, so never infer it from a date.
AGENT_ERA_CUTOFF = "2026-09-07"

# Modded characters use the same `CHARACTER.` prefix as the built-in ones - the
# Lex Ninja mod ships `CHARACTER.LEX_NINJA2_CHARACTER_LEX_NINJA2` - so a prefix
# test cannot tell them apart and the roster has to be listed out. These five
# are what the bridge reports on the character-select screen in v0.107.1.
VANILLA_CHARACTERS = {
    "CHARACTER.IRONCLAD",
    "CHARACTER.SILENT",
    "CHARACTER.DEFECT",
    "CHARACTER.NECROBINDER",
    "CHARACTER.REGENT",
}


# --- L1 ---------------------------------------------------------------------

def load_runs():
    """Load every vanilla run. Returns [(path, run_dict), ...].

    A bad file is skipped and reported, never fatal - this game does write
    corrupt saves (there are *.corrupt files sitting next to `history/`), and
    losing 119 good runs to one bad one would be the wrong trade.
    """
    files = glob.glob(VANILLA_GLOB)
    print(f"matched {len(files)} files")

    ok, bad = [], []
    for path in files:
        # One try per file, not one around the whole loop: the point is to skip
        # the bad file, not to abandon everything after it.
        try:
            with open(path, encoding="utf-8") as f:
                ok.append((path, json.load(f)))
        except Exception as e:
            bad.append((path, e))

    print(f"Success: {len(ok)}, Fail: {len(bad)}")
    for path, e in bad:
        print(f"  {path} -> {e!r}")
    return ok


def check_schema(runs) -> None:
    """Is the record layout the same across all files?

    If more than one version shows up, every later assumption about a field
    only holds for part of the data - worth knowing before writing any of it.
    """
    versions = collections.Counter(data.get("schema_version") for _, data in runs)
    print(f"schema_version: {dict(versions)}")





# --- L2 ---------------------------------------------------------------------

def owner_id(path: str) -> str:
    """The steamid directory this run file lives under."""
    for p in Path(path).parts:
        if p.isdigit():
            return p
    raise ValueError(f"path has no steamid: {path}")



def my_player(run: dict, owner: str) -> dict | None:
    """The player entry belonging to `owner`, or None if not present."""
    players = run["players"]
    if len(players) == 1:
        return players[0]
    for player in players:
        if str(player["id"]) == owner:
            return player
    return None


def check_attribution(runs) -> None:
    """Every run lives under my own steamid, so I must appear in all of them."""
    missing = sum(1 for path, data in runs if my_player(data, owner_id(path)) is None)
    print(f"runs where I am not found: {missing}")


# --- normalisation ------------------------------------------------------------

def to_record(data, character, origin):
    """Flatten the few fields the filters and groupings actually need.

    My saves and the community dump hold the same run object but arrive
    differently - one per file versus one per line - so the difference is
    absorbed here, and nothing downstream has to care where a run came from.
    """
    return {
        "origin": origin,
        "character": character,
        "build": data.get("build_id"),
        "ascension": data.get("ascension"),
        "mode": data.get("game_mode"),
        "n_players": len(data.get("players") or []),
        # `start_time` travels inside the run, so it means the same thing for
        # both sources; a file mtime would only exist for the local ones.
        "date": time.strftime("%Y-%m-%d", time.localtime(data["start_time"])),
        "cheated": bool(data.get("_isCheated")),
        "win": bool(data.get("win")),
        "run": data,
    }


def local_records():
    """My own saves, attributed through the steamid in the file path."""
    records = []
    for path, data in load_runs():
        me = my_player(data, owner_id(path))
        records.append(to_record(data, me["character"] if me else None,
                                os.path.basename(path)))
    return records


def load_external(path=EXTERNAL_DUMP):
    """The sts2runs.com monthly dump: NDJSON, one complete run per line.

    Every run in it is single-player, so `players[0]` is the uploader and the
    attribution problem that dominates my own saves simply does not arise.
    """
    opener = gzip.open if path.endswith(".gz") else open
    records, bad = [], 0
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception:
                bad += 1
                continue
            players = data.get("players") or []
            records.append(to_record(data,
                                    players[0].get("character") if players else None,
                                    f"dump:{i}"))
    print(f"external dump: {len(records)} runs, {bad} unparsable lines")
    return records


# --- L3 ---------------------------------------------------------------------

def file_date(path: str) -> str:
    """The file's mtime as YYYY-MM-DD. ISO dates compare correctly as strings."""
    return time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(path)))


def clean(records):
    """Keep only the runs that can speak for one player's own skill.

    Each filter runs on the survivors of the previous one and reports what it
    took, so a stage that quietly eats the data shows up immediately. A filter
    that drops nothing still earns its place: that zero is evidence the
    assumption behind it still holds, and the same filter bites hard on another
    source.
    """
    print(f"cleaning {len(records)} runs")

    kept, removed = [], 0
    for r in records:
        if r["n_players"] != 1:
            removed += 1
            continue
        kept.append(r)
    print(f"  dropped co-op runs          {removed:5}   kept {len(kept):5}")
    records = kept

    kept, removed = [], 0
    for r in records:
        if r["mode"] != "standard":
            removed += 1
            continue
        kept.append(r)
    print(f"  dropped non-standard modes  {removed:5}   kept {len(kept):5}")
    records = kept

    kept, removed = [], 0
    for r in records:
        if r["character"] not in VANILLA_CHARACTERS:
            removed += 1
            continue
        kept.append(r)
    print(f"  dropped modded characters   {removed:5}   kept {len(kept):5}")
    records = kept

    kept, removed = [], 0
    for r in records:
        if r["cheated"]:
            removed += 1
            continue
        kept.append(r)
    print(f"  dropped flagged as cheated  {removed:5}   kept {len(kept):5}")
    records = kept

    kept, removed = [], 0
    for r in records:
        if r["date"] >= AGENT_ERA_CUTOFF:
            removed += 1
            continue
        kept.append(r)
    print(f"  dropped agent-era runs      {removed:5}   kept {len(kept):5}")
    records = kept

    # Restate the filters as assertions. This compares the result against
    # nothing but itself, so it still catches a filter that silently did
    # nothing - which is exactly how the `startswith("CHARACTER.")` test failed.
    for r in records:
        assert r["n_players"] == 1, f"co-op survived: {r['origin']}"
        assert r["mode"] == "standard", f"non-standard survived: {r['origin']}"
        assert r["character"] in VANILLA_CHARACTERS, f"modded survived: {r['origin']}"
        assert not r["cheated"], f"cheated survived: {r['origin']}"
        assert r["date"] < AGENT_ERA_CUTOFF, f"agent-era survived: {r['origin']}"
    return records


# --- L4 ---------------------------------------------------------------------

def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion. Why not wins/n +- z*sqrt(p(1-p)/n)?"""
    win_rate = wins/n
    d = 1 + z*z/n
    center = (win_rate + z*z/(2*n)) / d
    half = z/d * sqrt(win_rate*(1-win_rate)/n + z*z/(4*n*n))
    return center - half, center + half


def stratify(records, field, label) -> None:
    """Split by one field and report each group with its uncertainty.

    Every rate is printed next to its n and its interval, because a rate on its
    own is unreadable: 44% off 9 runs and 44% off 700 runs look identical and
    mean completely different things. Groups are ordered by size so the ones
    worth reading come first, and anything under `THIN` is marked - not hidden,
    since knowing a slice is too thin is itself a result.
    """
    THIN = 30

    groups = {}
    for r in records:
        groups.setdefault(r[field], []).append(r)

    print()
    print(f"{label}  ({len(groups)} groups)")
    print(f"  {'group':<16}{'n':>6}{'wins':>6}{'rate':>8}   {'95% CI':<18}width")

    # Sort on group size only; the keys themselves are a mix of str, int and
    # None across the three fields and are not orderable against each other.
    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    for key, rows in ordered:
        n = len(rows)
        wins = sum(1 for r in rows if r["win"])
        lo, hi = wilson_interval(wins, n)
        mark = "  <- too thin" if n < THIN else ""
        print(f"  {str(key):<16}{n:6}{wins:6}{wins / n:8.1%}"
              f"   [{lo:5.1%}, {hi:5.1%}]  {(hi - lo) * 100:5.1f}pp{mark}")


# --- L5 ---------------------------------------------------------------------

def card_decisions(record):
    """Every card-reward decision in one run.

    Returns [(room_type, [offered card ids], picked id or None), ...]. The
    offered list is what makes this data worth having: a pick on its own says
    what was taken, but only the alternatives say what was passed over.
    """
    out = []
    for act in (record["run"].get("map_point_history") or []):
        if not isinstance(act, list):
            continue
        for point in act:
            for stats in (point.get("player_stats") or []):
                choices = stats.get("card_choices") or []
                if not choices:
                    continue
                offered, picked = [], None
                for c in choices:
                    card = c.get("card") or {}
                    cid = card.get("id")
                    if cid is None:
                        continue
                    offered.append(cid)
                    if c.get("was_picked"):
                        picked = cid
                out.append((point.get("map_point_type"), offered, picked))
    return out


def deck_timeline(record):
    """[(floor, card_id), ...] for the run's own deck, in acquisition order."""
    players = record["run"].get("players") or []
    deck = players[0].get("deck") if players else []
    timeline = [(c.get("floor_added_to_deck"), c.get("id")) for c in (deck or [])]
    timeline.sort(key=lambda item: (item[0] is None, item[0]))
    return timeline


def card_value(records, min_arm=40, top=12) -> None:
    """Win rate when a card was taken vs when it was offered and passed over.

    Comparing takers against everyone else would mostly measure which runs
    happen to be offered which cards. Conditioning on "was offered" removes
    that: both arms reached the same choice, and only the decision differs.
    It is still not a clean experiment - a player already doing well picks
    differently - but it is far closer than a raw pick-rate table.
    """
    offers = collections.Counter()
    n_take, w_take = collections.Counter(), collections.Counter()
    n_pass, w_pass = collections.Counter(), collections.Counter()

    for r in records:
        won = 1 if r["win"] else 0
        for _room, offered, picked in card_decisions(r):
            for cid in offered:
                offers[cid] += 1
                if cid == picked:
                    n_take[cid] += 1
                    w_take[cid] += won
                else:
                    n_pass[cid] += 1
                    w_pass[cid] += won

    rows = []
    for cid, total in offers.items():
        if n_take[cid] < min_arm or n_pass[cid] < min_arm:
            continue
        take = w_take[cid] / n_take[cid]
        skip = w_pass[cid] / n_pass[cid]
        rows.append((take - skip, cid, total, n_take[cid], take, skip))
    rows.sort(reverse=True)

    print()
    print(f"card value: {len(offers)} distinct cards offered, "
          f"{len(rows)} with >= {min_arm} runs on both sides")
    header = (f"  {'card':<28}{'offered':>8}{'pick%':>7}"
              f"{'win|take':>10}{'win|pass':>10}{'delta':>8}")

    def show(section, subset):
        print()
        print(f"  {section}")
        print(header)
        for delta, cid, total, taken, take, skip in subset:
            print(f"  {cid.replace('CARD.', ''):<28}{total:8}{taken / total:7.0%}"
                  f"{take:10.1%}{skip:10.1%}{delta * 100:+7.1f}pp")

    show("strongest when taken", rows[:top])
    show("weakest when taken", rows[-top:])


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "external"
    if source == "local":
        records = local_records()
    else:
        records = load_external()

    builds = collections.Counter(r["build"] for r in records)
    print(f"build_id: {len(builds)} versions, top 5 "
        f"{builds.most_common(5)}")

    records = clean(records)
    wins = sum(1 for r in records if r["win"])
    print(f"baseline: {len(records)} runs, {wins} wins, {wins / len(records):.1%}")
    print("NOTE: pooled across versions - the split below is the readable one")

    stratify(records, "build", "by game version")
    stratify(records, "character", "by character")
    stratify(records, "ascension", "by ascension")

    # Card pools shift between patches, so pooling versions here would mix
    # cards that never coexisted. Fix on the single largest build instead.
    builds = collections.Counter(r["build"] for r in records)
    build = builds.most_common(1)[0][0]
    fixed = [r for r in records if r["build"] == build]
    print()
    print(f"card analysis fixed on {build}: {len(fixed)} runs")
    card_value(fixed)


if __name__ == "__main__":
    main()
