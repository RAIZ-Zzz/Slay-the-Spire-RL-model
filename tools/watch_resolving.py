"""Watch for the one window where `is_resolving` might not be enough.

Background. `autoplay.py` decides to end the turn when the hand holds nothing
playable, and waits while `is_resolving` is true. That pairing is only safe if
the game's action queue is *already* non-empty by the time the hand empties. If
there is any gap - the last card finishes resolving, the queue drains, and only
then does a relic enqueue a draw - the loop can slip into that gap and end a turn
that still had cards coming. Spinning Top is the clearest case: empty hand
triggers a draw, and drawing takes time.

Nobody can see a sub-second gap by watching the screen, which is the only honest
reason to write this. It polls faster than the action loop, prints one line per
*change* rather than per poll, and shouts when it sees the dangerous sequence:

    is_resolving false, hand empty, player's turn   ->   hand becomes non-empty

If that never fires across a few combats, `is_resolving` alone is sufficient and
the loop needs nothing more. If it does fire, the printed gap duration says how
much slack a fix has to cover - and the fix is then evidence-driven instead of
another guess, which is how the three previous designs died.

This sends nothing. Play by hand, or run `autoplay.py --act` in another terminal;
either way this only reads.

    python tools/watch_resolving.py              # until Ctrl-C
    python tools/watch_resolving.py --interval 0.05

When `autoplay.py` spawns this into its own console it also passes `--stop-file`
and `--summary-file`, and then the window closes itself once the run is over.
Those two flags go together on purpose: the summary *is* the measurement, so a
window that vanishes on its own has to hand the numbers back first. autoplay
reads the summary file and prints it in the terminal you were already watching.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cli_anything.slay_the_spire_ii.core.state_adapter import normalize_state
from cli_anything.slay_the_spire_ii.utils.sts2_backend import ApiError, Sts2RawClient


def snapshot(state) -> dict:
    """The few fields that decide whether acting now is safe."""
    hand = state.get("hand") or []
    return {
        "decision": state.get("decision"),
        "is_resolving": state.get("is_resolving"),
        "turn": state.get("turn"),
        "round": state.get("round"),
        "hand": len(hand),
        "playable": sum(1 for c in hand if c.get("can_play")),
        "energy": state.get("energy"),
        "draw": state.get("draw_pile_count"),
        "discard": state.get("discard_pile_count"),
    }


def is_danger(s: dict) -> bool:
    """The exact state in which autoplay would end the turn.

    If a card arrives after this was true, the turn would have been thrown away.
    """
    return (
        s["decision"] == "combat_play"
        and s["turn"] == "player"
        and not s["is_resolving"]
        and s["hand"] == 0
    )


def summary_lines(polls: int, hits: int, saw_field: bool, combat_polls: int) -> list[str]:
    """The measurement, as text. Printed here and handed back to autoplay.

    Built in one place because it now has two destinations. When this runs in a
    console that closes itself, printing alone would lose it.

    Three outcomes, and `combat_polls` is what separates the first two. Without
    it the summary said "the running game has an older bridge DLL" whenever
    `is_resolving` was never seen - and on 2026-09-13 it said exactly that after
    50 polls spent entirely on a rewards screen, where `is_resolving` does not
    exist at all (only `_normalize_combat` carries it). The DLL was fine; the
    advice to go and reinstall it was not. "Never sampled combat" and "sampled
    combat and the field was missing" are different events and now read as such.
    """
    lines = [f"polls: {polls} ({combat_polls} in combat)   "
             f"danger windows that refilled: {hits}"]
    if combat_polls == 0:
        lines.append(
            "Nothing to report: no combat_play state was ever sampled, so "
            "is_resolving was never under test. This is not a failure - the run "
            "just never got into a fight while this was watching."
        )
    elif not saw_field:
        lines.append(
            "WARNING: `is_resolving` was None in every combat sample. The "
            "running game has an older bridge DLL - quit the game, confirm "
            "mods/STS2_Bridge/STS2_Bridge.dll is the new build, relaunch "
            "and tick the mod. Nothing was tested."
        )
    elif hits == 0:
        lines.append(
            "No refill ever followed an (is_resolving=False, hand=0) "
            "window. As far as this run goes, is_resolving is sufficient."
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interval", type=float, default=0.1,
                    help="seconds between polls (default 0.1)")
    ap.add_argument("--base-url", default="http://localhost:15526")
    ap.add_argument("--hold", action="store_true",
                    help="wait for Enter before exiting; for a spawned console "
                         "with no --stop-file, where the window would otherwise "
                         "vanish with the summary still in it")
    ap.add_argument("--stop-file", type=Path, default=None,
                    help="stop as soon as this file is gone. autoplay creates it "
                         "before spawning and removes it on the way out, so the "
                         "experiment's end is what closes this window - not a "
                         "timer, and not someone remembering to press Ctrl-C")
    ap.add_argument("--summary-file", type=Path, default=None,
                    help="write the summary here too, so autoplay can print it "
                         "in its own terminal after this window is gone")
    args = ap.parse_args()

    client = Sts2RawClient(base_url=args.base_url, timeout=10.0)
    t0 = time.monotonic()
    prev: dict | None = None
    danger_since: float | None = None
    hits = 0
    polls = 0
    combat_polls = 0   # only these can say anything about is_resolving
    saw_field = False

    if args.stop_file is not None:
        print(f"polling every {args.interval}s - closes itself when autoplay stops")
    else:
        print(f"polling every {args.interval}s - Ctrl-C to stop")
    print(f"{'t':>7} {'resolving':>9} {'turn':<7} {'rnd':>3} {'hand':>4} "
          f"{'play':>4} {'E':>2} {'draw':>4} {'disc':>4}")

    stopped_by = "Ctrl-C"
    try:
        while True:
            # Checked first, so the window goes away promptly once autoplay is
            # done rather than after one more poll of a game nobody is driving.
            #
            # A missing file, not a message: it covers every way autoplay can
            # end - clean stop, an exception, the user's own Ctrl-C - because
            # removing it sits in a `finally`. The one case it misses is autoplay
            # being killed outright, which leaves the file behind and this window
            # polling, exactly as it did before any of this existed.
            if args.stop_file is not None and not args.stop_file.exists():
                stopped_by = "autoplay finished"
                break

            now = time.monotonic() - t0
            try:
                state = normalize_state(client.get_state())
            except ApiError as e:
                print(f"{now:>7.1f}  bridge unreachable: {e}")
                time.sleep(1)
                continue
            polls += 1
            s = snapshot(state)

            # Counted separately because `is_resolving` only exists on a combat
            # state. Polls spent on a map, a shop or a rewards screen cannot say
            # anything about it either way, and counting them as evidence of a
            # missing field is what produced a false DLL warning.
            if s["decision"] == "combat_play":
                combat_polls += 1
                if s["is_resolving"] is not None:
                    saw_field = True

            # The danger window opened: remember when, so its length can be
            # reported if a card does turn up.
            if is_danger(s):
                if danger_since is None:
                    danger_since = now
            elif danger_since is not None:
                # It closed. Cards arriving is the bad way for that to happen;
                # the turn ending or combat moving on is the harmless way.
                if s["hand"] > 0 and s["turn"] == "player":
                    hits += 1
                    print(f"{now:>7.1f}  !!! DANGER: hand refilled to {s['hand']} "
                          f"after {now - danger_since:.2f}s of "
                          f"(is_resolving=False, hand=0). autoplay would have "
                          f"ended this turn. relics may explain it.")
                danger_since = None

            # One line per change keeps a 10Hz poll readable.
            if prev is None or s != prev:
                mark = "  <-- autoplay would end turn here" if is_danger(s) else ""
                print(f"{now:>7.1f} {str(s['is_resolving']):>9} {str(s['turn']):<7} "
                      f"{str(s['round']):>3} {s['hand']:>4} {s['playable']:>4} "
                      f"{str(s['energy']):>2} {str(s['draw']):>4} "
                      f"{str(s['discard']):>4}{mark}")
                prev = s

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    # Reached by both exits now, so the summary cannot depend on which one
    # happened. It used to live inside the KeyboardInterrupt handler, where an
    # automatic stop would have skipped it entirely.
    lines = summary_lines(polls, hits, saw_field, combat_polls)
    print()
    print(f"stopped: {stopped_by}")
    for line in lines:
        print(line)

    if args.summary_file is not None:
        # Written last and in one go. autoplay waits for this file to appear, so
        # a half-written one would be read as the finished article.
        try:
            args.summary_file.write_text("\n".join(lines), encoding="utf-8")
        except OSError as e:
            print(f"(could not write summary to {args.summary_file}: {e})")

    if args.hold:
        input("\n[Enter] to close this window ")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        # Only reachable when something unexpected escapes main(). In a console
        # spawned by autoplay that window closes the instant this process dies,
        # taking the traceback with it - so print it and wait.
        #
        # `--stop-file` counts as "spawned" as much as `--hold` does: a run that
        # closes its own window on success still has to keep it open on a crash,
        # or the only copy of the traceback is gone. Holding here is also what
        # makes autoplay's "no summary file appeared" message land on something
        # still readable.
        import traceback
        traceback.print_exc()
        if "--hold" in sys.argv or "--stop-file" in sys.argv:
            input("\n[Enter] to close this window ")
        raise
