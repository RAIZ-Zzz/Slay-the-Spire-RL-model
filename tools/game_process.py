"""Start, stop and wait for the game process itself.

Everything else in this repo talks to a game that is already running. This is the
layer below that: when a run wedges on something the loop cannot answer - an
unhandled overlay, eight refusals, a state that stopped moving - the only way to
keep an unattended session going is to restart the process.

Three facts make that practical, all checked on 2026-09-14 rather than assumed:

  * **The mod selection persists.** `settings.save` holds
    `mod_settings.mod_list = [{"id": "STS2_Bridge", "is_enabled": true}]`, so the
    startup dialog that had to be ticked by hand on 2026-09-07 is a one-time
    thing. Miss this and every relaunch looks like a broken DLL.
  * **Mods load before the UI.** In the startup log the bridge's HTTP server
    comes up at line 19, before display settings at line 43. So "is the bridge
    answering" is a usable readiness signal, and it arrives early.
  * **Steam is already running**, so the executable can be launched directly;
    Steamworks initialises from the running client.

Nothing here sleeps a guessed duration. Readiness is "the bridge answered" and
then "the state says main menu", both polled - the same rule the combat guards
arrived at after four attempts at picking a number.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from cli_anything.slay_the_spire_ii.core.state_adapter import normalize_state
from cli_anything.slay_the_spire_ii.utils.sts2_backend import ApiError

GAME_DIR = Path(r"A:\SteamLibrary\steamapps\common\Slay the Spire 2")
GAME_EXE = GAME_DIR / "SlayTheSpire2.exe"
IMAGE_NAME = "SlayTheSpire2.exe"
SETTINGS = (Path.home() / "AppData/Roaming/SlayTheSpire2/steam")

# Launched through Steam, not by running the executable.
#
# Measured on 2026-09-14, and the game says so itself. Starting the exe directly
# gives:
#
#     [ERROR] Steamworks initialization failed! ... No appID found. Either launch
#     the game from Steam, or put the file steam_appid.txt ... in your game folder.
#
# and that session loaded **zero** mods - `Loading assembly DLL` appears 0 times,
# so the bridge never exists and the port never opens. From the outside that is
# indistinguishable from a slow start, which is how the first attempt at this
# function sat and polled for the full 180 seconds against a game that was never
# going to answer.
#
# From appmanifest_2868840.acf in the Steam library.
STEAM_APP_ID = "2868840"


def is_running() -> bool:
    """True if the game process exists. Uses tasklist: no extra dependency."""
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {IMAGE_NAME}"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return IMAGE_NAME.lower() in out.stdout.lower()


def bridge_answers(game) -> bool:
    """True if the bridge is up. The readiness signal, not a timer."""
    try:
        game.get_state()
        return True
    except (ApiError, OSError):
        return False


def mods_enabled() -> tuple[bool, str]:
    """Is STS2_Bridge ticked in the persisted mod list?

    Checked *before* relaunching rather than after failing to connect: a game
    that comes up without the mod looks exactly like a game that has not
    finished loading, and the loop would sit and wait out the full timeout for
    a condition that is never going to become true.
    """
    import json
    saves = list(SETTINGS.glob("*/settings.save"))
    if not saves:
        return False, f"no settings.save under {SETTINGS}"
    try:
        doc = json.loads(saves[0].read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"could not read {saves[0].name}: {e}"
    settings = doc.get("mod_settings") or {}
    if not settings.get("mods_enabled"):
        return False, "mods_enabled is false in settings.save"
    for mod in settings.get("mod_list") or []:
        if mod.get("id") == "STS2_Bridge":
            if mod.get("is_enabled"):
                return True, "STS2_Bridge enabled in settings.save"
            return False, "STS2_Bridge is present but not enabled"
    return False, "STS2_Bridge is not in settings.save's mod_list"


def kill(timeout: float = 30.0) -> str | None:
    """Stop the game. Returns None on success, or why not.

    `taskkill /F`. There is no graceful path worth taking here: this is only
    called when the run is already wedged, and the save file is written on every
    floor, so what is lost is the current room rather than the run.
    """
    if not is_running():
        return None
    try:
        subprocess.run(["taskkill", "/F", "/IM", IMAGE_NAME],
                       capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return f"taskkill failed: {e}"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_running():
            return None
        time.sleep(0.5)
    return f"still running {timeout}s after taskkill"


def launch() -> str | None:
    """Start the game through Steam. Returns None on success, or why not."""
    if not GAME_EXE.exists():
        return f"{GAME_EXE} does not exist - edit GAME_DIR"
    ok, why = mods_enabled()
    if not ok:
        return (f"refusing to launch: {why}. Without the bridge the game comes up "
                f"and never answers, which is indistinguishable from a slow load.")
    try:
        # `start` hands the URL to Steam and returns immediately; Steam then
        # starts the game with the app context that mod loading depends on. The
        # empty string is `start`'s title argument - without it the URL is taken
        # as the window title and nothing launches.
        subprocess.run(["cmd", "/c", "start", "", f"steam://rungameid/{STEAM_APP_ID}"],
                       check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not launch via steam://rungameid/{STEAM_APP_ID}: {e}"
    return None


def wait_for_menu(game, timeout: float = 180.0, poll: float = 1.0) -> str | None:
    """Block until the game will actually accept `start_new_game`.

    Three conditions, each of which the previous version was missing one of.

    The bridge comes up during mod init, long before the menu scene exists, so a
    successful HTTP call means only "the mod loaded". And `decision == "menu"` is
    not enough either: measured 2026-09-14, that becomes true while `screen` is
    still settling, and `start_new_game` sent at that moment came back
    "Main menu is not open" - an action refused for being early, which reads
    exactly like an action that is wrong.

    So the wait is on `can_start_new_game`, the game's own answer to the question
    the caller is about to ask. Same rule as `legal_verbs`: wait for the state
    that accepts the action, not for one that looks like it should.

    180s because a cold start syncs cloud saves first.
    """
    deadline = time.monotonic() + timeout
    saw_bridge = False
    last = None
    while time.monotonic() < deadline:
        try:
            state = normalize_state(game.get_state())
        except (ApiError, OSError):
            time.sleep(poll)
            continue
        saw_bridge = True
        last = (state.get("decision"), state.get("screen"))
        menu = state.get("menu") or {}
        # Ready = the menu offers at least one of the things it can offer.
        # Not `can_start_new_game` alone: with a saved run the singleplayer
        # button is genuinely absent and continue/abandon are there instead, so
        # requiring it waits out the full timeout on a perfectly good menu -
        # measured 2026-09-14, 180s of it. The caller decides what to do with the
        # save; this only has to say the menu has finished appearing.
        if (state.get("decision") == "menu"
                and state.get("screen") == "main_menu"
                and (menu.get("can_start_new_game") or menu.get("can_continue_game"))):
            return None
        time.sleep(poll)
    if not saw_bridge:
        return (f"the bridge never answered in {timeout:.0f}s - the game may be up "
                f"without the mod loaded; check the startup log")
    return (f"the bridge answered but the main menu was never ready in "
            f"{timeout:.0f}s (last seen: decision={last[0]!r} screen={last[1]!r})")


def abandon_saved_run(game, timeout: float = 60.0, poll: float = 0.5) -> str | None:
    """Throw away the run on the main menu. None on success.

    Two calls, because the bridge's abandon is two-phase: the first opens a
    confirmation popup the game builds a frame or more later, the second presses
    yes. Waiting happens here rather than in the handler - the handler *is* what
    a caller blocks on, so it cannot wait for the main loop to do anything.

    Success is checked against `can_continue_game` going false, not against the
    replies. Both calls answer `status: ok`, and on 2026-09-14 the old one-call
    version answered `ok` while the save was still there - "I clicked" has never
    meant "it worked" in this bridge.
    """
    deadline = time.monotonic() + timeout
    asked = False
    while time.monotonic() < deadline:
        try:
            menu = (normalize_state(game.get_state()).get("menu") or {})
        except (ApiError, OSError) as e:
            return f"could not read the menu while abandoning: {e}"

        if not menu.get("can_continue_game"):
            return None
        if menu.get("abandon_confirm_open") or not asked:
            try:
                game.post_action("abandon_game")
            except (ApiError, OSError) as e:
                return f"abandon_game failed: {e}"
            asked = True
        time.sleep(poll)
    return (f"the saved run was still there {timeout:.0f}s after asking to abandon "
            f"it (can_continue_game never went false)")


def start_run(game, character: str = "IRONCLAD", ascension: int = 0,
              abandon_existing: bool = False,
              timeout: float = 90.0, poll: float = 1.0) -> str | None:
    """Get a fresh run going from the main menu. None on success.

    Two things measured on 2026-09-14, both of which the first version got wrong.

    **A saved run blocks this, and no amount of waiting helps.** With a run in
    progress the main menu shows continue/abandon and `ExecuteStartNewGame`
    answers "Singleplayer button is not available" for ever - retrying that for
    90 seconds was the first version's entire behaviour. This matters most for
    `--restart-game`, where the whole point is that the previous run wedged, so
    there is *always* a continuable save afterwards. Hence `abandon_existing`:
    the caller says whether discarding it is intended, rather than this guessing.

    **`can_start_new_game` is not a precondition.** It reads true on the main menu
    regardless, including in the state above. Nothing reports what the handler
    actually needs (`_singleplayerButton.IsVisibleInTree()`), so acceptance of the
    action is the only ground truth - and a refused start is a no-op, so asking
    again is free.
    """
    try:
        menu = (normalize_state(game.get_state()).get("menu") or {})
    except (ApiError, OSError) as e:
        return f"could not read the menu: {e}"

    if menu.get("can_continue_game"):
        if not abandon_existing:
            return ("a saved run is in progress, so the main menu offers no "
                    "singleplayer button. Continue it, or pass "
                    "abandon_existing=True to throw it away.")
        problem = abandon_saved_run(game, timeout=timeout, poll=poll)
        if problem:
            return problem

    deadline = time.monotonic() + timeout
    last = "never attempted"
    while time.monotonic() < deadline:
        try:
            reply = game.post_action("start_new_game",
                                     character=character, ascension=ascension)
        except (ApiError, OSError) as e:
            last = str(e)
            time.sleep(poll)
            continue
        if reply.get("status") == "ok":
            return None
        last = str(reply.get("error"))
        time.sleep(poll)
    return f"could not start a run in {timeout:.0f}s, last refusal: {last}"


def restart(game, timeout: float = 180.0) -> str | None:
    """Kill, relaunch, and wait until the main menu is up."""
    problem = kill()
    if problem:
        return problem
    # A beat before relaunching. Not a readiness wait - the file handles on the
    # save directory and the mod DLL are released by the OS, and starting a
    # second process on top of that is how a DLL ends up locked.
    time.sleep(2.0)
    problem = launch()
    if problem:
        return problem
    return wait_for_menu(game, timeout=timeout)
