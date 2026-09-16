from __future__ import annotations

from .types import JsonDict


def play_card(card_index: int, target: str | None = None) -> JsonDict:
    payload: JsonDict = {"action": "play_card", "card_index": card_index}
    if target is not None:
        payload["target"] = target
    return payload


def use_potion(slot: int, target: str | None = None) -> JsonDict:
    payload: JsonDict = {"action": "use_potion", "slot": slot}
    if target is not None:
        payload["target"] = target
    return payload


def discard_potion(slot: int) -> JsonDict:
    """Throw away a carried potion to free its slot.

    `slot` is the same number `use_potion` takes - the index into the player's
    potion slots, which is *not* the position in `held_potions`, because empty
    slots are skipped when that list is built. Read it off the entry.
    """
    return {"action": "discard_potion", "slot": slot}


def end_turn() -> JsonDict:
    return {"action": "end_turn"}


def choose_map_node(index: int) -> JsonDict:
    return {"action": "choose_map_node", "index": index}


def choose_event_option(index: int) -> JsonDict:
    return {"action": "choose_event_option", "index": index}


def advance_dialogue() -> JsonDict:
    return {"action": "advance_dialogue"}


def choose_rest_option(index: int) -> JsonDict:
    return {"action": "choose_rest_option", "index": index}


def shop_purchase(index: int) -> JsonDict:
    return {"action": "shop_purchase", "index": index}


def claim_reward(index: int) -> JsonDict:
    return {"action": "claim_reward", "index": index}


# No `skip_reward`. Four bridge-side targets were tried on 2026-09-16 and every
# one was accepted and changed nothing: NRewardsScreen.RewardSkippedFrom,
# CardReward.OnSkipped, NRewardButton's RewardSkipped signal, and the card
# screen's own 跳过 (which the game declares
# `EndSelectionAndDoNotCompleteReward`). **Declining a combat card reward has no
# working path.** Next attempt should dump the live control tree under the
# rewards screen and find the button a human presses, not pick method names out
# of the metadata - that is what these four were.


def select_card_reward(card_index: int) -> JsonDict:
    return {"action": "select_card_reward", "card_index": card_index}


def skip_card_reward(index: int = 0) -> JsonDict:
    """Click one of the card reward's alternative buttons, by position.

    Not always "skip" despite the name, which is the bridge's original wording:
    the row can also hold a reroll, or a sacrifice added by Pael's Wing. Read
    `alternatives` on the state and send that entry's `index`. The default of 0
    keeps older callers working, but it is a guess, not a synonym for skipping.
    """
    return {"action": "skip_card_reward", "index": index}


def proceed() -> JsonDict:
    return {"action": "proceed"}


def select_card(index: int) -> JsonDict:
    return {"action": "select_card", "index": index}


def confirm_selection() -> JsonDict:
    return {"action": "confirm_selection"}


def cancel_selection() -> JsonDict:
    return {"action": "cancel_selection"}


def combat_select_card(card_index: int) -> JsonDict:
    return {"action": "combat_select_card", "card_index": card_index}


def combat_confirm_selection() -> JsonDict:
    return {"action": "combat_confirm_selection"}


def select_relic(index: int) -> JsonDict:
    return {"action": "select_relic", "index": index}


def skip_relic_selection() -> JsonDict:
    return {"action": "skip_relic_selection"}


def claim_treasure_relic(index: int) -> JsonDict:
    return {"action": "claim_treasure_relic", "index": index}


def continue_game() -> JsonDict:
    return {"action": "continue_game"}


def start_new_game(character: str = "IRONCLAD", ascension: int = 0) -> JsonDict:
    return {"action": "start_new_game", "character": character, "ascension": ascension}


def abandon_game() -> JsonDict:
    return {"action": "abandon_game"}


def return_to_main_menu() -> JsonDict:
    return {"action": "return_to_main_menu"}


def from_name(name: str, **kwargs: object) -> JsonDict:
    factories = {
        "play_card": play_card,
        "use_potion": use_potion,
        "end_turn": end_turn,
        "choose_map_node": choose_map_node,
        "choose_event_option": choose_event_option,
        "advance_dialogue": advance_dialogue,
        "choose_rest_option": choose_rest_option,
        "shop_purchase": shop_purchase,
        "claim_reward": claim_reward,
        "select_card_reward": select_card_reward,
        "skip_card_reward": skip_card_reward,
        "proceed": proceed,
        "select_card": select_card,
        "confirm_selection": confirm_selection,
        "cancel_selection": cancel_selection,
        "combat_select_card": combat_select_card,
        "combat_confirm_selection": combat_confirm_selection,
        "select_relic": select_relic,
        "skip_relic_selection": skip_relic_selection,
        "claim_treasure_relic": claim_treasure_relic,
        "continue_game": continue_game,
        "start_new_game": start_new_game,
        "abandon_game": abandon_game,
        "return_to_main_menu": return_to_main_menu,
    }
    try:
        factory = factories[name]
    except KeyError as exc:
        raise ValueError(f"Unknown action name: {name}") from exc
    return factory(**kwargs)
