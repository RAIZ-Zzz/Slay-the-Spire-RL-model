"""The compact view of a game state that a policy is given to decide from.

Shared by every non-default policy, so that "what the agent sees" is one
decision in one place rather than a detail each driver re-invents. That framing
is Stage 5's observation design question, arriving early: a raw bridge state is
thousands of characters of nested JSON and an agent reading all of it pays for
all of it, per decision, on every call.
"""

from __future__ import annotations


def brief(state: dict) -> dict:
    """What the policy gets to see, and deliberately not the whole state.

    This is the Stage 5 observation design question in miniature: a raw state is
    thousands of characters of nested JSON, most of it irrelevant to the choice
    at hand, and an agent reading all of it pays for all of it. o each decision
    contributes onlyS its own options plus the run context that every decision
    needs - hp, gold, floor.

    One field is left out on purpose even though it is available: `draw_pile`.
    The bridge reports it in true draw order, which a human player cannot see
    (measured 2026-09-08: round 2's hand was exactly the first five entries of
    round 1's draw_pile, in order). A policy that uses it is not playing the same
    game as the human baseline it will be compared against.
    """
    ctx = state.get("context") or {}
    player = state.get("player") or {}
    out = {
        "decision": state.get("decision"),
        "act": ctx.get("act"),
        "floor": ctx.get("floor"),
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "gold": player.get("gold"),
    }
    # The run's whole deck, as counts. Present on every decision rather than only
    # on card rewards: what to buy, what to upgrade and what to remove are the
    # same question asked from different screens, and all of them are about the
    # deck rather than about the thing on offer. About forty tokens; the decision
    # it informs is the one that compounds for the rest of the run.
    if state.get("deck"):
        out["deck"] = state["deck"]
    # What you are carrying, on every screen. Several relics change which card is
    # worth taking, and a potion reward is unjudgeable without knowing what the
    # other slots already hold - both of which happen outside combat, which is
    # exactly where these used to be invisible.
    if state.get("held_relics"):
        out["held_relics"] = [{"name": r.get("name"), "description": r.get("description")}
                              for r in state["held_relics"]]
    if state.get("held_potions"):
        out["held_potions"] = [{"slot": p.get("slot"), "name": p.get("name"),
                                "description": p.get("description")}
                               for p in state["held_potions"]]

    def keep(items, fields):
        return [{f: it.get(f) for f in fields if it.get(f) is not None} for it in items or []]

    d = state.get("decision")
    if d == "map_select":
        # `leads_to` is the one-level lookahead, and without it three nodes of the
        # same type are indistinguishable - which is exactly the situation on act 1
        # floor 1 (three Monsters, differing only in what follows). Dropping it
        # would have meant choosing blind and calling it a decision.
        out["options"] = keep(state.get("choices"), ("index", "type", "col", "row", "leads_to"))
        # The whole act is visible, not just the next row. Route choice is
        # therefore a planning problem over a few dozen nodes, not a greedy pick -
        # unlike combat, whose branching is what makes it too expensive to hand to
        # a language model.
        out["full_map_available"] = bool((state.get("map") or {}).get("nodes"))
        out["boss"] = (state.get("boss") or {}).get("name") or state.get("boss")
        out["send"] = "choose_map_node(index=<list position>)"
    elif d == "card_reward":
        out["options"] = keep(state.get("cards"), ("index", "name", "type", "cost", "rarity", "description"))
        # The alternative row, spelled out. `can_skip` only ever said "at least
        # one of these exists", and with Pael's Wing on the run the row holds a
        # sacrifice next to the skip - a decider shown one boolean had no way to
        # tell them apart, and the handler clicked position 0 either way.
        out["alternatives"] = keep(state.get("alternatives"),
                                   ("index", "title", "option_id", "is_enabled"))
        out["can_skip"] = state.get("can_skip")
        out["send"] = ("select_card_reward(card_index=<index>) or "
                       "skip_card_reward(index=<an index from alternatives>)")
    elif d == "combat_rewards":
        out["options"] = keep(state.get("items"), ("index", "type", "gold_amount", "potion_name", "description"))
        out["open_potion_slots"] = player.get("open_potion_slots")
        # The hint has to agree with `legal_verbs`, because it is the other thing
        # telling the model what it may do. On 2026-09-13 at floor 14 it said
        # "or proceed()" while a potion sat unclaimed, the model answered
        # `proceed`, and only the legality check stopped it - the prompt itself
        # had suggested the move that was about to be refused.
        potions_full = player.get("open_potion_slots") == 0
        items = state.get("items") or []
        claimable = [i for i in items
                     if not (potions_full and i.get("type") == "potion")]
        if claimable:
            out["send"] = "claim_reward(index=<index>)"
        elif (potions_full and any(i.get("type") == "potion" for i in items)
                and state.get("can_discard_potions") and state.get("held_potions")):
            # The swap. Named here because the decision is only visible from this
            # screen: the potion on offer is in `options`, the ones already
            # carried are in `held_potions` with their slots, and until today
            # there was no verb connecting them.
            out["can_discard_potions"] = True
            out["send"] = ("discard_potion(slot=<a slot from held_potions>) to free a slot "
                           "and then claim_reward, or proceed() to leave the potion")
        else:
            out["send"] = "proceed() - nothing left that can be claimed"
    elif d == "event_choice":
        # `event_id` is the stable identifier - 涅奥 is NEOW in every language and
        # every patch, while the title and the option text are localised prose.
        # A policy that wants to recognise an event (or that will one day look it
        # up in the library dump) needs this, and it was not being shown.
        out["event_id"] = state.get("event_id")
        out["event"] = state.get("event_name")
        out["text"] = state.get("description")
        out["in_dialogue"] = state.get("in_dialogue")
        out["is_ancient"] = state.get("is_ancient")
        # `is_proceed` and `was_chosen` observed live 2026-09-13 on NEOW, and both
        # matter: after the blessing is taken the event comes back with a single
        # option titled 继续 and `is_proceed: true`, which is a "leave", not a
        # fourth choice. Without these two an agent cannot tell "pick one of three"
        # from "acknowledge what you already picked".
        out["options"] = keep(
            state.get("options"),
            ("index", "title", "description", "is_locked", "is_proceed", "was_chosen"),
        )
        out["send"] = "choose_event_option(index=<position among UNLOCKED>) or advance_dialogue()"
    elif d == "shop":
        # ⚠️ The three `*_description` keys were missing until 2026-09-16, and the
        # effect was that the agent never bought anything. The bridge reports all
        # of them (`BuildShopState` sets card_description, relic_description and
        # potion_description); this whitelist dropped them, so a shop decision
        # looked like "spend 150 gold on 燃烧之血, yes or no" with no way to find
        # out what 燃烧之血 does. Declining was the only reasonable answer.
        #
        # It matters most for relics: a card's name plus type plus rarity carries
        # something, but a relic's effect is not guessable from its name, and
        # relics are the one purchase with no deck cost at all.
        #
        # Same family as the 2026-09-14 `screen_type` bug - a field the bridge
        # reported that never reached the thing making the decision. Worth
        # checking the *whole* whitelist against the builder whenever either one
        # changes, not just the key that was noticed.
        out["options"] = keep(
            state.get("items"),
            ("index", "category", "cost", "can_afford", "is_stocked", "on_sale",
             "card_name", "card_type", "card_rarity", "card_description",
             "relic_name", "relic_description",
             "potion_name", "potion_description"),
        )
        out["send"] = ("shop_purchase(index=<index>) where the entry is is_stocked "
                       "and can_afford, or proceed()")
    elif d == "rest_site":
        out["options"] = keep(state.get("options"), ("index", "id", "name", "is_enabled"))
        out["send"] = "choose_rest_option(index=<position among ENABLED>) or proceed()"
    elif d in ("treasure", "relic_select"):
        out["options"] = keep(state.get("relics"), ("index", "name", "rarity", "description"))
        out["message"] = state.get("message")
        out["send"] = "claim_treasure_relic(index=) / select_relic(index=) / proceed()"
    elif d == "card_select":
        out["screen_type"] = state.get("screen_type")
        out["prompt"] = state.get("prompt")
        out["can_confirm"] = state.get("can_confirm")
        # How many are picked and how many the screen wants. Without these the
        # only way to find the target was to pick cards until confirm lit up,
        # which never happens on a screen asking for an exact number.
        out["selected_count"] = state.get("selected_count")
        out["min_select"] = state.get("min_select")
        out["max_select"] = state.get("max_select")
        out["preview_showing"] = state.get("preview_showing")
        out["can_skip"] = state.get("can_skip")
        out["options"] = keep(state.get("cards"), ("index", "name", "type", "cost", "is_upgraded"))
        out["send"] = "select_card(index=) then confirm_selection(); cancel_selection() to skip"
    elif d == "combat_play":
        # Added 2026-09-13, after `--combat` was shipped without it. The fallback
        # branch below hands over a list of *key names*, so the model was choosing
        # a card and a target from nothing at all - it answered
        # `{"card_index": 0, "target": 0}` and the bridge 500'd on the target,
        # which is the only reason anyone noticed.
        out["round"] = state.get("round")
        out["energy"] = state.get("energy")
        out["max_energy"] = state.get("max_energy")
        out["block"] = player.get("block")
        out["hand"] = keep(state.get("hand"),
                           ("index", "name", "type", "cost", "can_play",
                            "unplayable_reason", "target_type", "description"))
        out["enemies"] = keep(state.get("enemies"),
                              ("entity_id", "hp", "max_hp", "block", "intents", "status"))
        out["piles"] = {"draw": state.get("draw_pile_count"),
                        "discard": state.get("discard_pile_count"),
                        "exhaust": state.get("exhaust_pile_count")}
        # Potions. Offered as a legal verb since the day combat was handed over,
        # and never once used - because `use_potion` was in the verb list while
        # the potions themselves were not in the brief. Being told an action is
        # available without being told what it acts on is the same failure as
        # asking for a target with no enemy list.
        out["potions"] = keep(player.get("potions"),
                              ("slot", "id", "name", "description",
                               "can_use_in_combat", "target_type"))
        # The player's own buffs and debuffs. Weak, Vulnerable and Strength change
        # every damage number on the screen, and the enemies' statuses were being
        # shown while ours were not.
        out["status"] = keep(player.get("status"), ("id", "name", "amount", "type", "description"))
        out["relics"] = keep(player.get("relics"), ("id", "name", "description"))
        out["send"] = ("play_card(card_index=<the card's index field>, "
                       "target=<an entity_id STRING from enemies, only when "
                       "target_type is AnyEnemy>), "
                       "use_potion(slot=<the potion's slot field>, target=<entity_id "
                       "if its target_type is AnyEnemy>), or end_turn()")
        # Counts only, never `draw_pile` itself: the bridge reports it in true
        # draw order, which a human player cannot see. A policy that reads it is
        # not playing the same game as the baseline it gets compared against.
    elif d == "hand_select":
        # In-combat "choose a card from your hand" - exhaust, discard, upgrade.
        # Had no template, so it arrived as a list of key names, exactly like
        # combat did before 2026-09-13.
        out["mode"] = state.get("mode")
        out["prompt"] = state.get("prompt")
        out["can_confirm"] = state.get("can_confirm")
        out["cards"] = keep(state.get("cards"), ("index", "name", "type", "cost", "is_upgraded"))
        out["already_selected"] = keep(state.get("selected_cards"), ("index", "name"))
        out["send"] = "combat_select_card(card_index=<index>) then combat_confirm_selection()"
    elif d == "game_over":
        out["screen_type"] = state.get("screen_type")
        out["send"] = "the run is over; the loop stops here"
    elif d == "overlay":
        # A screen the bridge has no branch for. `screen_type` is the whole point
        # of logging this at all: without it the record reads "not handled yet:
        # overlay" and names nothing, which is what happened on 2026-09-14 when a
        # run died on floor 1 - the bridge had reported the class name and this
        # function dropped it, so which screen wedged that run is now unknowable.
        out["screen_type"] = state.get("screen_type")
        out["can_proceed"] = state.get("can_proceed")
        out["message"] = state.get("message")
        out["send"] = ("proceed() if can_proceed, otherwise nothing - "
                       "this screen needs a new branch in the bridge")
    else:
        out["raw_keys"] = sorted(state.keys())
        out["send"] = "no template for this decision - see action_adapter"
    return out
