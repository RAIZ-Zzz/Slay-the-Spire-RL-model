using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2_Bridge;

public static partial class BridgeMod
{
    private static Dictionary<string, object?> BuildGameState()
    {
        var result = new Dictionary<string, object?>();

        if (!RunManager.Instance.IsInProgress)
        {
            result["state_type"] = "menu";
            result["message"] = "No run in progress. Player is in the main menu.";
            result["menu"] = BuildMenuState();
            return result;
        }

        var runState = RunManager.Instance.DebugOnlyGetState();
        if (runState == null)
        {
            result["state_type"] = "unknown";
            return result;
        }

        // Card selection overlays can appear on top of any room (events, rest sites, combat)
        var topOverlay = NOverlayStack.Instance?.Peek();
        var currentRoom = runState.CurrentRoom;

        // A rewards screen with nothing left to offer is treated as not being
        // there, so the room dispatch below can report the decision the game is
        // actually waiting on.
        //
        // Measured 2026-09-13 on a run wedged at act 1 / floor 4:
        //
        //     items                    []      nothing left to claim
        //     IsComplete               true    the screen's own verdict
        //     FindLiveProceedButton()  null    no way out anywhere in the scene
        //     NMapScreen.IsOpen        true, with 1 travelable node
        //
        // The game was waiting for a map choice, and `choose_map_node` was then
        // sent by hand against that exact state and worked - with this screen
        // still on top of the stack.
        //
        // That kills the idea that an overlay blocks what is underneath it.
        // Every action this bridge sends is a direct call
        // (`OnMapPointSelectedLocally`, `ForceClick`, `EmitSignal`), never a
        // synthesised mouse event, so z-order blocks nothing. A stale overlay is
        // a *reporting* problem and only that.
        //
        // So the test has to be a property of this screen. `NMapScreen.IsOpen`
        // cannot be it: it reads true both while rewards are being claimed and
        // after they are done, which is why the guard built on it failed in both
        // directions - present, it hid the rewards screen during claiming;
        // absent, the spent screen hid the map.
        if (topOverlay is NRewardsScreen spentRewards
            && !RewardsScreenHasWorkLeft(spentRewards, runState))
        {
            // Said out loud in the state. "A rewards screen is on top but the
            // state says map" is otherwise a silent contradiction to debug, and
            // an abandoned reward should never be something you have to notice
            // by its absence.
            int inert = CountInertRewards(spentRewards, runState);
            result["shadowed_overlay"] =
                "NRewardsScreen (nothing claimable, no live proceed button) - "
                + "reported as the room underneath"
                + (inert > 0
                    ? $"; leaving {inert} potion reward(s) behind, every potion slot is full"
                    : "");
            topOverlay = null;
        }

        if (topOverlay is NCardGridSelectionScreen cardSelectScreen)
        {
            result["state_type"] = "card_select";
            result["card_select"] = BuildCardSelectState(cardSelectScreen, runState);
        }
        else if (topOverlay is NChooseACardSelectionScreen chooseCardScreen)
        {
            result["state_type"] = "card_select";
            result["card_select"] = BuildChooseCardState(chooseCardScreen, runState);
        }
        // "Choose one of these card bundles" - the screen ScrollBoxes and some
        // Neow blessings open. It had no branch, so it fell to the catch-all and
        // reported `state_type: "overlay"`, which stopped a run dead on floor 1
        // (2026-09-13). Reported as a `card_select` with `screen_type: "bundle"`
        // rather than a new decision type: the agent's job is the same - pick an
        // index and the screen closes - and a new decision would need a branch in
        // every policy before any of them could get past this screen.
        else if (topOverlay is NChooseABundleSelectionScreen bundleScreen)
        {
            result["state_type"] = "card_select";
            result["card_select"] = BuildBundleState(bundleScreen, runState);
        }
        else if (topOverlay is NChooseARelicSelection relicSelectScreen)
        {
            result["state_type"] = "relic_select";
            result["relic_select"] = BuildRelicSelectState(relicSelectScreen, runState);
        }
        else if (topOverlay is NGameOverScreen gameOverScreen)
        {
            result["state_type"] = "game_over";
            result["game_over"] = BuildGameOverState(gameOverScreen, runState);
        }
        // Reward screens are not combat-only. An event can hand out a potion, a
        // card, a relic or gold through the same UI - DROWNING_BEACON (溺水信标)
        // does, and on 2026-09-13 it soft-locked the agent: these two types were
        // excluded from the catch-all below because the CombatRoom branch handled
        // them, so over an EventRoom they matched nothing and fell through to
        // `state_type = "event"` with `options: []`. Nothing to click, no error.
        //
        // No map-precedence guard here. A first attempt added
        // `&& NMapScreen.Instance is not { IsOpen: true }`, reasoning that a
        // rewards screen can linger in the stack after the map opens and the
        // agent would try to claim rewards it already took. Tested 2026-09-13:
        // the map is normally *already open behind* the rewards screen, so the
        // guard never matched and every reward fell through to the catch-all as
        // `state_type = "overlay"`.
        //
        // The case that guard was aimed at is real, though - it is what wedged
        // the next run at floor 4 - and it is handled at the top of this method
        // instead, by a test on the rewards screen rather than on the map. Two
        // claims made here on 2026-09-13 were both wrong and are worth naming,
        // because each one sounded obvious:
        //
        //   * "the agent's handling of an empty `items` is to press proceed" -
        //     it cannot. Once the room's proceed button has been used,
        //     FindLiveProceedButton() returns null and there is no proceed to
        //     press. Measured: can_proceed=false with is_complete=true.
        //   * "whatever is underneath cannot be interacted with" - it can.
        //     Actions are direct calls, not mouse events, and a map node was
        //     selected successfully with this screen sitting on top.
        else if (topOverlay is NCardRewardSelectionScreen anyRoomCardReward)
        {
            result["state_type"] = "card_reward";
            result["card_reward"] = BuildCardRewardState(anyRoomCardReward);
        }
        else if (topOverlay is NRewardsScreen anyRoomRewards)
        {
            result["state_type"] = "combat_rewards";
            result["rewards"] = BuildRewardsState(anyRoomRewards, runState);
        }
        else if (topOverlay is IOverlayScreen)
        {
            // Catch-all for unhandled overlays — prevents soft-locks.
            //
            // `can_proceed` is what makes this an exit rather than just a label.
            // Of the 14 IOverlayScreen implementers, 13 have a branch above; the
            // one that does not is NCrystalSphereScreen, a minigame with a grid
            // of cells and its own _proceedButton. Playing that minigame is a
            // feature and needs live data, but leaving the room does not - and
            // "the run stops here" was the alternative. Reported for any
            // unhandled overlay, not just that one, because the next screen a
            // patch adds will arrive the same way.
            //
            // Deliberately FindLiveProceedButton() rather than a button owned by
            // the overlay: the live one may belong to the room underneath, which
            // is the same thing the rewards screen taught on 2026-09-13.
            var escape = FindLiveProceedButton();
            result["state_type"] = "overlay";
            result["overlay"] = new Dictionary<string, object?>
            {
                ["screen_type"] = topOverlay.GetType().Name,
                ["can_proceed"] = escape?.IsEnabled ?? false,
                ["message"] = $"An overlay ({topOverlay.GetType().Name}) is active with no branch in the bridge. "
                              + (escape?.IsEnabled == true
                                 ? "A live proceed button exists, so `proceed` can leave it - that may forfeit whatever it offers."
                                 : "No live proceed button either: this one needs manual interaction in-game.")
            };
        }
        else if (currentRoom is CombatRoom combatRoom)
        {
            if (CombatManager.Instance.IsInProgress)
            {
                // Check for in-combat hand card selection (e.g., "Select a card to exhaust")
                var playerHand = NPlayerHand.Instance;
                if (playerHand != null && playerHand.IsInCardSelection)
                {
                    result["state_type"] = "hand_select";
                    result["hand_select"] = BuildHandSelectState(playerHand, runState);
                    result["battle"] = BuildBattleState(runState, combatRoom);
                }
                else
                {
                    result["state_type"] = combatRoom.RoomType.ToString().ToLower(); // monster, elite, boss
                    result["battle"] = BuildBattleState(runState, combatRoom);
                }
            }
            else
            {
                // After combat ends, check: map open (post-rewards) > overlays > fallback
                if (NMapScreen.Instance is { IsOpen: true })
                {
                    result["state_type"] = "map";
                    result["map"] = BuildMapState(runState);
                }
                else
                {
                    // The overlay decided once at the top of this method, not a
                    // second Peek(). A second read can disagree with the first -
                    // and it would also undo the spent-rewards rule up there,
                    // putting the stall straight back for the case where the
                    // rewards are done but the map has not opened yet.
                    //
                    // Reaching here therefore means `topOverlay` is null: every
                    // overlay the stack can hold is either matched or caught by
                    // the catch-all above. The two branches below are kept
                    // anyway - deleting unreachable code is a separate change
                    // from fixing a stall, and mixing the two is how you end up
                    // unable to say which one moved the needle.
                    var overlay = topOverlay;
                    if (overlay is NCardRewardSelectionScreen cardScreen)
                    {
                        result["state_type"] = "card_reward";
                        result["card_reward"] = BuildCardRewardState(cardScreen);
                    }
                    else if (overlay is NRewardsScreen rewardsScreen)
                    {
                        result["state_type"] = "combat_rewards";
                        result["rewards"] = BuildRewardsState(rewardsScreen, runState);
                    }
                    else
                    {
                        result["state_type"] = combatRoom.RoomType.ToString().ToLower();
                        result["message"] = "Combat ended. Waiting for rewards...";
                    }
                }
            }
        }
        else if (currentRoom is EventRoom eventRoom)
        {
            if (NMapScreen.Instance is { IsOpen: true })
            {
                result["state_type"] = "map";
                result["map"] = BuildMapState(runState);
            }
            else
            {
                result["state_type"] = "event";
                result["event"] = BuildEventState(eventRoom, runState);
            }
        }
        else if (currentRoom is MapRoom)
        {
            result["state_type"] = "map";
            result["map"] = BuildMapState(runState);
        }
        else if (currentRoom is MerchantRoom merchantRoom)
        {
            if (NMapScreen.Instance is { IsOpen: true })
            {
                result["state_type"] = "map";
                result["map"] = BuildMapState(runState);
            }
            else
            {
                // Auto-open the shopkeeper's inventory if not already open
                var merchUI = NMerchantRoom.Instance;
                if (merchUI != null && !merchUI.Inventory.IsOpen)
                {
                    merchUI.OpenInventory();
                }
                result["state_type"] = "shop";
                result["shop"] = BuildShopState(merchantRoom, runState);
            }
        }
        else if (currentRoom is RestSiteRoom restSiteRoom)
        {
            if (NMapScreen.Instance is { IsOpen: true })
            {
                result["state_type"] = "map";
                result["map"] = BuildMapState(runState);
            }
            else
            {
                result["state_type"] = "rest_site";
                result["rest_site"] = BuildRestSiteState(restSiteRoom, runState);
            }
        }
        else if (currentRoom is TreasureRoom treasureRoom)
        {
            if (NMapScreen.Instance is { IsOpen: true })
            {
                result["state_type"] = "map";
                result["map"] = BuildMapState(runState);
            }
            else
            {
                result["state_type"] = "treasure";
                result["treasure"] = BuildTreasureState(treasureRoom, runState);
            }
        }
        else
        {
            result["state_type"] = "unknown";
            result["room_type"] = currentRoom?.GetType().Name;
        }

        // Common run info
        result["run"] = new Dictionary<string, object?>
        {
            ["act"] = runState.CurrentActIndex + 1,
            ["floor"] = runState.TotalFloor,
            ["ascension"] = runState.AscensionLevel
        };

        result["deck"] = BuildDeckSummary(runState);
        result["held_relics"] = BuildHeldRelics(runState);
        result["held_potions"] = BuildHeldPotions(runState);
        // Whether `discard_potion` will be accepted right now. Reported next to
        // the potions themselves so a policy can tell "the slot is full" from
        // "the slot is full and cannot be emptied" - the two used to be the same
        // observation, and the agent simply forfeited every potion reward once
        // its slots filled. null means the property could not be read, which is
        // deliberately not the same as false.
        result["can_discard_potions"] = SafeBool(
            () => LocalContext.GetMe(runState)?.CanRemovePotions ?? false);

        return result;
    }

    /// <summary>The relics you own, on every state rather than only in combat.</summary>
    /// <remarks>
    /// `BuildPlayerState` has reported these since the start, but it is only
    /// called for a battle - so at a card reward, an event or a shop the agent
    /// could not see what it was already carrying. Several relics change which
    /// card is worth taking, which makes that the exact moment the list matters.
    ///
    /// Named `held_relics`, not `relics`, because `relics` is already the key for
    /// "the relics on offer" in treasure and relic_select states. Two different
    /// meanings under one name is how an agent claims a relic it already owns.
    /// </remarks>
    private static List<Dictionary<string, object?>>? BuildHeldRelics(RunState runState)
    {
        try
        {
            var me = LocalContext.GetMe(runState);
            if (me == null) return null;
            var relics = new List<Dictionary<string, object?>>();
            foreach (var relic in me.Relics)
            {
                if (relic == null) continue;
                relics.Add(new Dictionary<string, object?>
                {
                    ["id"] = relic.Id.Entry,
                    ["name"] = SafeGetText(() => relic.Title),
                    ["description"] = SafeGetText(() => relic.DynamicDescription),
                    ["counter"] = relic.ShowCounter ? relic.DisplayAmount : null,
                });
            }
            return relics;
        }
        catch { return null; }
    }

    /// <summary>The potions you are carrying, with their slots, on every state.</summary>
    /// <remarks>
    /// Same gap as the relics, and with a sharper consequence: a potion reward
    /// arrives on a screen that reported the number of free slots but not what
    /// was in the used ones, so "is this potion better than one I have" was not
    /// an answerable question. `slot` is the number `use_potion` takes, and it is
    /// the slot index - empty slots are skipped when building this, so the two
    /// diverge as soon as slot 0 is free.
    /// </remarks>
    private static List<Dictionary<string, object?>>? BuildHeldPotions(RunState runState)
    {
        try
        {
            var me = LocalContext.GetMe(runState);
            if (me == null) return null;
            var potions = new List<Dictionary<string, object?>>();
            int slot = 0;
            foreach (var potion in me.PotionSlots)
            {
                if (potion != null)
                {
                    potions.Add(new Dictionary<string, object?>
                    {
                        ["slot"] = slot,
                        ["id"] = potion.Id.Entry,
                        ["name"] = SafeGetText(() => potion.Title),
                        ["description"] = SafeGetText(() => potion.DynamicDescription),
                        ["usage"] = potion.Usage.ToString(),
                        ["target_type"] = potion.TargetType.ToString(),
                    });
                }
                slot++;
            }
            return potions;
        }
        catch { return null; }
    }

    /// <summary>
    /// The whole run's deck, as counts rather than a list of card objects.
    /// </summary>
    /// <remarks>
    /// Added 2026-09-13 because no meta decision could see it. A card reward
    /// screen reported only the three cards on offer, so "is this card worth
    /// adding" had to be answered without knowing what it was being added to -
    /// and the deck is the whole of that question. Every card taken dilutes the
    /// rest, so past some point a mediocre card is worse than no card, and
    /// skipping is a real option that nothing could previously reason about.
    ///
    /// In combat the same information is reachable as
    /// hand + draw_pile + discard_pile + exhaust_pile, but a card reward appears
    /// *after* combat, when those piles are gone. `Player.Deck` is the run-level
    /// pile and is always there.
    ///
    /// Counts, not objects: this rides on every `/state` response, including the
    /// ~3/sec polled while waiting out an animation, and forty card objects at
    /// that rate is a lot of bytes for something that changes a few times a
    /// floor. Name -> count is also the shape the decision actually wants ("I
    /// already have five 打击").
    ///
    /// Wrapped, like `is_resolving`: if `Player.Deck` is renamed by a patch this
    /// degrades to null instead of turning every `/state` into a 500, which is
    /// what the equivalent mistake did on 2026-09-07.
    /// </remarks>
    private static Dictionary<string, object?>? BuildDeckSummary(RunState runState)
    {
        try
        {
            var me = LocalContext.GetMe(runState);
            if (me?.Deck == null) return null;

            var byName = new Dictionary<string, int>();
            var byType = new Dictionary<string, int>();
            int upgraded = 0;

            foreach (var card in me.Deck.Cards)
            {
                if (card == null) continue;
                string name = SafeGetText(() => card.Title) ?? card.Id.Entry;
                byName[name] = byName.TryGetValue(name, out var n) ? n + 1 : 1;
                string type = card.Type.ToString();
                byType[type] = byType.TryGetValue(type, out var t) ? t + 1 : 1;
                if (card.IsUpgraded) upgraded++;
            }

            return new Dictionary<string, object?>
            {
                ["size"] = me.Deck.Cards.Count,
                ["by_type"] = byType,
                ["cards"] = byName,
                ["upgraded"] = upgraded
            };
        }
        catch
        {
            return null;
        }
    }

    private static Dictionary<string, object?> BuildGameOverState(NGameOverScreen screen, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold,
            };
        }

        state["screen_type"] = nameof(NGameOverScreen);

        var returnButton = FindFirst<NReturnToMainMenuButton>(screen);
        var continueButton = FindFirst<NGameOverContinueButton>(screen);
        var viewRunButton = FindFirst<NViewRunButton>(screen);

        state["can_return_to_main_menu"] = returnButton?.IsVisibleInTree() == true && returnButton.IsEnabled;
        state["can_continue"] = continueButton?.IsVisibleInTree() == true && continueButton.IsEnabled;
        state["can_view_run"] = viewRunButton?.IsVisibleInTree() == true && viewRunButton.IsEnabled;

        var options = new List<Dictionary<string, object?>>();
        if (returnButton != null && returnButton.IsVisibleInTree())
        {
            options.Add(new Dictionary<string, object?>
            {
                ["id"] = "return_to_main_menu",
                ["title"] = "Return To Main Menu",
                ["is_enabled"] = returnButton.IsEnabled,
            });
        }
        if (continueButton != null && continueButton.IsVisibleInTree())
        {
            options.Add(new Dictionary<string, object?>
            {
                ["id"] = "continue",
                ["title"] = "Continue",
                ["is_enabled"] = continueButton.IsEnabled,
            });
        }
        if (viewRunButton != null && viewRunButton.IsVisibleInTree())
        {
            options.Add(new Dictionary<string, object?>
            {
                ["id"] = "view_run",
                ["title"] = "View Run",
                ["is_enabled"] = viewRunButton.IsEnabled,
            });
        }
        state["options"] = options;

        return state;
    }

    private static Dictionary<string, object?> BuildMenuState()
    {
        var state = new Dictionary<string, object?>();

        var root = ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root;
        var characterSelect = FindFirst<NCharacterSelectScreen>(root);
        if (characterSelect != null && characterSelect.IsVisibleInTree())
        {
            state["screen"] = "character_select";

            var characters = new List<Dictionary<string, object?>>();
            foreach (var button in FindAll<NCharacterSelectButton>(characterSelect))
            {
                var character = button.Character;
                if (character == null) continue;

                characters.Add(new Dictionary<string, object?>
                {
                    ["id"] = character.Id.Entry,
                    ["name"] = SafeGetText(() => character.Title),
                });
            }
            state["characters"] = characters;

            var ascensionPanel = FindFirst<NAscensionPanel>(characterSelect);
            if (ascensionPanel != null)
                state["ascension"] = ascensionPanel.Ascension;

            var embarkButton = FindAll<NConfirmButton>(characterSelect)
                .FirstOrDefault(button => button.IsVisibleInTree());
            state["can_start_new_game"] = embarkButton?.IsEnabled ?? false;
            state["can_continue_game"] = false;
            state["can_abandon_game"] = false;

            return state;
        }

        var mainMenu = FindFirst<NMainMenu>(root);
        if (mainMenu != null && mainMenu.IsVisibleInTree())
        {
            state["screen"] = "main_menu";

            var continueInfo = mainMenu.ContinueRunInfo;
            bool canContinue = continueInfo != null && continueInfo.IsVisibleInTree();

            state["can_continue_game"] = canContinue;
            state["can_abandon_game"] = canContinue;
            // What `ExecuteStartNewGame` actually needs, instead of the literal
            // `true` this used to be. That constant cost 90 seconds on
            // 2026-09-14: with a saved run on the menu it reported true while
            // every start attempt came back "Singleplayer button is not
            // available", so a caller polling it waited for a condition that was
            // never false and never sufficient.
            state["can_start_new_game"] =
                GetFieldValue<NMainMenuTextButton>(mainMenu, "_singleplayerButton")
                    ?.IsVisibleInTree() ?? false;
            // The abandon confirmation is a separate popup that appears a frame
            // or more after the button is clicked. Reported so a caller can tell
            // "the request has not landed yet" from "it landed and did nothing" -
            // which is exactly what could not be told apart before.
            var confirmPopup = FindFirst<NAbandonRunConfirmPopup>(root);
            state["abandon_confirm_open"] =
                confirmPopup != null && confirmPopup.IsVisibleInTree();
            return state;
        }

        state["screen"] = "unknown";
        state["can_continue_game"] = false;
        state["can_abandon_game"] = false;
        state["can_start_new_game"] = false;
        return state;
    }

    private static Dictionary<string, object?> BuildBattleState(RunState runState, CombatRoom combatRoom)
    {
        var combatState = CombatManager.Instance.DebugOnlyGetState();
        var battle = new Dictionary<string, object?>();

        if (combatState == null)
        {
            battle["error"] = "Combat state unavailable";
            return battle;
        }

        battle["round"] = combatState.RoundNumber;
        battle["turn"] = combatState.CurrentSide.ToString().ToLower();

        // Player state
        var player = LocalContext.GetMe(runState);
        // v0.107.1 removed the global CombatManager.IsPlayPhase flag; whether it
        // is your turn is now asked per player, since combat can be co-op.
        battle["is_play_phase"] =
            player != null && CombatManager.Instance.IsPartOfPlayerTurn(player);

        // Whether the game still has queued GameActions to resolve: drawing,
        // shuffling, a relic firing, an enemy attacking. An agent must not act
        // while this is true, and there is no way to infer it from a snapshot -
        // `is_play_phase` is already true while the new hand is still being
        // dealt, and `hand` being empty cannot tell "I played everything" apart
        // from "Spinning Top is about to draw me a card".
        //
        // Waiting on a timer instead does not work: the pause scales with the
        // work. Drawing one card is brief, reshuffling a 20-card discard pile is
        // not, and no single constant covers both. `IsEmpty` is false for the
        // whole of either, whatever it takes.
        //
        // Wrapped because a thrown member removal here would take the entire
        // /state endpoint down with it - which is exactly what happened on
        // 2026-09-07 when v0.107.1 deleted CombatManager.IsPlayPhase and combat
        // became 100% unreadable. A null reads as "bridge cannot tell", which a
        // caller can handle; an HTTP 500 is not.
        // Two separate questions, both of which have to be "no" before the game
        // is idle. Measured 2026-09-12: `ActionQueueSet.IsEmpty` alone is useless
        // here - it was true for the entire enemy turn and the entire deal
        // animation. That queue holds actions *waiting* to run; `GetReadyAction()`
        // takes the current one out, so an executing action is not in it. The
        // bridge enqueues through this same set, which is what made it look like
        // the right thing to ask.
        try
        {
            var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            battle["is_resolving"] =
                rm.ActionExecutor.IsRunning || !rm.ActionQueueSet.IsEmpty;
            // Which action, for diagnostics: when this turns out to be true at a
            // surprising moment, the type name says why without another rebuild.
            battle["resolving_action"] =
                rm.ActionExecutor.CurrentlyRunningAction?.GetType().Name;
        }
        catch
        {
            battle["is_resolving"] = null;
            battle["resolving_action"] = null;
        }
        if (player != null)
        {
            battle["player"] = BuildPlayerState(player);
        }

        // Enemies
        var enemies = new List<Dictionary<string, object?>>();
        var entityCounts = new Dictionary<string, int>();
        foreach (var creature in combatState.Enemies)
        {
            if (creature.IsAlive)
            {
                enemies.Add(BuildEnemyState(creature, entityCounts));
            }
        }
        battle["enemies"] = enemies;

        return battle;
    }

    private static Dictionary<string, object?> BuildPlayerState(Player player)
    {
        var state = new Dictionary<string, object?>();
        var creature = player.Creature;
        var combatState = player.PlayerCombatState;

        state["character"] = SafeGetText(() => player.Character.Title);
        state["hp"] = creature.CurrentHp;
        state["max_hp"] = creature.MaxHp;
        state["block"] = creature.Block;

        if (combatState != null)
        {
            state["energy"] = combatState.Energy;
            state["max_energy"] = combatState.MaxEnergy;

            // Stars (The Regent's resource, conditionally shown)
            if (player.Character.ShouldAlwaysShowStarCounter || combatState.Stars > 0)
            {
                state["stars"] = combatState.Stars;
            }

            // Hand
            var hand = new List<Dictionary<string, object?>>();
            int cardIndex = 0;
            foreach (var card in combatState.Hand.Cards)
            {
                hand.Add(BuildCardState(card, cardIndex));
                cardIndex++;
            }
            state["hand"] = hand;

            // Pile counts
            state["draw_pile_count"] = combatState.DrawPile.Cards.Count;
            state["discard_pile_count"] = combatState.DiscardPile.Cards.Count;
            state["exhaust_pile_count"] = combatState.ExhaustPile.Cards.Count;

            // Pile contents
            state["draw_pile"] = BuildPileCardList(combatState.DrawPile.Cards, PileType.Draw);
            state["discard_pile"] = BuildPileCardList(combatState.DiscardPile.Cards, PileType.Discard);
            state["exhaust_pile"] = BuildPileCardList(combatState.ExhaustPile.Cards, PileType.Exhaust);

            // Orbs
            if (combatState.OrbQueue.Capacity > 0)
            {
                var orbs = new List<Dictionary<string, object?>>();
                foreach (var orb in combatState.OrbQueue.Orbs)
                {
                    // Populate SmartDescription placeholders with Focus-modified values,
                    // mirroring OrbModel.HoverTips getter (OrbModel.cs:92-94)
                    string? description = SafeGetText(() =>
                    {
                        var desc = orb.SmartDescription;
                        desc.Add("energyPrefix", orb.Owner.Character.CardPool.Title);
                        desc.Add("Passive", orb.PassiveVal);
                        desc.Add("Evoke", orb.EvokeVal);
                        return desc;
                    });
                    orbs.Add(new Dictionary<string, object?>
                    {
                        ["id"] = orb.Id.Entry,
                        ["name"] = SafeGetText(() => orb.Title),
                        ["description"] = description,
                        ["passive_val"] = orb.PassiveVal,
                        ["evoke_val"] = orb.EvokeVal,
                        ["keywords"] = BuildHoverTips(orb.HoverTips)
                    });
                }
                state["orbs"] = orbs;
                state["orb_slots"] = combatState.OrbQueue.Capacity;
                state["orb_empty_slots"] = combatState.OrbQueue.Capacity - combatState.OrbQueue.Orbs.Count;
            }
        }

        state["gold"] = player.Gold;

        // Powers (status effects)
        state["status"] = BuildPowersState(creature);

        // Relics
        var relics = new List<Dictionary<string, object?>>();
        foreach (var relic in player.Relics)
        {
            relics.Add(new Dictionary<string, object?>
            {
                ["id"] = relic.Id.Entry,
                ["name"] = SafeGetText(() => relic.Title),
                ["description"] = SafeGetText(() => relic.DynamicDescription),
                ["counter"] = relic.ShowCounter ? relic.DisplayAmount : null,
                ["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic)
            });
        }
        state["relics"] = relics;

        // Potions
        var potions = new List<Dictionary<string, object?>>();
        int slotIndex = 0;
        foreach (var potion in player.PotionSlots)
        {
            if (potion != null)
            {
                potions.Add(new Dictionary<string, object?>
                {
                    ["id"] = potion.Id.Entry,
                    ["name"] = SafeGetText(() => potion.Title),
                    ["description"] = SafeGetText(() => potion.DynamicDescription),
                    ["slot"] = slotIndex,
                    ["can_use_in_combat"] = potion.Usage == PotionUsage.CombatOnly || potion.Usage == PotionUsage.AnyTime,
                    ["target_type"] = potion.TargetType.ToString(),
                    ["keywords"] = BuildHoverTips(potion.ExtraHoverTips)
                });
            }
            slotIndex++;
        }
        state["potions"] = potions;

        return state;
    }

    private static Dictionary<string, object?> BuildCardState(CardModel card, int index)
    {
        string costDisplay;
        if (card.EnergyCost.CostsX)
            costDisplay = "X";
        else
        {
            int cost = card.EnergyCost.GetAmountToSpend();
            costDisplay = cost.ToString();
        }

        card.CanPlay(out var unplayableReason, out _);

        // Star cost (The Regent's cards; CanonicalStarCost >= 0 means card has a star cost)
        string? starCostDisplay = null;
        if (card.HasStarCostX)
            starCostDisplay = "X";
        else if (card.CurrentStarCost >= 0)
            starCostDisplay = card.GetStarCostWithModifiers().ToString();

        return new Dictionary<string, object?>
        {
            ["index"] = index,
            ["id"] = card.Id.Entry,
            ["name"] = card.Title,
            ["type"] = card.Type.ToString(),
            ["cost"] = costDisplay,
            ["star_cost"] = starCostDisplay,
            ["description"] = SafeGetCardDescription(card),
            ["target_type"] = card.TargetType.ToString(),
            ["can_play"] = unplayableReason == UnplayableReason.None,
            ["unplayable_reason"] = unplayableReason != UnplayableReason.None ? unplayableReason.ToString() : null,
            ["is_upgraded"] = card.IsUpgraded,
            // Whether playing this card gains block, as the game itself knows it.
            // Added 2026-09-13 for a combat policy that has to tell "defend" from
            // "attack" without reading the description: CardModel exposes no
            // damage or block *amount* (the numbers are computed through hooks at
            // play time), so the only structured alternatives to a regex over
            // localised text are `Type` and this flag. Wrapped, because a
            // property that disappears in a patch should cost this one field and
            // not the whole `/state` response.
            ["gains_block"] = SafeBool(() => card.GainsBlock),
            ["keywords"] = BuildHoverTips(card.HoverTips)
        };
    }

    private static List<Dictionary<string, object?>> BuildPileCardList(IEnumerable<CardModel> cards, PileType pile)
    {
        var list = new List<Dictionary<string, object?>>();
        foreach (var card in cards)
        {
            list.Add(new Dictionary<string, object?>
            {
                ["name"] = SafeGetText(() => card.Title),
                ["description"] = SafeGetCardDescription(card, pile)
            });
        }
        return list;
    }

    private static Dictionary<string, object?> BuildEnemyState(Creature creature, Dictionary<string, int> entityCounts)
    {
        var monster = creature.Monster;
        string baseId = monster?.Id.Entry ?? "unknown";

        // Generate entity_id like "jaw_worm_0"
        if (!entityCounts.TryGetValue(baseId, out int count))
            count = 0;
        entityCounts[baseId] = count + 1;
        string entityId = $"{baseId}_{count}";

        var state = new Dictionary<string, object?>
        {
            ["entity_id"] = entityId,
            ["combat_id"] = creature.CombatId,
            ["name"] = SafeGetText(() => monster?.Title),
            ["hp"] = creature.CurrentHp,
            ["max_hp"] = creature.MaxHp,
            ["block"] = creature.Block,
            ["status"] = BuildPowersState(creature)
        };

        // Intents
        if (monster?.NextMove is MoveState moveState)
        {
            var intents = new List<Dictionary<string, object?>>();
            foreach (var intent in moveState.Intents)
            {
                var intentData = new Dictionary<string, object?>
                {
                    ["type"] = intent.IntentType.ToString()
                };
                try
                {
                    var targets = creature.CombatState?.PlayerCreatures;
                    if (targets != null)
                    {
                        string label = intent.GetIntentLabel(targets, creature).GetFormattedText();
                        intentData["label"] = StripRichTextTags(label);

                        var hoverTip = intent.GetHoverTip(targets, creature);
                        if (hoverTip.Title != null)
                            intentData["title"] = StripRichTextTags(hoverTip.Title);
                        if (hoverTip.Description != null)
                            intentData["description"] = StripRichTextTags(hoverTip.Description);
                    }
                }
                catch { /* intent label may fail for some types */ }
                intents.Add(intentData);
            }
            state["intents"] = intents;
        }

        return state;
    }

    private static Dictionary<string, object?> BuildEventState(EventRoom eventRoom, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold
            };
        }

        var eventModel = eventRoom.CanonicalEvent;
        bool isAncient = eventModel is AncientEventModel;
        state["event_id"] = eventModel.Id.Entry;
        state["event_name"] = SafeGetText(() => eventModel.Title);
        state["is_ancient"] = isAncient;

        // Check dialogue state for ancients
        bool inDialogue = false;
        var uiRoom = NEventRoom.Instance;
        if (isAncient && uiRoom != null)
        {
            var ancientLayout = FindFirst<NAncientEventLayout>(uiRoom);
            if (ancientLayout != null)
            {
                var hitbox = ancientLayout.GetNodeOrNull<NClickableControl>("%DialogueHitbox");
                inDialogue = hitbox != null && hitbox.Visible && hitbox.IsEnabled;
            }
        }
        state["in_dialogue"] = inDialogue;

        // Self-diagnosis for the case above being wrong. If an event ever reports
        // no options and no dialogue, something is on screen that this builder
        // cannot see, and the single most useful fact is the class name of
        // whatever is on top. Reporting it turns a silent soft-lock into a lead:
        // the fix for DROWNING_BEACON was guessed from source, and if the guess
        // was wrong this field says so by naming the real type instead.
        state["top_overlay_type"] = NOverlayStack.Instance?.Peek()?.GetType().Name;

        // Event body text
        state["body"] = SafeGetText(() => eventModel.Description);

        // Options from UI
        var options = new List<Dictionary<string, object?>>();
        if (uiRoom != null)
        {
            var buttons = FindAll<NEventOptionButton>(uiRoom);
            int index = 0;
            foreach (var button in buttons)
            {
                var opt = button.Option;
                var optData = new Dictionary<string, object?>
                {
                    ["index"] = index,
                    ["title"] = SafeGetText(() => opt.Title),
                    ["description"] = SafeGetText(() => opt.Description),
                    ["is_locked"] = opt.IsLocked,
                    ["is_proceed"] = opt.IsProceed,
                    ["was_chosen"] = opt.WasChosen
                };
                if (opt.Relic != null)
                {
                    optData["relic_name"] = SafeGetText(() => opt.Relic.Title);
                    optData["relic_description"] = SafeGetText(() => opt.Relic.DynamicDescription);
                }
                optData["keywords"] = BuildHoverTips(opt.HoverTips);
                options.Add(optData);
                index++;
            }
        }
        state["options"] = options;

        return state;
    }

    private static Dictionary<string, object?> BuildRestSiteState(RestSiteRoom restSiteRoom, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold
            };
        }

        var options = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var opt in restSiteRoom.Options)
        {
            options.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = opt.OptionId,
                ["name"] = SafeGetText(() => opt.Title),
                ["description"] = SafeGetText(() => opt.Description),
                ["is_enabled"] = opt.IsEnabled
            });
            index++;
        }
        state["options"] = options;

        var proceedButton = NRestSiteRoom.Instance?.ProceedButton;
        state["can_proceed"] = proceedButton?.IsEnabled ?? false;

        return state;
    }

    private static Dictionary<string, object?> BuildShopState(MerchantRoom merchantRoom, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold,
                ["potion_slots"] = player.PotionSlots.Count,
                ["open_potion_slots"] = player.PotionSlots.Count(s => s == null)
            };
        }

        var inventory = merchantRoom.GetLocalInventory();
        var items = new List<Dictionary<string, object?>>();
        int index = 0;

        // Cards
        foreach (var entry in inventory.CardEntries)
        {
            var item = new Dictionary<string, object?>
            {
                ["index"] = index,
                ["category"] = "card",
                ["cost"] = entry.Cost,
                ["is_stocked"] = entry.IsStocked,
                ["can_afford"] = entry.EnoughGold,
                ["on_sale"] = entry.IsOnSale
            };
            if (entry.CreationResult?.Card is { } card)
            {
                item["card_id"] = card.Id.Entry;
                item["card_name"] = SafeGetText(() => card.Title);
                item["card_type"] = card.Type.ToString();
                item["card_rarity"] = card.Rarity.ToString();
                item["card_description"] = SafeGetCardDescription(card, PileType.None);
                item["keywords"] = BuildHoverTips(card.HoverTips);
            }
            items.Add(item);
            index++;
        }

        // Relics
        foreach (var entry in inventory.RelicEntries)
        {
            var item = new Dictionary<string, object?>
            {
                ["index"] = index,
                ["category"] = "relic",
                ["cost"] = entry.Cost,
                ["is_stocked"] = entry.IsStocked,
                ["can_afford"] = entry.EnoughGold
            };
            if (entry.Model is { } relic)
            {
                item["relic_id"] = relic.Id.Entry;
                item["relic_name"] = SafeGetText(() => relic.Title);
                item["relic_description"] = SafeGetText(() => relic.DynamicDescription);
                item["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic);
            }
            items.Add(item);
            index++;
        }

        // Potions
        foreach (var entry in inventory.PotionEntries)
        {
            var item = new Dictionary<string, object?>
            {
                ["index"] = index,
                ["category"] = "potion",
                ["cost"] = entry.Cost,
                ["is_stocked"] = entry.IsStocked,
                ["can_afford"] = entry.EnoughGold
            };
            if (entry.Model is { } potion)
            {
                item["potion_id"] = potion.Id.Entry;
                item["potion_name"] = SafeGetText(() => potion.Title);
                item["potion_description"] = SafeGetText(() => potion.DynamicDescription);
                item["keywords"] = BuildHoverTips(potion.ExtraHoverTips);
            }
            items.Add(item);
            index++;
        }

        // Card removal
        if (inventory.CardRemovalEntry is { } removal)
        {
            items.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["category"] = "card_removal",
                ["cost"] = removal.Cost,
                ["is_stocked"] = removal.IsStocked,
                ["can_afford"] = removal.EnoughGold
            });
        }

        state["items"] = items;

        var proceedButton = NMerchantRoom.Instance?.ProceedButton;
        state["can_proceed"] = proceedButton?.IsEnabled ?? false;

        return state;
    }

    private static Dictionary<string, object?> BuildMapState(RunState runState)
    {
        var state = new Dictionary<string, object?>();

        // Player summary
        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            int totalSlots = player.PotionSlots.Count;
            int openSlots = player.PotionSlots.Count(s => s == null);
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold,
                ["potion_slots"] = totalSlots,
                ["open_potion_slots"] = openSlots
            };
        }

        var map = runState.Map;
        var visitedCoords = runState.VisitedMapCoords;

        // Current position
        if (visitedCoords.Count > 0)
        {
            var cur = visitedCoords[visitedCoords.Count - 1];
            state["current_position"] = new Dictionary<string, object?>
            {
                ["col"] = cur.col, ["row"] = cur.row,
                ["type"] = map.GetPoint(cur)?.PointType.ToString()
            };
        }

        // Visited path
        var visited = new List<Dictionary<string, object?>>();
        foreach (var coord in visitedCoords)
        {
            visited.Add(new Dictionary<string, object?>
            {
                ["col"] = coord.col, ["row"] = coord.row,
                ["type"] = map.GetPoint(coord)?.PointType.ToString()
            });
        }
        state["visited"] = visited;

        // Next options — read travelable state from UI nodes
        var nextOptions = new List<Dictionary<string, object?>>();
        var mapScreen = NMapScreen.Instance;
        if (mapScreen != null)
        {
            var travelable = FindAll<NMapPoint>(mapScreen)
                .Where(mp => mp.State == MapPointState.Travelable)
                .OrderBy(mp => mp.Point.coord.col)
                .ToList();

            int index = 0;
            foreach (var nmp in travelable)
            {
                var pt = nmp.Point;
                var option = new Dictionary<string, object?>
                {
                    ["index"] = index,
                    ["col"] = pt.coord.col,
                    ["row"] = pt.coord.row,
                    ["type"] = pt.PointType.ToString()
                };

                // 1-level lookahead
                var children = pt.Children
                    .OrderBy(c => c.coord.col)
                    .Select(c => new Dictionary<string, object?>
                    {
                        ["col"] = c.coord.col, ["row"] = c.coord.row,
                        ["type"] = c.PointType.ToString()
                    }).ToList();
                if (children.Count > 0)
                    option["leads_to"] = children;

                nextOptions.Add(option);
                index++;
            }
        }
        state["next_options"] = nextOptions;

        // Full map — all nodes organized for planning
        var nodes = new List<Dictionary<string, object?>>();

        // Starting point
        var start = map.StartingMapPoint;
        nodes.Add(BuildMapNode(start));

        // Grid nodes
        foreach (var pt in map.GetAllMapPoints())
            nodes.Add(BuildMapNode(pt));

        // Boss
        nodes.Add(BuildMapNode(map.BossMapPoint));
        if (map.SecondBossMapPoint != null)
            nodes.Add(BuildMapNode(map.SecondBossMapPoint));

        state["nodes"] = nodes;
        state["boss"] = new Dictionary<string, object?>
        {
            ["col"] = map.BossMapPoint.coord.col,
            ["row"] = map.BossMapPoint.coord.row
        };

        return state;
    }

    private static Dictionary<string, object?> BuildMapNode(MapPoint pt)
    {
        return new Dictionary<string, object?>
        {
            ["col"] = pt.coord.col,
            ["row"] = pt.coord.row,
            ["type"] = pt.PointType.ToString(),
            ["children"] = pt.Children
                .OrderBy(c => c.coord.col)
                .Select(c => new List<int> { c.coord.col, c.coord.row })
                .ToList()
        };
    }

    /// <summary>
    /// Does this rewards screen still have anything for the agent to do?
    /// </summary>
    /// <remarks>
    /// Mirrors the two places that consume the screen, on purpose:
    /// BuildRewardsState numbers exactly the buttons with
    /// <c>Reward != null &amp;&amp; IsEnabled</c>, and ExecuteClaimReward indexes
    /// that same filtered list. Keeping all three in step is what makes "items
    /// is empty" mean "nothing is claimable" instead of "nothing got listed".
    ///
    /// The proceed button counts because it is the screen's only other
    /// affordance: rewards all claimed, room not yet left. It lives on the room
    /// rather than on this screen - see FindLiveProceedButton.
    ///
    /// <c>IsComplete</c> is deliberately *not* the test, though it looked like
    /// the obvious choice. It has only ever been observed true, in the wedged
    /// state of 2026-09-13; what it reads while rewards are still unclaimed has
    /// never been measured. If it is true there too, keying on it would walk the
    /// agent past its own gold and cards without a word - a silent loss, which is
    /// worse than the loud stall it would fix. It stays reported in the state so
    /// that the next run measures it rather than anyone reasoning about it.
    ///
    /// The potion clause is the second half of the same lesson, and it cost
    /// another wedge to learn. The first version of this method took
    /// <c>IsEnabled</c> to mean "clicking this does something". It does not: a
    /// potion reward offered with every slot full stays enabled for ever.
    /// Measured at act 1 / floor 9 - <c>claim_reward</c> returned
    /// <c>status: ok</c> ("Claiming reward: potion (灰水)") and the state was
    /// byte-identical 3.2s later, while the map sat open underneath with one
    /// travelable node. Because potion slots stay full until something drinks
    /// one, and nothing in this bridge can, it then happened after *every* fight:
    /// floor 9, floor 11, and it would have been every floor after that.
    ///
    /// Potions are the only reward with a capacity limit, so this is the only
    /// such clause. If another reward type ever turns out to be enabled-but-inert
    /// it belongs here next to this one, not in a guard somewhere else.
    /// </remarks>
    private static bool RewardsScreenHasWorkLeft(NRewardsScreen screen, RunState runState)
    {
        var me = LocalContext.GetMe(runState);
        bool potionsFull = me != null && me.PotionSlots.All(slot => slot != null);

        foreach (var button in FindAll<NRewardButton>(screen))
        {
            if (button.Reward == null || !button.IsEnabled) continue;
            // Enabled, but it cannot land anywhere. Not work.
            if (potionsFull && button.Reward is PotionReward) continue;
            return true;
        }
        return FindLiveProceedButton() != null;
    }

    /// <summary>
    /// Rewards this screen is listing that cannot actually be taken, for the log.
    /// </summary>
    /// <remarks>
    /// Only called when the screen is about to be treated as spent, so that
    /// abandoning a reward is stated rather than inferred from its absence.
    /// Leaving a potion behind is the right move - freeing a slot means drinking
    /// one, and which potion to drink when is a policy decision - but it should
    /// never look like the reward was simply missed.
    /// </remarks>
    private static int CountInertRewards(NRewardsScreen screen, RunState runState)
    {
        var me = LocalContext.GetMe(runState);
        if (me == null || !me.PotionSlots.All(slot => slot != null)) return 0;
        return FindAll<NRewardButton>(screen)
            .Count(b => b.Reward is PotionReward && b.IsEnabled);
    }

    private static Dictionary<string, object?> BuildRewardsState(NRewardsScreen rewardsScreen, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        // Player summary for decision-making context
        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            int totalSlots = player.PotionSlots.Count;
            int openSlots = player.PotionSlots.Count(s => s == null);
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold,
                ["potion_slots"] = totalSlots,
                ["open_potion_slots"] = openSlots
            };
        }

        // Reward items
        var rewardButtons = FindAll<NRewardButton>(rewardsScreen);
        var items = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var button in rewardButtons)
        {
            if (button.Reward == null || !button.IsEnabled) continue;
            var reward = button.Reward;

            var item = new Dictionary<string, object?>
            {
                ["index"] = index,
                ["type"] = GetRewardTypeName(reward),
                ["description"] = SafeGetText(() => reward.Description)
            };

            // Type-specific details
            if (reward is GoldReward goldReward)
                item["gold_amount"] = goldReward.Amount;
            else if (reward is PotionReward potionReward && potionReward.Potion != null)
            {
                item["potion_id"] = potionReward.Potion.Id.Entry;
                item["potion_name"] = SafeGetText(() => potionReward.Potion.Title);
            }

            // ⚠️ The authoritative "is this reward done with", added 2026-09-16.
            // Until now this screen was described entirely by `IsEnabled` on the
            // buttons, and that turned out to be the wrong question twice in one
            // day: a potion reward with no free slot stays enabled for ever, and
            // a *declined* card reward stays enabled and stays listed. So a
            // frozen-deck run skipped the same entry eight times in a row while
            // the reported `items` never shrank.
            //
            // `Reward.SuccessfullySelected` is the game's own flag and it is what
            // the room is waiting on. Reported per item rather than used to
            // filter, deliberately: if it turns out to stay false for a declined
            // reward too, filtering would make the entry vanish from the state
            // and the run would walk past its own gold in silence. Measure it
            // first, then decide - the same reason `is_complete` is reported and
            // not acted on.
            item["successfully_selected"] = reward.SuccessfullySelected;
            item["is_populated"] = reward.IsPopulated;
            // `CanSkip` exists on CardReward, not on the Reward base, so it is
            // read by reflection and reported as null where the type has no such
            // property. Null means "this type does not say", which is different
            // from false - and conflating those is how the potion clause above
            // got written twice.
            item["can_skip"] = reward.GetType().GetProperty("CanSkip")?.GetValue(reward);

            items.Add(item);
            index++;
        }
        state["items"] = items;

        // Proceed button
        // Not FindFirst<NProceedButton>(rewardsScreen): the button belongs to the
        // room underneath, not to this overlay. See FindLiveProceedButton.
        var proceedButton = FindLiveProceedButton();
        // The screen's own verdict on whether everything has been taken. Worth
        // reporting next to `items`, because "no items left" and "this screen is
        // finished" are not guaranteed to coincide.
        state["is_complete"] = rewardsScreen.IsComplete;
        state["can_proceed"] = proceedButton?.IsEnabled ?? false;

        // Read-only diagnostics, kept: these are what settled "is the card
        // reward resolvable" (DisallowSkipping false, CanSkip true) and
        // "did that call do anything" (SuccessfullySelected stayed false through
        // four different attempts). None of them changes behaviour.
        var rewardsSet = GetFieldValue<RewardsSet>(rewardsScreen, "_rewardsSet");
        if (rewardsSet != null)
        {
            state["all_rewards_selected"] = rewardsSet.AllRewardsSuccessfullySelected;
            state["disallow_skipping"] = rewardsSet.DisallowSkipping;
            state["rewards_in_set"] = rewardsSet.Rewards?.Count;
        }
        state["skipped_button_count"] =
            GetFieldValue<System.Collections.ICollection>(rewardsScreen, "_skippedRewardButtons")?.Count;
        state["reward_button_count"] =
            GetFieldValue<System.Collections.ICollection>(rewardsScreen, "_rewardButtons")?.Count;

        return state;
    }

    private static Dictionary<string, object?> BuildCardRewardState(NCardRewardSelectionScreen cardScreen)
    {
        var state = new Dictionary<string, object?>();

        var cardHolders = FindAllSortedByPosition<NCardHolder>(cardScreen);
        var cards = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in cardHolders)
        {
            var card = holder.CardModel;
            if (card == null) continue;

            string costDisplay = card.EnergyCost.CostsX
                ? "X"
                : card.EnergyCost.GetAmountToSpend().ToString();

            string? starCostDisplay = null;
            if (card.HasStarCostX)
                starCostDisplay = "X";
            else if (card.CurrentStarCost >= 0)
                starCostDisplay = card.GetStarCostWithModifiers().ToString();

            cards.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = costDisplay,
                ["star_cost"] = starCostDisplay,
                ["description"] = SafeGetCardDescription(card, PileType.None),
                ["rarity"] = card.Rarity.ToString(),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
            index++;
        }
        state["cards"] = cards;

        // Every alternative, not just "is there one". These buttons are not all
        // "skip": the game models each as a CardRewardAlternative with its own
        // OptionId/Title, CardReward tracks CanSkip and CanReroll separately,
        // and Hook.ModifyCardRewardAlternatives lets a relic add more - Pael's
        // Wing (PAELS_WING) adds a sacrifice. So the count is a function of the
        // run, and collapsing the list to one bool left the policy picking
        // blind: the handler clicked altButtons[0] whatever that happened to be,
        // and reported "Skipping card reward" either way.
        var altButtons = FindAll<NCardRewardAlternativeButton>(cardScreen);
        var extras = ReadExtraOptions(cardScreen);
        var alternatives = new List<Dictionary<string, object?>>();
        for (int i = 0; i < altButtons.Count; i++)
            alternatives.Add(DescribeAlternative(i, altButtons[i], extras));
        state["alternatives"] = alternatives;

        // Kept so existing callers keep working, but it is now the weaker claim
        // it always was: "an alternative exists", which stopped implying "and it
        // is skip" the moment a relic could prepend one.
        state["can_skip"] = altButtons.Count > 0;

        return state;
    }

    /// <summary>
    /// One entry of the card reward's alternative row, by position.
    /// </summary>
    /// <remarks>
    /// Both the builder and <c>ExecuteSkipCardReward</c> call this over the same
    /// <c>FindAll&lt;NCardRewardAlternativeButton&gt;</c> enumeration, so the
    /// index the agent is shown and the index the handler clicks cannot drift
    /// apart. That is deliberate: every index bug in this bridge so far came
    /// from a builder and a handler walking two different lists and agreeing
    /// only by accident.
    ///
    /// The identity is read privately. <c>NCardRewardAlternativeButton</c>
    /// publishes nothing but a Godot Control - no Title, no Option, no Model -
    /// while holding <c>_optionName</c> and <c>_label</c> internally, so a null
    /// here means the field was renamed, not that the button is anonymous.
    /// </remarks>
    /// <summary>
    /// Every alternative on this screen, keyed by displayed title, for its stable id.
    /// </summary>
    /// <remarks>
    /// The buttons carry a localised label and nothing else. That was found the
    /// hard way on 2026-09-14: `_optionName` was reported as `option_id` on the
    /// assumption it held the model's stable `OptionId`, and a live run came back
    /// with <c>"option_id": "跳过"</c> - the display string, in whatever language
    /// the game is running. A policy matching on "SKIP" would never fire in a
    /// Chinese client, and would happily pick a sacrifice instead.
    ///
    /// The stable ids live on <c>CardRewardAlternative.OptionId</c>, reachable
    /// through the screen's <c>_extraOptions</c> - which, despite the name, holds
    /// the ordinary skip too ("Skip", confirmed live). Keyed by title rather than
    /// by position, because nothing proves the button order matches the list
    /// order - the assumption this whole area keeps getting wrong.
    /// </remarks>
    private static Dictionary<string, CardRewardAlternative> ReadExtraOptions(
        NCardRewardSelectionScreen screen)
    {
        var byTitle = new Dictionary<string, CardRewardAlternative>();
        var raw = GetFieldValue<System.Collections.IEnumerable>(screen, "_extraOptions");
        if (raw == null) return byTitle;
        try
        {
            foreach (var item in raw)
            {
                if (item is not CardRewardAlternative alt) continue;
                var title = SafeGetText(() => alt.Title);
                if (!string.IsNullOrWhiteSpace(title))
                    byTitle[title] = alt;
            }
        }
        catch { /* a renamed field costs the ids, not the screen */ }
        return byTitle;
    }

    private static Dictionary<string, object?> DescribeAlternative(
        int index, NCardRewardAlternativeButton button,
        Dictionary<string, CardRewardAlternative> extras)
    {
        var label = GetFieldValue<Godot.Label>(button, "_label");
        var text = label == null ? null : StripRichTextTags(label.Text);
        // `_optionName` is the *label*, measured 2026-09-14 - it came back "跳过".
        // Reported as `title` alongside the label rather than as an id, so nothing
        // downstream can mistake a localised string for a stable one again.
        var name = GetFieldValue<string>(button, "_optionName");
        var display = string.IsNullOrWhiteSpace(text) ? name : text;

        CardRewardAlternative? alt = null;
        if (display != null) extras.TryGetValue(display, out alt);
        string? optionId = alt?.OptionId;

        return new Dictionary<string, object?>
        {
            ["index"] = index,
            ["title"] = display,
            // ⚠️ The field that decides whether this option can decline the
            // reward at all, exposed 2026-09-16 after three failed attempts to
            // walk past a card reward. `PostAlternateCardRewardAction` has both
            // `EndSelectionAndCompleteReward` and
            // `EndSelectionAndDoNotCompleteReward`, so an option can close the
            // screen and leave the reward pending - which is exactly the state a
            // frozen-deck run kept ending up in, with the rewards screen's
            // proceed button disabled until the card was resolved.
            //
            // Read it before assuming any option is an exit.
            ["after_selected"] = alt?.AfterSelected.ToString(),
            // The stable id, e.g. "Skip". Measured live 2026-09-14: every
            // alternative is in `_extraOptions`, including the ordinary skip - so
            // "is it an extra" turned out to mean nothing more than "did the
            // lookup succeed", and the boolean that said so has been removed
            // rather than left to be misread. A null here means the title did not
            // match any entry, i.e. one of these private field names has moved.
            ["option_id"] = optionId,
            ["is_enabled"] = SafeBool(() => button.IsEnabled),
        };
    }

    /// <summary>
    /// A bundle-choice screen, shaped like a card selection so policies need no
    /// new branch.
    /// </summary>
    /// <remarks>
    /// Each entry is one bundle: its `index` is what `select_card` takes, its
    /// `cards` are what taking it adds to the deck. The summary `name` exists
    /// because a policy reading a list of three unnamed groups has nothing to
    /// choose between - and the deck summary now on every state is what makes
    /// "three more attacks" a judgeable offer rather than a coin flip.
    /// </remarks>
    private static Dictionary<string, object?> BuildBundleState(
        NChooseABundleSelectionScreen screen, RunState runState)
    {
        var state = new Dictionary<string, object?> { ["screen_type"] = "bundle" };

        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold
            };
        }

        var bundles = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var bundle in FindAllSortedByPosition<NCardBundle>(screen))
        {
            var cards = new List<Dictionary<string, object?>>();
            try
            {
                foreach (var card in bundle.Bundle)
                {
                    if (card == null) continue;
                    cards.Add(new Dictionary<string, object?>
                    {
                        ["id"] = card.Id.Entry,
                        ["name"] = SafeGetText(() => card.Title),
                        ["type"] = card.Type.ToString(),
                        ["cost"] = card.EnergyCost.CostsX
                            ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                        ["rarity"] = card.Rarity.ToString(),
                        ["description"] = SafeGetCardDescription(card, PileType.None),
                    });
                }
            }
            catch { /* a bundle that will not enumerate still gets an index */ }

            bundles.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["name"] = string.Join(" + ", cards.ConvertAll(c => c["name"]?.ToString() ?? "?")),
                ["cards"] = cards,
            });
            index++;
        }

        state["prompt"] = "Choose one bundle.";
        state["cards"] = bundles;      // the key every card_select branch reads
        state["can_confirm"] = false;  // clicking a bundle takes it; nothing to confirm
        state["can_skip"] = false;
        return state;
    }

    private static Dictionary<string, object?> BuildCardSelectState(NCardGridSelectionScreen screen, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        // Screen type
        state["screen_type"] = screen switch
        {
            NDeckTransformSelectScreen => "transform",
            NDeckUpgradeSelectScreen => "upgrade",
            NDeckCardSelectScreen => "select",
            NSimpleCardSelectScreen => "simple_select",
            _ => screen.GetType().Name
        };

        // Player summary
        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold
            };
        }

        // Prompt text from UI label
        var bottomLabel = screen.GetNodeOrNull("%BottomLabel");
        if (bottomLabel != null)
        {
            var textVariant = bottomLabel.Get("text");
            string? prompt = textVariant.VariantType != Godot.Variant.Type.Nil ? StripRichTextTags(textVariant.AsString()) : null;
            state["prompt"] = prompt;
        }

        // Cards in the grid (sorted by visual position — MoveToFront can reorder children)
        var cardHolders = FindAllSortedByPosition<NGridCardHolder>(screen);
        var cards = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in cardHolders)
        {
            var card = holder.CardModel;
            if (card == null) continue;

            cards.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                ["description"] = SafeGetCardDescription(card, PileType.None),
                ["rarity"] = card.Rarity.ToString(),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
            index++;
        }
        state["cards"] = cards;

        // Preview container showing? (selection complete, awaiting confirm)
        // Upgrade screens use UpgradeSinglePreviewContainer / UpgradeMultiPreviewContainer
        var previewSingle = screen.GetNodeOrNull<Godot.Control>("%UpgradeSinglePreviewContainer");
        var previewMulti = screen.GetNodeOrNull<Godot.Control>("%UpgradeMultiPreviewContainer");
        var previewGeneric = screen.GetNodeOrNull<Godot.Control>("%PreviewContainer");
        bool previewShowing = (previewSingle?.Visible ?? false)
                            || (previewMulti?.Visible ?? false)
                            || (previewGeneric?.Visible ?? false);
        state["preview_showing"] = previewShowing;

        // Button states
        var closeButton = screen.GetNodeOrNull<NBackButton>("%Close");
        state["can_cancel"] = closeButton?.IsEnabled ?? false;

        // Confirm button — search all preview containers and main screen
        bool canConfirm = false;
        foreach (var container in new[] { previewSingle, previewMulti, previewGeneric })
        {
            if (container?.Visible == true)
            {
                var confirm = container.GetNodeOrNull<NConfirmButton>("Confirm")
                              ?? container.GetNodeOrNull<NConfirmButton>("%PreviewConfirm");
                if (confirm?.IsEnabled == true) { canConfirm = true; break; }
            }
        }
        if (!canConfirm)
        {
            var mainConfirm = screen.GetNodeOrNull<NConfirmButton>("Confirm")
                              ?? screen.GetNodeOrNull<NConfirmButton>("%Confirm");
            if (mainConfirm?.IsEnabled == true) canConfirm = true;
        }
        // Fallback: search entire screen tree for any enabled confirm button
        // (covers subclasses like NDeckEnchantSelectScreen)
        if (!canConfirm)
        {
            canConfirm = FindAll<NConfirmButton>(screen).Any(b => b.IsEnabled && b.IsVisibleInTree());
        }
        state["can_confirm"] = canConfirm;

        // How many are picked, and how many the screen wants.
        //
        // Without these a policy can only guess, and the guess it made was "keep
        // picking indices nobody has picked yet, the count must go up eventually
        // and confirm must light up". That is wrong for any screen asking for an
        // exact number: on 2026-09-14 a run wedged on 选择2张牌来移除 (14 cards,
        // pick 2), reported `picked all 14 cards and can_confirm never became
        // true`, and gave up - 23 wasted decisions and a dead run, twice.
        //
        // The screen knows both numbers and neither is public: `_selectedCards`
        // is the live set, `_prefs` carries MinSelect/MaxSelect. Read here so the
        // policy can pick until it has the right number and then confirm, instead
        // of inferring from a button that only lights up once it already has.
        var selected = GetFieldValue<System.Collections.ICollection>(screen, "_selectedCards");
        state["selected_count"] = selected?.Count;

        var prefsObj = GetFieldValue<object>(screen, "_prefs");
        if (prefsObj is CardSelectorPrefs prefs)
        {
            state["min_select"] = prefs.MinSelect;
            state["max_select"] = prefs.MaxSelect;
            state["requires_confirmation"] = prefs.RequireManualConfirmation;
        }

        // Which of the two confirmation stages is showing. The deck screens
        // confirm twice: pick enough cards, press confirm, then confirm again in
        // a preview of what the picks will do.
        var previewContainer = screen.GetNodeOrNull<Godot.Control>("%PreviewContainer");
        state["preview_showing"] = previewContainer?.Visible == true;

        return state;
    }

    private static Dictionary<string, object?> BuildChooseCardState(NChooseACardSelectionScreen screen, RunState runState)
    {
        var state = new Dictionary<string, object?>();
        state["screen_type"] = "choose";

        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold
            };
        }

        state["prompt"] = "Choose a card.";

        var cardHolders = FindAllSortedByPosition<NGridCardHolder>(screen);
        var cards = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in cardHolders)
        {
            var card = holder.CardModel;
            if (card == null) continue;

            cards.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                ["description"] = SafeGetCardDescription(card, PileType.None),
                ["rarity"] = card.Rarity.ToString(),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
            index++;
        }
        state["cards"] = cards;

        var skipButton = screen.GetNodeOrNull<NClickableControl>("SkipButton");
        state["can_skip"] = skipButton?.IsEnabled == true && skipButton.Visible;
        state["preview_showing"] = false;
        state["can_confirm"] = false;
        state["can_cancel"] = state["can_skip"];

        return state;
    }

    private static Dictionary<string, object?> BuildHandSelectState(NPlayerHand hand, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        // Mode
        state["mode"] = hand.CurrentMode switch
        {
            NPlayerHand.Mode.SimpleSelect => "simple_select",
            NPlayerHand.Mode.UpgradeSelect => "upgrade_select",
            _ => hand.CurrentMode.ToString()
        };

        // Prompt text from %SelectionHeader
        var headerLabel = hand.GetNodeOrNull<Godot.Control>("%SelectionHeader");
        if (headerLabel != null)
        {
            var textVariant = headerLabel.Get("text");
            string? prompt = textVariant.VariantType != Godot.Variant.Type.Nil
                ? StripRichTextTags(textVariant.AsString())
                : null;
            state["prompt"] = prompt;
        }

        // Selectable cards (visible holders in the hand)
        var selectableCards = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in hand.ActiveHolders)
        {
            var card = holder.CardModel;
            if (card == null) continue;

            selectableCards.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                ["description"] = SafeGetCardDescription(card),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
            index++;
        }
        state["cards"] = selectableCards;

        // Already-selected cards (in the SelectedHandCardContainer)
        var selectedContainer = hand.GetNodeOrNull<Godot.Control>("%SelectedHandCardContainer");
        if (selectedContainer != null)
        {
            var selectedCards = new List<Dictionary<string, object?>>();
            var selectedHolders = FindAll<NSelectedHandCardHolder>(selectedContainer);
            int selIdx = 0;
            foreach (var holder in selectedHolders)
            {
                var card = holder.CardModel;
                if (card == null) continue;
                selectedCards.Add(new Dictionary<string, object?>
                {
                    ["index"] = selIdx,
                    ["name"] = SafeGetText(() => card.Title)
                });
                selIdx++;
            }
            if (selectedCards.Count > 0)
                state["selected_cards"] = selectedCards;
        }

        // Confirm button state
        var confirmBtn = hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton");
        state["can_confirm"] = confirmBtn?.IsEnabled ?? false;

        return state;
    }

    private static Dictionary<string, object?> BuildRelicSelectState(NChooseARelicSelection screen, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold
            };
        }

        state["prompt"] = "Choose a relic.";

        var relicHolders = FindAll<NRelicBasicHolder>(screen);
        var relics = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in relicHolders)
        {
            var relic = holder.Relic?.Model;
            if (relic == null) continue;

            relics.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = relic.Id.Entry,
                ["name"] = SafeGetText(() => relic.Title),
                ["description"] = SafeGetText(() => relic.DynamicDescription),
                ["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic)
            });
            index++;
        }
        state["relics"] = relics;

        var skipButton = screen.GetNodeOrNull<NClickableControl>("SkipButton");
        state["can_skip"] = skipButton?.IsEnabled == true && skipButton.Visible;

        return state;
    }

    private static Dictionary<string, object?> BuildTreasureState(TreasureRoom treasureRoom, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
        {
            state["player"] = new Dictionary<string, object?>
            {
                ["character"] = SafeGetText(() => player.Character.Title),
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["gold"] = player.Gold
            };
        }

        var treasureUI = FindFirst<NTreasureRoom>(
            ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);

        if (treasureUI == null)
        {
            state["message"] = "Treasure room loading...";
            return state;
        }

        // Auto-open chest if not yet opened
        var chestButton = treasureUI.GetNodeOrNull<NClickableControl>("Chest");
        if (chestButton is { IsEnabled: true })
        {
            chestButton.ForceClick();
            state["message"] = "Opening chest...";
            return state;
        }

        // Show relics available for picking
        var relicCollection = treasureUI.GetNodeOrNull<NTreasureRoomRelicCollection>("%RelicCollection");
        if (relicCollection?.Visible == true)
        {
            var holders = FindAll<NTreasureRoomRelicHolder>(relicCollection)
                .Where(h => h.IsEnabled && h.Visible)
                .ToList();

            var relics = new List<Dictionary<string, object?>>();
            int index = 0;
            foreach (var holder in holders)
            {
                var relic = holder.Relic?.Model;
                if (relic == null) continue;
                relics.Add(new Dictionary<string, object?>
                {
                    ["index"] = index,
                    ["id"] = relic.Id.Entry,
                    ["name"] = SafeGetText(() => relic.Title),
                    ["description"] = SafeGetText(() => relic.DynamicDescription),
                    ["rarity"] = relic.Rarity.ToString(),
                    ["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic)
                });
                index++;
            }
            state["relics"] = relics;
        }

        state["can_proceed"] = treasureUI.ProceedButton?.IsEnabled ?? false;

        return state;
    }

    private static string GetRewardTypeName(Reward reward) => reward switch
    {
        GoldReward => "gold",
        PotionReward => "potion",
        RelicReward => "relic",
        CardReward => "card",
        SpecialCardReward => "special_card",
        CardRemovalReward => "card_removal",
        _ => reward.GetType().Name.ToLower()
    };

    private static List<Dictionary<string, object?>> BuildPowersState(Creature creature)
    {
        var powers = new List<Dictionary<string, object?>>();
        foreach (var power in creature.Powers)
        {
            if (!power.IsVisible) continue;

            // HoverTips resolves all dynamic vars (Amount, DynamicVars, etc.)
            // The first tip is the power's own description; the rest are extra keywords
            var allTips = power.HoverTips.ToList();
            string? resolvedDesc = null;
            var extraTips = new List<IHoverTip>();
            foreach (var tip in allTips)
            {
                if (tip.Id == power.Id.ToString())
                {
                    // This is the power's own hover tip — extract its resolved description
                    if (tip is HoverTip ht)
                        resolvedDesc = StripRichTextTags(ht.Description);
                }
                else
                {
                    extraTips.Add(tip);
                }
            }
            // Fallback to raw SmartDescription if HoverTips extraction failed
            resolvedDesc ??= SafeGetText(() => power.SmartDescription);

            powers.Add(new Dictionary<string, object?>
            {
                ["id"] = power.Id.Entry,
                ["name"] = SafeGetText(() => power.Title),
                ["amount"] = power.DisplayAmount,
                ["type"] = power.Type.ToString(),
                ["description"] = resolvedDesc,
                ["keywords"] = BuildHoverTips(extraTips)
            });
        }
        return powers;
    }
}
