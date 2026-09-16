using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Runs;

namespace STS2_Bridge;

public static partial class BridgeMod
{
    private static Dictionary<string, object?> ExecuteAction(string action, Dictionary<string, JsonElement> data)
    {
        if (!RunManager.Instance.IsInProgress)
            return ExecuteMenuAction(action, data);

        var runState = RunManager.Instance.DebugOnlyGetState()!;
        var player = LocalContext.GetMe(runState);
        if (player == null)
            return Error("Could not find local player");

        return action switch
        {
            "play_card" => ExecutePlayCard(player, data),
            "use_potion" => ExecuteUsePotion(player, data),
            "discard_potion" => ExecuteDiscardPotion(player, data),
            "end_turn" => ExecuteEndTurn(player),
            "choose_map_node" => ExecuteChooseMapNode(data),
            "choose_event_option" => ExecuteChooseEventOption(data),
            "advance_dialogue" => ExecuteAdvanceDialogue(),
            "choose_rest_option" => ExecuteChooseRestOption(data),
            "shop_purchase" => ExecuteShopPurchase(player, data),
            "claim_reward" => ExecuteClaimReward(data),
            "select_card_reward" => ExecuteSelectCardReward(data),
            "skip_card_reward" => ExecuteSkipCardReward(data),
            "proceed" => ExecuteProceed(),
            "select_card" => ExecuteSelectCard(data),
            "confirm_selection" => ExecuteConfirmSelection(),
            "cancel_selection" => ExecuteCancelSelection(),
            "combat_select_card" => ExecuteCombatSelectCard(data),
            "combat_confirm_selection" => ExecuteCombatConfirmSelection(),
            "select_relic" => ExecuteSelectRelic(data),
            "skip_relic_selection" => ExecuteSkipRelicSelection(),
            "claim_treasure_relic" => ExecuteClaimTreasureRelic(data),
            "return_to_main_menu" => ExecuteReturnToMainMenu(),
            _ => Error($"Unknown action: {action}")
        };
    }

    private static Dictionary<string, object?> ExecuteMenuAction(string action, Dictionary<string, JsonElement> data)
    {
        return action switch
        {
            "continue_game" => ExecuteContinueGame(),
            "start_new_game" => ExecuteStartNewGame(data),
            "abandon_game" => ExecuteAbandonGame(),
            "return_to_main_menu" => ExecuteReturnToMainMenu(),
            _ => Error("No run in progress")
        };
    }

    private static Dictionary<string, object?> ExecuteContinueGame()
    {
        var root = ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root;
        var mainMenu = FindFirst<NMainMenu>(root);
        if (mainMenu == null || !mainMenu.IsVisibleInTree())
            return Error("Main menu is not open");

        var continueInfo = mainMenu.ContinueRunInfo;
        if (continueInfo == null || !continueInfo.IsVisibleInTree())
            return Error("No continueable run found");

        var continueButton = GetFieldValue<NMainMenuTextButton>(mainMenu, "_continueButton");
        if (continueButton == null || !continueButton.IsVisibleInTree())
            return Error("Continue button is not available on this game build");

        continueButton.ForceClick();
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Continuing saved run"
        };
    }

    /// <summary>Abandon the saved run. Two calls: request, then confirm.</summary>
    /// <remarks>
    /// Split because the confirmation popup is created asynchronously. The old
    /// version clicked the abandon button and then looked for the popup in the
    /// same call, which always found nothing - so the confirmation was never
    /// pressed, the save survived, and the reply still said
    /// <c>"status":"ok", "message":"Abandoning saved run"</c>. Measured
    /// 2026-09-14: after that call, <c>can_continue_game</c> stayed true for the
    /// twelve seconds it was watched, and `--restart-game` could never get to a
    /// fresh run because there was always a save in the way.
    ///
    /// A single call cannot fix this by waiting: the popup is built by the game's
    /// main loop, and this handler is what the caller is blocking on. So the wait
    /// belongs to the caller, and the reply says which half happened - the same
    /// shape as the treasure chest, where reading the state opens it and the
    /// relic shows up a poll or two later.
    /// </remarks>
    private static Dictionary<string, object?> ExecuteAbandonGame()
    {
        var root = ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root;

        // Confirm first: on the second call the popup is up, and that is the
        // click that actually abandons anything.
        var popup = FindFirst<NAbandonRunConfirmPopup>(root);
        if (popup != null && popup.IsVisibleInTree())
        {
            var yesButton = FindAll<NPopupYesNoButton>(popup)
                .FirstOrDefault(button => button.IsVisibleInTree());
            if (yesButton == null)
                return Error("Abandon confirmation is open but has no visible yes button");
            yesButton.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Confirmed abandoning the saved run"
            };
        }

        var mainMenu = FindFirst<NMainMenu>(root);
        if (mainMenu == null || !mainMenu.IsVisibleInTree())
            return Error("Main menu is not open");

        var continueInfo = mainMenu.ContinueRunInfo;
        if (continueInfo == null || !continueInfo.IsVisibleInTree())
            return Error("No continueable run found");

        var abandonButton = GetFieldValue<NMainMenuTextButton>(mainMenu, "_abandonRunButton");
        if (abandonButton == null || !abandonButton.IsVisibleInTree())
            return Error("Abandon button is not available on this game build");

        abandonButton.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            // Says which half happened. "Abandoning saved run" was the old text
            // and it claimed the whole thing.
            ["message"] = "Requested abandon; the confirmation popup is opening. "
                          + "Poll until menu.abandon_confirm_open is true, then send "
                          + "abandon_game again to confirm.",
            ["confirm_pending"] = true
        };
    }

    private static Dictionary<string, object?> ExecuteStartNewGame(Dictionary<string, JsonElement> data)
    {
        var root = ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root;
        var characterSelect = FindFirst<NCharacterSelectScreen>(root);

        if (characterSelect == null || !characterSelect.IsVisibleInTree())
        {
            var mainMenu = FindFirst<NMainMenu>(root);
            if (mainMenu == null || !mainMenu.IsVisibleInTree())
                return Error("Main menu is not open");

            var singleplayerButton = GetFieldValue<NMainMenuTextButton>(mainMenu, "_singleplayerButton");
            if (singleplayerButton == null || !singleplayerButton.IsVisibleInTree())
                return Error("Singleplayer button is not available");
            singleplayerButton.ForceClick();

            var submenu = mainMenu.OpenSingleplayerSubmenu();
            if (submenu == null || !submenu.IsVisibleInTree())
                return Error("Singleplayer submenu could not be opened");

            var standardButton = GetFieldValue<NSubmenuButton>(submenu, "_standardButton");
            if (standardButton == null || !standardButton.IsVisibleInTree())
                return Error("Standard new game button is not available");
            standardButton.ForceClick();

            characterSelect = FindFirst<NCharacterSelectScreen>(root);
            if (characterSelect == null || !characterSelect.IsVisibleInTree())
                return Error("Character select could not be opened");
        }

        string characterId = NormalizeCharacterId(
            data.TryGetValue("character", out var charElem) ? charElem.GetString() : null
        );

        int ascension = 0;
        if (data.TryGetValue("ascension", out var ascElem))
            ascension = ascElem.GetInt32();

        var button = FindAll<NCharacterSelectButton>(characterSelect)
            .FirstOrDefault(b => b.Character?.Id.Entry == characterId);
        if (button == null || button.Character == null)
            return Error($"Character '{characterId}' is not available on the character select screen");

        characterSelect.SelectCharacter(button, button.Character);

        var ascensionPanel = FindFirst<NAscensionPanel>(characterSelect);
        ascensionPanel?.SetAscensionLevel(ascension);

        var embarkButton = GetFieldValue<NConfirmButton>(characterSelect, "_embarkButton");
        if (embarkButton == null || !embarkButton.IsVisibleInTree() || !embarkButton.IsEnabled)
            return Error("Embark button is not available on this game build");
        embarkButton.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Starting new singleplayer run as {characterId} on Ascension {ascension}"
        };
    }

    /// <summary>Read a field by name, public or not, or null if it is not there.</summary>
    /// <remarks>
    /// Used where the public surface carries no identity but the object plainly
    /// knows who it is: <c>NCardRewardAlternativeButton</c> publishes nothing but
    /// a Godot Control and holds <c>_optionName</c> privately, which is the only
    /// way to tell a skip button from a sacrifice one.
    ///
    /// Naming a private field is a weaker contract than calling an API - a rename
    /// in any patch is silent here, where it would be a compile error elsewhere.
    /// So this returns null instead of throwing and callers report the field as
    /// unknown rather than substituting something plausible.
    /// </remarks>
    private static T? GetFieldValue<T>(object? target, string fieldName) where T : class
    {
        if (target == null) return null;
        try
        {
            var field = target.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            return field?.GetValue(target) as T;
        }
        catch { return null; }
    }

    /// <summary>
    /// Call a private instance method by name. Returns false if it is not there.
    /// </summary>
    /// <remarks>
    /// Added 2026-09-16 for <c>OnAlternateRewardSelected</c>. The lesson behind
    /// it: <c>ForceClick()</c> on a button is not the same as the handler that
    /// button is wired to. Clicking the card reward's 跳过 closed the screen and
    /// left the reward pending, three separate fixes in a row were built on the
    /// assumption that it had done more than that, and the screen's own named
    /// handler had been sitting in the probe output the whole time.
    ///
    /// Same failure mode as <c>GetFieldValue</c>: a rename is silent, so this
    /// returns false rather than throwing and the caller says which name it
    /// could not find.
    /// </remarks>
    private static bool TryInvoke(object? target, string methodName, params object[] args)
    {
        if (target == null) return false;
        try
        {
            var method = target.GetType().GetMethod(
                methodName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null,
                args.Select(a => a.GetType()).ToArray(),
                null);
            if (method == null) return false;
            method.Invoke(target, args);
            return true;
        }
        catch { return false; }
    }

    private static string NormalizeCharacterId(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "IRONCLAD";

        return value.Trim().ToUpperInvariant() switch
        {
            "IRONCLAD" or "铁甲战士" => "IRONCLAD",
            "SILENT" or "静默猎手" => "SILENT",
            "DEFECT" or "故障机器人" => "DEFECT",
            "NECROBINDER" or "亡灵契约师" => "NECROBINDER",
            "REGENT" or "储君" => "REGENT",
            var other => other
        };
    }

    private static Dictionary<string, object?> ExecuteReturnToMainMenu()
    {
        var root = ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root;
        var gameOverScreen = FindFirst<NGameOverScreen>(root);
        if (gameOverScreen != null && gameOverScreen.IsVisibleInTree())
        {
            var returnButton = FindFirst<NReturnToMainMenuButton>(gameOverScreen);
            if (returnButton != null && returnButton.IsVisibleInTree() && returnButton.IsEnabled)
            {
                returnButton.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = "Leaving game over screen and returning to main menu"
                };
            }

            var continueButton = FindFirst<NGameOverContinueButton>(gameOverScreen);
            if (continueButton != null && continueButton.IsVisibleInTree() && continueButton.IsEnabled)
            {
                continueButton.ForceClick();

                returnButton = FindFirst<NReturnToMainMenuButton>(gameOverScreen);
                if (returnButton != null && returnButton.IsVisibleInTree() && returnButton.IsEnabled)
                {
                    returnButton.ForceClick();
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = "Continuing past game over screen and returning to main menu"
                    };
                }

                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = "Continuing past game over screen"
                };
            }

            return Error("Game over screen is open but no usable main-menu button is available");
        }

        var game = FindFirst<NGame>(((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);
        if (game == null)
            return Error("Game node is not available");

        game.ReturnToMainMenu().GetAwaiter().GetResult();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Returning to main menu"
        };
    }

    private static Dictionary<string, object?> ExecutePlayCard(Player player, Dictionary<string, JsonElement> data)
    {
        if (!CombatManager.Instance.IsInProgress)
            return Error("Not in combat");
        // v0.107.1 removed the global CombatManager.IsPlayPhase flag; whether it
        // is your turn is now asked per player, since combat can be co-op.
        if (!CombatManager.Instance.IsPartOfPlayerTurn(player))
            return Error("Not in play phase — cannot act during enemy turn");
        if (CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled");
        if (!player.Creature.IsAlive)
            return Error("Player creature is dead — cannot play cards");

        var combatState = player.Creature.CombatState;
        if (combatState == null)
            return Error("No combat state");

        // Get card by index in hand
        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index'");

        int cardIndex = indexElem.GetInt32();
        var hand = player.PlayerCombatState?.Hand;
        if (hand == null)
            return Error("No hand available");

        if (cardIndex < 0 || cardIndex >= hand.Cards.Count)
            return Error($"card_index {cardIndex} out of range (hand has {hand.Cards.Count} cards)");

        var card = hand.Cards[cardIndex];

        if (!card.CanPlay(out var reason, out _))
            return Error($"Card '{card.Title}' cannot be played: {reason}");

        // Resolve target
        Creature? target = null;
        if (card.TargetType == TargetType.AnyEnemy)
        {
            if (!data.TryGetValue("target", out var targetElem))
                return Error("Card requires a target. Provide 'target' with an entity_id.");

            string targetId = targetElem.GetString() ?? "";
            target = ResolveTarget(combatState, targetId);
            if (target == null)
                return Error($"Target '{targetId}' not found among alive enemies");
        }

        // Play the card via the action queue (same path as the game UI)
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, target));

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Playing '{card.Title}'" + (target != null ? $" targeting {SafeGetText(() => target.Monster?.Title) ?? "target"}" : "")
        };
    }

    private static Dictionary<string, object?> ExecuteEndTurn(Player player)
    {
        if (!CombatManager.Instance.IsInProgress)
            return Error("Not in combat");
        if (!CombatManager.Instance.IsPartOfPlayerTurn(player))
            return Error("Not in play phase — cannot act during enemy turn");
        if (CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled (turn may already be ending)");

        // Match the game's own CanTurnBeEnded guard (NEndTurnButton.cs:114-123)
        var hand = NCombatRoom.Instance?.Ui?.Hand;
        if (hand != null && (hand.InCardPlay || hand.CurrentMode != NPlayerHand.Mode.Play))
            return Error("Cannot end turn while a card is being played or hand is in selection mode");

        PlayerCmd.EndTurn(player, canBackOut: false);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Ending turn"
        };
    }

    private static Dictionary<string, object?> ExecuteUsePotion(Player player, Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("slot", out var slotElem))
            return Error("Missing 'slot' (potion slot index)");

        int slot = slotElem.GetInt32();
        if (slot < 0 || slot >= player.PotionSlots.Count)
            return Error($"Potion slot {slot} out of range (player has {player.PotionSlots.Count} slots)");

        var potion = player.GetPotionAtSlotIndex(slot);
        if (potion == null)
            return Error($"No potion in slot {slot}");
        if (potion.IsQueued)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' is already queued for use");
        if (potion.Owner.Creature.IsDead)
            return Error("Cannot use potion — player creature is dead");
        if (!potion.PassesCustomUsabilityCheck)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' cannot be used right now");

        bool inCombat = CombatManager.Instance.IsInProgress;
        if (potion.Usage == PotionUsage.CombatOnly)
        {
            if (!inCombat)
                return Error($"Potion '{SafeGetText(() => potion.Title)}' can only be used in combat");
            if (!CombatManager.Instance.IsPartOfPlayerTurn(player))
                return Error("Cannot use potions outside of play phase");
        }
        else if (potion.Usage == PotionUsage.Automatic)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' is automatic and cannot be manually used");

        if (inCombat && CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled");

        // Resolve target
        Creature? target = null;
        var combatState = player.Creature.CombatState;

        switch (potion.TargetType)
        {
            case TargetType.AnyEnemy:
                if (!data.TryGetValue("target", out var targetElem))
                    return Error("Potion requires a target enemy. Provide 'target' with an entity_id.");
                string targetId = targetElem.GetString() ?? "";
                if (combatState == null)
                    return Error("No combat state for target resolution");
                target = ResolveTarget(combatState, targetId);
                if (target == null)
                    return Error($"Target '{targetId}' not found among alive enemies");
                break;
            case TargetType.Self:
            case TargetType.AnyAlly:
            case TargetType.AnyPlayer:
                target = player.Creature;
                break;
            default:
                target = null;
                break;
        }

        potion.EnqueueManualUse(target);

        string targetMsg = potion.TargetType switch
        {
            TargetType.AnyEnemy => $" targeting {SafeGetText(() => target?.Monster?.Title) ?? "enemy"}",
            TargetType.Self or TargetType.AnyPlayer or TargetType.AnyAlly => " on self",
            _ => ""
        };

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Using potion '{SafeGetText(() => potion.Title)}' from slot {slot}{targetMsg}"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseEventOption(Dictionary<string, JsonElement> data)
    {
        var uiRoom = NEventRoom.Instance;
        if (uiRoom == null)
            return Error("Event room is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (event option index)");

        int index = indexElem.GetInt32();

        var buttons = FindAll<NEventOptionButton>(uiRoom)
            .Where(b => !b.Option.IsLocked)
            .ToList();

        if (buttons.Count == 0)
            return Error("No unlocked event options available");
        if (index < 0 || index >= buttons.Count)
            return Error($"Event option index {index} out of range ({buttons.Count} unlocked options)");

        var button = buttons[index];
        string title = SafeGetText(() => button.Option.Title) ?? "option";
        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Choosing event option: {title}"
        };
    }

    private static Dictionary<string, object?> ExecuteAdvanceDialogue()
    {
        var uiRoom = NEventRoom.Instance;
        if (uiRoom == null)
            return Error("Event room is not open");

        var ancientLayout = FindFirst<NAncientEventLayout>(uiRoom);
        if (ancientLayout == null)
            return Error("No ancient dialogue active");

        var hitbox = ancientLayout.GetNodeOrNull<NClickableControl>("%DialogueHitbox");
        if (hitbox == null || !hitbox.Visible || !hitbox.IsEnabled)
            return Error("Dialogue hitbox not available — dialogue may have ended");

        hitbox.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Advancing dialogue"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseRestOption(Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (rest site option index)");

        int index = indexElem.GetInt32();

        var restRoom = NRestSiteRoom.Instance;
        if (restRoom == null)
            return Error("Rest site room is not open");

        var buttons = FindAll<NRestSiteButton>(restRoom)
            .Where(b => b.Option.IsEnabled)
            .ToList();

        if (index < 0 || index >= buttons.Count)
            return Error($"Rest option index {index} out of range ({buttons.Count} enabled options)");

        var button = buttons[index];
        string optionName = SafeGetText(() => button.Option.Title) ?? button.Option.OptionId;
        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting rest site option: {optionName}"
        };
    }

    private static Dictionary<string, object?> ExecuteShopPurchase(Player player, Dictionary<string, JsonElement> data)
    {
        if (player.RunState.CurrentRoom is not MerchantRoom merchantRoom)
            return Error("Not in a shop");

        // Auto-open inventory if needed
        var merchUI = NMerchantRoom.Instance;
        if (merchUI != null && !merchUI.Inventory.IsOpen)
            merchUI.OpenInventory();

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (shop item index)");

        int index = indexElem.GetInt32();

        var allEntries = merchantRoom.GetLocalInventory().AllEntries.ToList();
        if (index < 0 || index >= allEntries.Count)
            return Error($"Shop item index {index} out of range ({allEntries.Count} items)");

        var entry = allEntries[index];
        if (!entry.IsStocked)
            return Error("Item is sold out");
        if (!entry.EnoughGold)
            return Error($"Not enough gold (need {entry.Cost}, have {player.Gold})");

        // Fire-and-forget purchase (same path as AutoSlay)
        _ = entry.OnTryPurchaseWrapper(merchantRoom.GetLocalInventory());

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Purchasing item for {entry.Cost} gold"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseMapNode(Dictionary<string, JsonElement> data)
    {
        var mapScreen = NMapScreen.Instance;
        if (mapScreen == null || !mapScreen.IsOpen)
            return Error("Map screen is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (map node index from next_options)");

        int index = indexElem.GetInt32();

        var travelable = FindAll<NMapPoint>(mapScreen)
            .Where(mp => mp.State == MapPointState.Travelable)
            .OrderBy(mp => mp.Point.coord.col)
            .ToList();

        if (travelable.Count == 0)
            return Error("No travelable map nodes available");
        if (index < 0 || index >= travelable.Count)
            return Error($"Map node index {index} out of range ({travelable.Count} options available)");

        var target = travelable[index];
        mapScreen.OnMapPointSelectedLocally(target);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Traveling to {target.Point.PointType} at ({target.Point.coord.col},{target.Point.coord.row})"
        };
    }

    private static Dictionary<string, object?> ExecuteClaimReward(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NRewardsScreen rewardsScreen)
            return Error("Rewards screen is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (reward index)");

        int index = indexElem.GetInt32();

        var enabledButtons = FindAll<NRewardButton>(rewardsScreen)
            .Where(b => b.IsEnabled && b.Reward != null)
            .ToList();

        if (index < 0 || index >= enabledButtons.Count)
            return Error($"Reward index {index} out of range (screen has {enabledButtons.Count} claimable rewards)");

        var button = enabledButtons[index];
        var reward = button.Reward!;
        string rewardDesc = GetRewardTypeName(reward);
        if (reward is GoldReward g)
            rewardDesc = $"gold ({g.Amount})";
        else if (reward is PotionReward p)
            rewardDesc = $"potion ({SafeGetText(() => p.Potion?.Title)})";
        else if (reward is CardReward)
            rewardDesc = "card (opens card selection)";

        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Claiming reward: {rewardDesc}"
        };
    }

    private static Dictionary<string, object?> ExecuteSelectCardReward(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCardRewardSelectionScreen cardScreen)
            return Error("Card reward selection screen is not open");

        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index'");

        int cardIndex = indexElem.GetInt32();

        var cardHolders = FindAllSortedByPosition<NCardHolder>(cardScreen);
        if (cardIndex < 0 || cardIndex >= cardHolders.Count)
            return Error($"Card index {cardIndex} out of range (screen has {cardHolders.Count} cards)");

        var holder = cardHolders[cardIndex];
        string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting card: {cardName}"
        };
    }

    /// <summary>Throw away a carried potion, freeing its slot.</summary>
    /// <remarks>
    /// The bridge had no way to free a slot at all, so an agent with three full
    /// slots simply forfeited every later potion reward - on 2026-09-13 that was
    /// written down as "the agent cannot free a slot; a human obviously can, the
    /// bridge just has no verb for it". It does now: the game models this as
    /// <c>Player.CanRemovePotions</c> plus <c>NPotionHolder.DiscardPotion()</c>.
    ///
    /// The holder is matched by <c>PotionModel</c> reference, not by position.
    /// <c>FindAll</c> returns scene-tree order and <c>PotionSlots</c> is the
    /// model's order, and nothing proves the two agree - assuming they do is the
    /// whole card_index family of bugs. The model object is the same reference
    /// on both sides, so the question never comes up.
    /// </remarks>
    private static Dictionary<string, object?> ExecuteDiscardPotion(
        Player player, Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("slot", out var slotElem))
            return Error("Missing 'slot' (potion slot index)");

        int slot = slotElem.GetInt32();
        if (slot < 0 || slot >= player.PotionSlots.Count)
            return Error($"Potion slot {slot} out of range (player has {player.PotionSlots.Count} slots)");

        var potion = player.GetPotionAtSlotIndex(slot);
        if (potion == null)
            return Error($"No potion in slot {slot}");

        // The game's own gate. Checked rather than assumed: "the slot has a
        // potion" and "the game will let go of it right now" are different
        // claims, and the second one is the one that decides.
        if (!player.CanRemovePotions)
            return Error("Potions cannot be removed right now (CanRemovePotions is false)");

        string name = SafeGetText(() => potion.Title) ?? "unknown";

        var root = ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root;
        var holder = FindAll<NPotionHolder>(root)
            .FirstOrDefault(h => h.HasPotion && ReferenceEquals(h.Potion?.Model, potion));
        if (holder == null)
            return Error($"No on-screen potion holder is showing '{name}' (slot {slot})");

        holder.DiscardPotion();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Discarding potion '{name}' from slot {slot}"
        };
    }

    private static Dictionary<string, object?> ExecuteSkipCardReward(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCardRewardSelectionScreen cardScreen)
            return Error("Card reward selection screen is not open");

        // Same enumeration BuildCardRewardState builds `alternatives` from, so
        // index N here is the entry the agent was shown as N.
        var altButtons = FindAll<NCardRewardAlternativeButton>(cardScreen);
        if (altButtons.Count == 0)
            return Error("No alternative option available on this card reward");

        // Defaults to 0 so existing callers are unchanged, but 0 is no longer
        // assumed to be skip - it used to be hardcoded, and with Pael's Wing on
        // the run that blind click could sacrifice the reward instead.
        int index = 0;
        if (data.TryGetValue("index", out var indexElem))
            index = indexElem.GetInt32();

        if (index < 0 || index >= altButtons.Count)
            return Error(
                $"Alternative index {index} out of range (screen has {altButtons.Count}). "
                + "Read `alternatives` on the card_reward state: a relic can add options "
                + "and index 0 is not guaranteed to be skip.");

        var chosen = DescribeAlternative(index, altButtons[index], ReadExtraOptions(cardScreen));
        altButtons[index].ForceClick();

        // Names what was clicked. The old message said "Skipping card reward"
        // unconditionally, which made a sacrifice and a skip identical in the
        // log - and the log is the only record of what the agent actually did.
        //
        // ⚠️ Clicking this does **not** finish the reward. The game declares the
        // built-in skip `EndSelectionAndDoNotCompleteReward`, measured live on
        // 2026-09-16, so the screen closes and the reward stays outstanding.
        // Declining a card reward has no working path yet; see the notes.
        var what = chosen["title"] ?? chosen["option_id"] ?? "unknown (private field renamed?)";
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Card reward alternative {index}: {what} "
                          + $"(after_selected={chosen["after_selected"] ?? "unknown"})"
        };
    }

    private static Dictionary<string, object?> ExecuteProceed()
    {
        // Try rewards overlay. The button is not inside it - it belongs to the
        // room underneath - so search the scene, not the screen. The old
        // FindFirst<NProceedButton>(rewardsScreen) always returned null, which
        // made this whole branch dead and left a claimed rewards screen with no
        // way out (2026-09-13).
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is NRewardsScreen)
        {
            // ⚠️ Reverted to exactly this on 2026-09-16, after an evening of
            // changes that broke it. Two of them, and both looked like
            // improvements: preferring the screen's own `_proceedButton` over the
            // tree search, and pressing/releasing instead of ForceClick. With
            // both in, a screen reporting `0 still listed, is_complete True` -
            // the ordinary finished state - stopped advancing at all.
            //
            // This version had been carrying full runs since 2026-09-13. The
            // wedge that started the evening was a *card reward that cannot be
            // declined*, which is a different problem and is still open; nothing
            // here was ever the cause of it. Do not "fix" this again without a
            // run that reproduces a failure of this call specifically.
            var btn = FindLiveProceedButton();
            if (btn != null)
            {
                btn.ForceClick();
                return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from rewards" };
            }
        }

        // Try rest site
        if (NRestSiteRoom.Instance is { } restRoom && restRoom.ProceedButton.IsEnabled)
        {
            restRoom.ProceedButton.ForceClick();
            return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from rest site" };
        }

        // Try merchant — close inventory first if open, then proceed
        if (NMerchantRoom.Instance is { } merchRoom)
        {
            if (merchRoom.Inventory.IsOpen)
            {
                var backBtn = FindFirst<NBackButton>(merchRoom);
                if (backBtn is { IsEnabled: true })
                    backBtn.ForceClick();
            }
            if (merchRoom.ProceedButton.IsEnabled)
            {
                merchRoom.ProceedButton.ForceClick();
                return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from shop" };
            }
        }

        // Try treasure room
        var treasureUI = FindFirst<NTreasureRoom>(
            ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);
        if (treasureUI != null && treasureUI.ProceedButton.IsEnabled)
        {
            treasureUI.ProceedButton.ForceClick();
            return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from treasure room" };
        }

        return Error("No proceed button available or enabled");
    }

    private static Dictionary<string, object?> ExecuteSelectCard(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (card index in the grid)");

        int index = indexElem.GetInt32();

        if (overlay is NCardGridSelectionScreen gridScreen)
        {
            var grid = FindFirst<NCardGrid>(gridScreen);
            if (grid == null)
                return Error("Card grid not found in selection screen");

            var holders = FindAllSortedByPosition<NGridCardHolder>(gridScreen);
            if (index < 0 || index >= holders.Count)
                return Error($"Card index {index} out of range ({holders.Count} cards available)");

            var holder = holders[index];
            string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
            grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holder);

            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Toggling card selection: {cardName}"
            };
        }
        else if (overlay is NChooseACardSelectionScreen chooseScreen)
        {
            var holders = FindAllSortedByPosition<NGridCardHolder>(chooseScreen);
            if (index < 0 || index >= holders.Count)
                return Error($"Card index {index} out of range ({holders.Count} cards available)");

            var holder = holders[index];
            string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
            holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Choosing card: {cardName}"
            };
        }

        else if (overlay is NChooseABundleSelectionScreen bundleScreen)
        {
            // Same finder and the same ordering as BuildBundleState, so the index
            // the state reported is the index this clicks. Mirroring the filter
            // on both sides is the only one of these pairs that has never gone
            // wrong - the ones that derive the order independently are still
            // carrying an unverified assumption.
            var bundles = FindAllSortedByPosition<NCardBundle>(bundleScreen);
            if (index < 0 || index >= bundles.Count)
                return Error($"Bundle index {index} out of range ({bundles.Count} bundles)");

            var chosen = bundles[index];
            if (chosen.Hitbox is not { } hitbox)
                return Error("Bundle has no hitbox to click");
            hitbox.ForceClick();

            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Choosing bundle {index} of {bundles.Count}"
            };
        }

        return Error("No card selection screen is open");
    }

    private static Dictionary<string, object?> ExecuteConfirmSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is NChooseACardSelectionScreen)
            return Error("Choose-a-card screen requires no confirmation — use select_card(index) to pick directly");
        if (overlay is not NCardGridSelectionScreen screen)
            return Error("No card selection screen is open");

        // Check all preview containers (upgrade uses UpgradeSinglePreviewContainer / UpgradeMultiPreviewContainer,
        // NDeckCardSelectScreen uses PreviewContainer with %PreviewConfirm)
        foreach (var containerName in new[] { "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer", "%PreviewContainer" })
        {
            var container = screen.GetNodeOrNull<Godot.Control>(containerName);
            if (container?.Visible == true)
            {
                var confirm = container.GetNodeOrNull<NConfirmButton>("Confirm")
                              ?? container.GetNodeOrNull<NConfirmButton>("%PreviewConfirm");
                if (confirm is { IsEnabled: true })
                {
                    confirm.ForceClick();
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = "Confirming selection from preview"
                    };
                }
            }
        }

        // Try main confirm button
        var mainConfirm = screen.GetNodeOrNull<NConfirmButton>("Confirm")
                          ?? screen.GetNodeOrNull<NConfirmButton>("%Confirm");
        if (mainConfirm is { IsEnabled: true })
        {
            mainConfirm.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Confirming selection"
            };
        }

        // Fallback: find ANY enabled NConfirmButton in the screen tree.
        // Covers NCardGridSelectionScreen subclasses (like NDeckEnchantSelectScreen)
        // whose confirm button isn't in any of the known container paths above.
        var allConfirmButtons = FindAll<NConfirmButton>(screen);
        foreach (var btn in allConfirmButtons)
        {
            if (btn.IsEnabled && btn.IsVisibleInTree())
            {
                btn.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = "Confirming selection"
                };
            }
        }

        return Error("No confirm button is currently enabled — select more cards first");
    }

    private static Dictionary<string, object?> ExecuteCancelSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();

        // Handle choose-a-card screen (skip button)
        if (overlay is NChooseACardSelectionScreen chooseScreen)
        {
            var skipButton = chooseScreen.GetNodeOrNull<NClickableControl>("SkipButton");
            if (skipButton is { IsEnabled: true })
            {
                skipButton.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = "Skipping card choice"
                };
            }
            return Error("No skip option available — a card must be chosen");
        }

        if (overlay is not NCardGridSelectionScreen screen)
            return Error("No card selection screen is open");

        // If preview is showing, cancel back to selection
        foreach (var containerName in new[] { "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer", "%PreviewContainer" })
        {
            var container = screen.GetNodeOrNull<Godot.Control>(containerName);
            if (container?.Visible == true)
            {
                var cancelBtn = container.GetNodeOrNull<NBackButton>("Cancel")
                                ?? container.GetNodeOrNull<NBackButton>("%PreviewCancel");
                if (cancelBtn is { IsEnabled: true })
                {
                    cancelBtn.ForceClick();
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = "Cancelling preview — returning to card selection"
                    };
                }
            }
        }

        // Close the screen entirely
        var closeButton = screen.GetNodeOrNull<NBackButton>("%Close");
        if (closeButton is { IsEnabled: true })
        {
            closeButton.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Closing card selection screen"
            };
        }

        return Error("No cancel/close button is currently enabled — selection may be mandatory");
    }

    private static Dictionary<string, object?> ExecuteCombatSelectCard(Dictionary<string, JsonElement> data)
    {
        var hand = NPlayerHand.Instance;
        if (hand == null || !hand.IsInCardSelection)
            return Error("No in-combat card selection is active");

        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index' (index of the card in hand)");

        int index = indexElem.GetInt32();
        var holders = hand.ActiveHolders;
        if (index < 0 || index >= holders.Count)
            return Error($"Card index {index} out of range ({holders.Count} selectable cards)");

        var holder = holders[index];
        string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";

        // Emit the Pressed signal — same path the game UI uses
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting card from hand: {cardName}"
        };
    }

    private static Dictionary<string, object?> ExecuteCombatConfirmSelection()
    {
        var hand = NPlayerHand.Instance;
        if (hand == null || !hand.IsInCardSelection)
            return Error("No in-combat card selection is active");

        var confirmBtn = hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton");
        if (confirmBtn == null || !confirmBtn.IsEnabled)
            return Error("Confirm button is not enabled — select more cards first");

        confirmBtn.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Confirming combat card selection"
        };
    }

    private static Dictionary<string, object?> ExecuteSelectRelic(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseARelicSelection screen)
            return Error("No relic selection screen is open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (relic index)");

        int index = indexElem.GetInt32();

        var holders = FindAll<NRelicBasicHolder>(screen);
        if (index < 0 || index >= holders.Count)
            return Error($"Relic index {index} out of range ({holders.Count} relics available)");

        var holder = holders[index];
        string relicName = SafeGetText(() => holder.Relic?.Model?.Title) ?? "unknown";
        holder.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting relic: {relicName}"
        };
    }

    private static Dictionary<string, object?> ExecuteSkipRelicSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseARelicSelection screen)
            return Error("No relic selection screen is open");

        var skipButton = screen.GetNodeOrNull<NClickableControl>("SkipButton");
        if (skipButton is not { IsEnabled: true })
            return Error("No skip option available");

        skipButton.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Skipping relic selection"
        };
    }

    private static Dictionary<string, object?> ExecuteClaimTreasureRelic(Dictionary<string, JsonElement> data)
    {
        var treasureUI = FindFirst<NTreasureRoom>(
            ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);
        if (treasureUI == null)
            return Error("Treasure room is not open");

        var relicCollection = treasureUI.GetNodeOrNull<NTreasureRoomRelicCollection>("%RelicCollection");
        if (relicCollection?.Visible != true)
            return Error("Relic collection is not visible — chest may not be opened yet");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (relic index)");

        int index = indexElem.GetInt32();

        var holders = FindAll<NTreasureRoomRelicHolder>(relicCollection)
            .Where(h => h.IsEnabled && h.Visible)
            .ToList();

        if (index < 0 || index >= holders.Count)
            return Error($"Relic index {index} out of range ({holders.Count} relics available)");

        var holder = holders[index];
        string relicName = SafeGetText(() => holder.Relic?.Model?.Title) ?? "unknown";
        holder.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Claiming treasure relic: {relicName}"
        };
    }

    // v0.107.1: Creature.CombatState is typed as the ICombatState interface.
    private static Creature? ResolveTarget(ICombatState combatState, string entityId)
    {
        // Try to match by entity_id pattern: "model_entry_N"
        // First try matching by combat_id if it's a pure number
        if (uint.TryParse(entityId, out uint combatId))
            return combatState.GetCreature(combatId);

        // Match by entity_id pattern (e.g., "jaw_worm_0")
        // We rebuild the entity IDs the same way as BuildEnemyState
        var entityCounts = new Dictionary<string, int>();
        foreach (var creature in combatState.Enemies)
        {
            if (!creature.IsAlive) continue;
            string baseId = creature.Monster?.Id.Entry ?? "unknown";
            if (!entityCounts.TryGetValue(baseId, out int count))
                count = 0;
            entityCounts[baseId] = count + 1;
            string generatedId = $"{baseId}_{count}";

            if (generatedId == entityId)
                return creature;
        }

        return null;
    }
}
