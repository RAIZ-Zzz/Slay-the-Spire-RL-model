using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;   // PileType, for the card-library dump
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Multiplayer.Game;

namespace STS2_Bridge;

[ModInitializer("Initialize")]
public static partial class BridgeMod
{
    public const string Version = "0.3.0";

    private static HttpListener? _listener;
    private static Thread? _serverThread;
    private static readonly ConcurrentQueue<Action> _mainThreadQueue = new();
    internal static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public static void Initialize()
    {
        try
        {
            // Connect to main thread process frame for action execution
            var tree = (SceneTree)Engine.GetMainLoop();
            tree.Connect(SceneTree.SignalName.ProcessFrame, Callable.From(ProcessMainThreadQueue));

            _listener = new HttpListener();
            _listener.Prefixes.Add("http://localhost:15526/");
            _listener.Prefixes.Add("http://127.0.0.1:15526/");
            _listener.Start();

            _serverThread = new Thread(ServerLoop)
            {
                IsBackground = true,
                Name = "STS2_Bridge_Server"
            };
            _serverThread.Start();

            GD.Print($"[STS2 Bridge] v{Version} server started on http://localhost:15526/");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[STS2 Bridge] Failed to start: {ex}");
        }
    }

    private static void ProcessMainThreadQueue()
    {
        int processed = 0;
        while (_mainThreadQueue.TryDequeue(out var action) && processed < 10)
        {
            try { action(); }
            catch (Exception ex) { GD.PrintErr($"[STS2 Bridge] Main thread action error: {ex}"); }
            processed++;
        }
    }

    internal static Task<T> RunOnMainThread<T>(Func<T> func)
    {
        var tcs = new TaskCompletionSource<T>();
        _mainThreadQueue.Enqueue(() =>
        {
            try { tcs.SetResult(func()); }
            catch (Exception ex) { tcs.SetException(ex); }
        });
        return tcs.Task;
    }

    internal static Task RunOnMainThread(Action action)
    {
        var tcs = new TaskCompletionSource<bool>();
        _mainThreadQueue.Enqueue(() =>
        {
            try { action(); tcs.SetResult(true); }
            catch (Exception ex) { tcs.SetException(ex); }
        });
        return tcs.Task;
    }

    private static void ServerLoop()
    {
        while (_listener?.IsListening == true)
        {
            try
            {
                var context = _listener.GetContext();
                // Handle each request asynchronously so we don't block the listener
                ThreadPool.QueueUserWorkItem(_ => HandleRequest(context));
            }
            catch (HttpListenerException) { break; }
            catch (ObjectDisposedException) { break; }
        }
    }

    private static void HandleRequest(HttpListenerContext context)
    {
        try
        {
            var request = context.Request;
            var response = context.Response;
            response.Headers.Add("Access-Control-Allow-Origin", "*");
            response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.Headers.Add("Access-Control-Allow-Headers", "Content-Type");

            if (request.HttpMethod == "OPTIONS")
            {
                response.StatusCode = 204;
                response.Close();
                return;
            }

            string path = request.Url?.AbsolutePath ?? "/";

            if (path == "/")
            {
                SendJson(response, new { message = $"Hello from STS2 Bridge v{Version}", status = "ok" });
            }
            else if (path == "/api/v1/singleplayer")
            {
                // Hard-block singleplayer endpoint during multiplayer runs
                // to prevent calling the non-sync-safe end_turn path
                if (IsMultiplayerRun())
                {
                    SendError(response, 409,
                        "Multiplayer run is active. Use /api/v1/multiplayer instead.");
                    return;
                }

                if (request.HttpMethod == "GET")
                    HandleGetState(request, response);
                else if (request.HttpMethod == "POST")
                    HandlePostAction(request, response);
                else
                    SendError(response, 405, "Method not allowed");
            }
            else if (path == "/api/v1/cards")
            {
                // Every card in the game, dumped once and cached to disk by the
                // caller. A policy that has to judge "is this card good" needs the
                // numbers, and they are not on CardModel as properties - the game
                // computes damage and block through hooks at play time - so the
                // description text is the only place they appear. Dumping the
                // whole library once means that text gets parsed once, offline,
                // where the result can be checked, instead of being regexed on the
                // hot path of every decision.
                //
                // Not part of /state on purpose: this is static per game version,
                // several hundred entries, and has no business riding on a
                // response polled three times a second.
                if (request.HttpMethod == "GET")
                    HandleGetCards(response);
                else
                    SendError(response, 405, "Method not allowed");
            }
            else if (path == "/api/v1/multiplayer")
            {
                // Guard: reject multiplayer endpoint during singleplayer runs
                if (!IsMultiplayerRun())
                {
                    SendError(response, 409,
                        "Not in a multiplayer run. Use /api/v1/singleplayer instead.");
                    return;
                }

                if (request.HttpMethod == "GET")
                    HandleGetMultiplayerState(request, response);
                else if (request.HttpMethod == "POST")
                    HandlePostMultiplayerAction(request, response);
                else
                    SendError(response, 405, "Method not allowed");
            }
            else
            {
                SendError(response, 404, "Not found");
            }
        }
        catch (Exception ex)
        {
            try
            {
                SendError(context.Response, 500, $"Internal error: {ex.Message}");
            }
            catch { /* response may already be closed */ }
        }
    }

    // Called on HTTP thread (not main thread) as a best-effort guard.
    // The try/catch handles race conditions during run transitions.
    // Authoritative checks happen inside RunOnMainThread lambdas.
    internal static bool IsMultiplayerRun()
    {
        try
        {
            return MegaCrit.Sts2.Core.Runs.RunManager.Instance.IsInProgress
                && MegaCrit.Sts2.Core.Runs.RunManager.Instance.NetService.Type.IsMultiplayer();
        }
        catch { return false; }
    }

    private static void HandleGetMultiplayerState(HttpListenerRequest request, HttpListenerResponse response)
    {
        string format = request.QueryString["format"] ?? "json";

        try
        {
            var stateTask = RunOnMainThread(() => BuildMultiplayerGameState());
            var state = stateTask.GetAwaiter().GetResult();

            if (format == "markdown")
            {
                string md = FormatAsMarkdown(state);
                SendText(response, md, "text/markdown");
            }
            else
            {
                SendJson(response, state);
            }
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Failed to read multiplayer game state: {ex.Message}");
        }
    }

    private static void HandlePostMultiplayerAction(HttpListenerRequest request, HttpListenerResponse response)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
            body = reader.ReadToEnd();

        Dictionary<string, JsonElement>? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(body);
        }
        catch
        {
            SendError(response, 400, "Invalid JSON");
            return;
        }

        if (parsed == null || !parsed.TryGetValue("action", out var actionElem))
        {
            SendError(response, 400, "Missing 'action' field");
            return;
        }

        string action = actionElem.GetString() ?? "";

        try
        {
            var resultTask = RunOnMainThread(() => ExecuteMultiplayerAction(action, parsed));
            var result = resultTask.GetAwaiter().GetResult();
            SendJson(response, result);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Multiplayer action failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Dump every card the game knows about, from its own registry.
    /// </summary>
    /// <remarks>
    /// `ModelDb.AllCards` is the game's static card table, so this is the version
    /// the running build actually uses - not a table transcribed by hand, which
    /// would be wrong the first time somebody mistyped a number and wrong again
    /// at the next patch. The learning route says the same thing about Stage 3's
    /// simulator: take the values from the game data, do not copy them out.
    ///
    /// `description` is included because it is the only place the numbers appear
    /// at all. Every structured field the model does expose is here beside it, so
    /// a later parse of that text can be checked against `type`, `cost` and
    /// `gains_block` rather than trusted on its own.
    /// </remarks>
    private static void HandleGetCards(HttpListenerResponse response)
    {
        try
        {
            var cards = new List<Dictionary<string, object?>>();
            foreach (var card in MegaCrit.Sts2.Core.Models.ModelDb.AllCards)
            {
                if (card == null) continue;
                cards.Add(new Dictionary<string, object?>
                {
                    ["id"] = card.Id.Entry,
                    ["name"] = SafeGetText(() => card.Title),
                    ["type"] = card.Type.ToString(),
                    ["rarity"] = card.Rarity.ToString(),
                    ["cost"] = card.EnergyCost.CostsX
                        ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                    ["target_type"] = card.TargetType.ToString(),
                    ["gains_block"] = SafeBool(() => card.GainsBlock),
                    ["is_upgraded"] = card.IsUpgraded,
                    ["description"] = SafeGetCardDescription(card, PileType.None),
                    ["keywords"] = BuildHoverTips(card.HoverTips),
                });
            }

            var relics = new List<Dictionary<string, object?>>();
            foreach (var relic in MegaCrit.Sts2.Core.Models.ModelDb.AllRelics)
            {
                if (relic == null) continue;
                relics.Add(new Dictionary<string, object?>
                {
                    ["id"] = relic.Id.Entry,
                    ["name"] = SafeGetText(() => relic.Title),
                    ["rarity"] = relic.Rarity.ToString(),
                    ["description"] = SafeGetText(() => relic.DynamicDescription),
                    ["flavor"] = SafeGetText(() => relic.Flavor),
                    ["is_stackable"] = SafeBool(() => relic.IsStackable),
                    ["in_shops"] = SafeBool(() => relic.IsAllowedInShops),
                    ["has_pickup_effect"] = SafeBool(() => relic.HasUponPickupEffect),
                    ["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic),
                });
            }

            var potions = new List<Dictionary<string, object?>>();
            foreach (var potion in MegaCrit.Sts2.Core.Models.ModelDb.AllPotions)
            {
                if (potion == null) continue;
                potions.Add(new Dictionary<string, object?>
                {
                    ["id"] = potion.Id.Entry,
                    ["name"] = SafeGetText(() => potion.Title),
                    ["rarity"] = potion.Rarity.ToString(),
                    ["usage"] = potion.Usage.ToString(),
                    ["target_type"] = potion.TargetType.ToString(),
                    ["description"] = SafeGetText(() => potion.DynamicDescription),
                });
            }

            // Events are the case the user called "guessing": the policy had only
            // the localised option text at decision time, so it matched on 失去 /
            // 获得 and hoped. `GameInfoOptions` is the static option list the
            // game's own compendium shows, so an event can be recognised by `id`
            // and its options known before the screen appears.
            //
            // The option element type is not named anywhere reachable, so each is
            // pushed through SafeGetText - which unwraps a LocString and otherwise
            // falls back to ToString(). Guessing a type here would be the same
            // mistake as guessing STRIKE_RED was the id of Strike.
            var events = new List<Dictionary<string, object?>>();
            foreach (var ev in MegaCrit.Sts2.Core.Models.ModelDb.AllEvents)
            {
                if (ev == null) continue;
                var options = new List<string?>();
                try
                {
                    foreach (var opt in ev.GameInfoOptions)
                        options.Add(SafeGetText(() => opt));
                }
                catch { /* an event that will not enumerate keeps its other fields */ }

                events.Add(new Dictionary<string, object?>
                {
                    ["id"] = ev.Id.Entry,
                    ["name"] = SafeGetText(() => ev.Title),
                    ["description"] = SafeGetText(() => ev.InitialDescription)
                                      ?? SafeGetText(() => ev.Description),
                    ["is_shared"] = SafeBool(() => ev.IsShared),
                    ["is_deterministic"] = SafeBool(() => ev.IsDeterministic),
                    ["options"] = options,
                });
            }

            SendJson(response, new Dictionary<string, object?>
            {
                ["bridge_version"] = Version,
                ["counts"] = new Dictionary<string, object?>
                {
                    ["cards"] = cards.Count,
                    ["relics"] = relics.Count,
                    ["potions"] = potions.Count,
                    ["events"] = events.Count,
                },
                ["cards"] = cards,
                ["relics"] = relics,
                ["potions"] = potions,
                ["events"] = events,
            });
        }
        catch (Exception e)
        {
            SendError(response, 500, $"Failed to dump the library: {e.Message}");
        }
    }

    private static void HandleGetState(HttpListenerRequest request, HttpListenerResponse response)
    {
        string format = request.QueryString["format"] ?? "json";

        try
        {
            var stateTask = RunOnMainThread(() => BuildGameState());
            var state = stateTask.GetAwaiter().GetResult();

            if (format == "markdown")
            {
                string md = FormatAsMarkdown(state);
                SendText(response, md, "text/markdown");
            }
            else
            {
                SendJson(response, state);
            }
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Failed to read game state: {ex.Message}");
        }
    }

    private static void HandlePostAction(HttpListenerRequest request, HttpListenerResponse response)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
            body = reader.ReadToEnd();

        Dictionary<string, JsonElement>? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(body);
        }
        catch
        {
            SendError(response, 400, "Invalid JSON");
            return;
        }

        if (parsed == null || !parsed.TryGetValue("action", out var actionElem))
        {
            SendError(response, 400, "Missing 'action' field");
            return;
        }

        string action = actionElem.GetString() ?? "";

        try
        {
            var resultTask = RunOnMainThread(() => ExecuteAction(action, parsed));
            var result = resultTask.GetAwaiter().GetResult();
            SendJson(response, result);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Action failed: {ex.Message}");
        }
    }
}
