using System.Reflection;

// Metadata-only dump of the shipped game assembly: prints the members of the
// types the bridge failed to compile against, so the fix is based on the real
// v0.107.1 API rather than a guess.

string dataDir = @"A:\SteamLibrary\steamapps\common\Slay the Spire 2\data_sts2_windows_x86_64";

var assemblies = Directory.GetFiles(dataDir, "*.dll").ToList();
var resolver = new PathAssemblyResolver(assemblies);
using var mlc = new MetadataLoadContext(resolver, "System.Private.CoreLib");
var sts2 = mlc.LoadFromAssemblyPath(Path.Combine(dataDir, "sts2.dll"));

// Set by the `all:` prefix. The bridge is a mod loaded into the game's own
// process, so a private field is reachable with reflection exactly like a public
// one - "the public surface has no identity on it" therefore does not mean "the
// object does not know who it is". That distinction decided the card-reward
// alternatives fix: NCardRewardAlternativeButton exposes nothing publicly, and
// the private side is where the answer had to be looked for.
bool includePrivate = false;

void Dump(string typeName, string filter = "")
{
    var t = sts2.GetTypes().FirstOrDefault(x => x.FullName == typeName || x.Name == typeName);
    if (t is null) { Console.WriteLine($"### {typeName}: NOT FOUND\n"); return; }
    Console.WriteLine($"### {t.FullName}   (enum={t.IsEnum}, interface={t.IsInterface})");
    // The inheritance chain, because the bridge dispatches with `is` - so
    // "does the branch for X already catch Y" is a question about base types,
    // and nothing else here could answer it. Stops at the Godot boundary: every
    // screen descends from Control, and printing that adds noise to every dump.
    for (var b = t.BaseType; b is not null && b.Name != "Control" && b.Name != "Object"; b = b.BaseType)
        Console.WriteLine($"    base  {b.FullName}");
    foreach (var i in t.GetInterfaces().Where(i => (i.Namespace ?? "").StartsWith("MegaCrit")))
        Console.WriteLine($"    iface {i.FullName}");
    if (t.IsEnum)
    {
        foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.Static))
            Console.WriteLine($"    {f.Name}");
        Console.WriteLine();
        return;
    }
    bool Keep(string name) =>
        filter.Length == 0 || name.Contains(filter, StringComparison.OrdinalIgnoreCase);

    var flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static;
    if (includePrivate) flags |= BindingFlags.NonPublic;
    // Declared-only when private members are wanted: otherwise every Godot base
    // class dumps hundreds of internals and buries the handful of fields the
    // type actually declares, which is the thing being looked for.
    if (includePrivate) flags |= BindingFlags.DeclaredOnly;
    string Vis(bool isPublic) => isPublic ? " " : "-";

    foreach (var p in t.GetProperties(flags).Where(p => Keep(p.Name)).OrderBy(p => p.Name))
        Console.WriteLine($"   {Vis(p.GetMethod?.IsPublic ?? false)}prop  {p.PropertyType.Name,-30} {p.Name}");
    foreach (var f in t.GetFields(flags).Where(f => Keep(f.Name)).OrderBy(f => f.Name))
        Console.WriteLine($"   {Vis(f.IsPublic)}field {f.FieldType.Name,-30} {f.Name}");
    foreach (var m in t.GetMethods(flags).Where(m => !m.IsSpecialName && Keep(m.Name)).OrderBy(m => m.Name))
        Console.WriteLine($"   {Vis(m.IsPublic)}meth  {m.ReturnType.Name,-30} {m.Name}({string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name))})");
    Console.WriteLine();
}

// Reverse lookup: given a member name, find which types expose it. `Dump` can
// only answer "what does this type have", which is the wrong direction when you
// know the thing you want (`ActionQueueSet`, `IsEmpty`) but not where it hangs.
// Guessing candidate type names one at a time is how 2026-09-12 was spent.
void Find(string needle)
{
    Console.WriteLine($"### members matching \"{needle}\"");
    foreach (var t in sts2.GetTypes().OrderBy(t => t.FullName))
    {
        if (t.IsEnum) continue;
        var flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static;
        foreach (var p in t.GetProperties(flags))
            if (p.Name.Contains(needle, StringComparison.OrdinalIgnoreCase))
                Console.WriteLine($"    prop  {t.FullName}.{p.Name} : {p.PropertyType.Name}  (static={p.GetMethod?.IsStatic})");
        foreach (var f in t.GetFields(flags))
            if (f.Name.Contains(needle, StringComparison.OrdinalIgnoreCase))
                Console.WriteLine($"    field {t.FullName}.{f.Name} : {f.FieldType.Name}  (static={f.IsStatic})");
        foreach (var m in t.GetMethods(flags))
            if (!m.IsSpecialName && m.Name.Contains(needle, StringComparison.OrdinalIgnoreCase))
                Console.WriteLine($"    meth  {t.FullName}.{m.Name}() : {m.ReturnType.Name}  (static={m.IsStatic})");
    }
    Console.WriteLine();
}

// List every type whose full name contains a fragment. `find:` matches member
// names, `Dump` needs a type name you already know - neither answers "what is in
// this namespace", which is the question when you have found a promising corner of
// the assembly (e.g. MegaCrit.Sts2.Core.AutoSlay) and want its inventory.
void Types(string fragment)
{
    Console.WriteLine($"### types matching \"{fragment}\"");
    foreach (var t in sts2.GetTypes()
                          .Where(t => (t.FullName ?? "").Contains(fragment, StringComparison.OrdinalIgnoreCase))
                          .OrderBy(t => t.FullName))
        Console.WriteLine($"    {(t.IsEnum ? "enum " : t.IsInterface ? "iface" : "class")} {t.FullName}");
    Console.WriteLine();
}

// Every type implementing an interface. The bridge's state builder is one long
// `is` chain over IOverlayScreen implementers, so "which screens can sit on top
// and have no branch" is exactly this query - and `types:` cannot answer it,
// because the implementers are scattered across a dozen namespaces.
void Implementers(string ifaceName)
{
    Console.WriteLine($"### types implementing \"{ifaceName}\"");
    foreach (var t in sts2.GetTypes()
                          .Where(t => !t.IsInterface && !t.Name.Contains('<')
                                      && t.GetInterfaces().Any(i => i.Name == ifaceName))
                          .OrderBy(t => t.Name))
        Console.WriteLine($"    {t.Name,-42} {t.Namespace}");
    Console.WriteLine();
}

foreach (var spec0 in args.DefaultIfEmpty("CombatManager"))
{
    // `all:Type` = declared members including private ones. A `-` in the left
    // margin marks a non-public member: reachable from the mod, but a rename in
    // any patch will not be a compile error, so anything read this way needs the
    // same try/catch treatment the other reflective reads got on 2026-09-12.
    var spec = spec0;
    includePrivate = spec.StartsWith("all:", StringComparison.OrdinalIgnoreCase);
    if (includePrivate) spec = spec["all:".Length..];

    if (spec.StartsWith("impl:", StringComparison.OrdinalIgnoreCase))
    {
        Implementers(spec["impl:".Length..]);
        continue;
    }
    if (spec.StartsWith("find:", StringComparison.OrdinalIgnoreCase))
    {
        Find(spec["find:".Length..]);
        continue;
    }
    if (spec.StartsWith("types:", StringComparison.OrdinalIgnoreCase))
    {
        Types(spec["types:".Length..]);
        continue;
    }
    var parts = spec.Split('|');
    Dump(parts[0], parts.Length > 1 ? parts[1] : "");
}
