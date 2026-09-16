# api_probe — 游戏程序集 API 探测器

游戏更新后桥接模组编译失败时用这个查"这个 API 变成什么了"。

用元数据反射（`MetadataLoadContext`）读 `sts2.dll`，**只读元数据不执行代码**，
所以不需要 Godot 运行时，也不会启动游戏。

## 用法

```bash
export PATH="$PATH:/c/Program Files/dotnet"
cd tools/api_probe
dotnet run -- "CombatManager"              # 列出该类型的全部成员
dotnet run -- "CombatManager|Phase"        # 只列名字含 Phase 的成员
dotnet run -- "ICombatState" "MerchantRoom"  # 一次查多个
```

类型名可以写全名（`MegaCrit.Sts2.Core.Combat.CombatManager`）或短名（`CombatManager`）。
枚举类型会直接列出所有取值。

游戏目录写死在 `Program.cs` 的 `dataDir`，换机器要改。

## 为什么需要它

2026-09-07 桥接对 v0.107.1 编译失败时，`strings` 只能看出
`get_IsPlayPhase` 消失了，看不出**替代品是什么**。这个工具直接给出答案：
`CombatManager` 上有 `IsPartOfPlayerTurn(Player)`——布尔属性改成了按玩家提问的方法。

## 排错流程（下次游戏更新照做）

1. `dotnet build` 桥接 → 编译器一次性列出所有不匹配（**不要一个个试运行**）
2. 对每个报错的类型跑 `api_probe` → 找替代 API
3. 改源码 → 重编 → 退出游戏 → 替换 `<游戏>\mods\STS2_Bridge\STS2_Bridge.dll`
4. 进游戏实际打一场验证
