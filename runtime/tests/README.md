# Tests

Planned test layers:

- unit: allocator arithmetic, admission decisions, KV page lifetimes;
- integration: model load and synthetic execution paths;
- perf: decode/prefill benchmark regressions;
- fault_injection: OOM, invalid page release, graph address instability, strict-mode rejection.

The first executable validation is currently:

```powershell
python .\tools\bench\numba_residency_bench.py --steps 32 --weights-mib 64 --kv-pages 64
```

Policy unit tests:

```powershell
python -m unittest .\runtime\tests\unit\test_policy_sim.py
```

Once Visual C++ build tools are available, native perf smoke tests are:

```powershell
cmake --preset windows-vs2022-cuda86-release
cmake --build --preset windows-vs2022-cuda86-release
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_bench.exe"
& "E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release\runtime_graph_bench.exe"
```
