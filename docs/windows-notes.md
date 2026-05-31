# Windows Notes

This project targets Windows 11 on a GeForce RTX 3060 in WDDM mode.

## WDDM Constraints

The GPU watchdog is enabled on this machine. Runtime work should be split into short, replayable units. Persistent state is desirable; one unbounded persistent kernel is not.

## No Silent Spill

Default policy is `strict`. If a request cannot reserve model, KV, and workspace memory up front, it should be rejected or queued. Host spill belongs behind an explicit `overflow` policy because it changes both performance and desktop responsiveness.

## Native Toolchain

CUDA on Windows needs the Visual C++ compiler. On this machine, `cl.exe` is not visible in a plain PowerShell shell, but Visual C++ tools are installed under:

```text
D:\Visual build tools
```

Build through the detected `vcvars64.bat` environment:

```powershell
.\scripts\check_toolchain.ps1
.\scripts\build_native_vs.ps1
```

The Visual Studio generator is the currently working native path. The Ninja preset can pick up WinLibs `windres.exe`/`ld.exe` from PATH and fail CUDA try-compile with a mixed MSVC/MinGW link environment, so prefer `windows-vs2022-cuda86-release` until the Ninja environment is pinned.

Native build outputs:

```text
E:\AI project\build\rtx3060-runtime\windows-vs2022-cuda86-release\Release
```

## Cache and Temporary Files

Use:

```powershell
. .\scripts\env.ps1
```

before downloads or builds. It sets Hugging Face and temporary paths to `E:\AI project`.
