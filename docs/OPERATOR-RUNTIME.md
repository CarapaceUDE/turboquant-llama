# Operator runtime folder (your local “server kit”)

There are **two different things** people often mix up:

| Layer | What it is | Where it lives |
|--------|------------|----------------|
| **A. Forked `llama-server` binary** | Build output from this repo (`llama.exe`, CUDA, TurboQuant, any merged `tools/server/` C++ changes). | `build/bin/Release/` after CMake; source under **`tools/server/`** in **this repo** (published as [turboquant-llama](https://github.com/CarapaceUDE/turboquant-llama)). |
| **B. Local launch kit** | Scripts + **`models.ini`** + client JSON snippets + logs/cache dirs. **Machine-specific paths**, ports, and integration with *your* apps. | e.g. a folder on your PC (historically named `llama-turboquant-openclaw`). **Not** required for the fork to be useful to others. |

Throughout maintenance work we treated **layer A** (git + build) as authoritative. **Layer B** lives outside the repo unless you choose to version it—so it was easy to miss unless we asked “where do you double-click from?”

---

## What your current kit does (typical pattern)

- **`Start-Llama-For-OpenClaw.cmd`**: sets a dummy `HF_HUB_CACHE`, points at **`build\bin\Release\llama-server.exe`**, runs with **`--models-preset`**, **`--models-max`**, host/port (e.g. `11436`).
- **`models.ini`**: upstream-supported **preset file** ([`--models-preset`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) with **`[*]`** defaults and per-model sections (`model`, `alias`, `ctx-size`, `parallel`, …). Your paths point at **Ollama blobs** and local GGUFs—that is **private data**.
- **`OPENCLAW-SNIPPET.json`**: **client config** (provider URL + model list) for an app that speaks OpenAI-compatible HTTP—rename/replace when that integration changes.

None of that **replaces** the need to build from this repo; it **configures** the stock fork binary for your desk.

---

## Rename / cleanup (recommended locally)

The folder name **`llama-turboquant-openclaw`** is misleading if OpenClaw is gone. Suggested rename:

- e.g. **`llama-turboquant-runtime`**, **`carapace-llama-server`**, or **`turboquant-local-server`**.

After renaming:

1. Update **`C:\app\Start-Llama-Server.lnk`** to target the **`.cmd`** in the new path (or move the `.cmd` under `C:\app\...` and point the shortcut there).
2. Rename **`Start-Llama-For-OpenClaw.cmd`** → **`Start-Llama-Server.cmd`** (optional clarity).
3. Refresh **`README.txt`** to drop OpenClaw-only wording.

No need to publish your **absolute paths** or client secrets.

---

## What to expose “to the outside world”

**Already public:** this fork’s **`llama-server`** and server source (merge of upstream + TurboQuant branch). Others build and run the same binary.

**Worth publishing if you want copy-paste ergonomics:**

- A **sanitized** preset: see **[examples/models-preset/](../examples/models-preset/)** (`models.ini.example` + short README). Users substitute their own `model = ...` paths and ports.
- This doc (**OPERATOR-RUNTIME**) so people understand: *binary from git, presets local*.

**Usually keep private:**

- Real **`models.ini`** with your disk paths and model lineup.
- Client-specific JSON for your internal apps.
- **`logs/`**, cache dirs.

**If you have extra C++ behavior** that never landed in `CarapaceUDE/turboquant-llama`, that **should** be committed and pushed there—or it isn’t really “available to the outside world.” Compare your `tools/server/` tree to `upstream/master` and your publishing remote.

---

## Relationship to [MAINTAINING-FORK.md](MAINTAINING-FORK.md)

After each upstream merge: rebuild **`build/`**, then **your launcher keeps working** as long as **`LLAMA_EXE`** still points at `build\bin\Release\llama-server.exe` and your **`models.ini`** stays valid for the new server flags.
