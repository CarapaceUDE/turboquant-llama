# Maintaining this fork (upstream + TurboQuant)

This playbook is the **standard pattern** for pulling in **ggml-org/llama.cpp** while keeping **TurboQuant**, Qwen/hybrid patches, and local tooling working. Follow it each time you refresh from upstream.

---

## 1. One-time: remotes

| Remote    | Purpose |
|-----------|---------|
| `upstream` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — read-only source of upstream `master`. |
| `origin`   | Your legacy or backup remote (e.g. spiritbuun) — optional; may host **`feature/turboquant-kv-cache`** if new CUDA work lands there first. |
| `carapace` / publishing | e.g. [CarapaceUDE/turboquant-llama](https://github.com/CarapaceUDE/turboquant-llama) — what users clone. |

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp.git   # if missing
git fetch upstream
```

Do **not** use a **shallow** clone for maintainers who need to push full history (shallow clones broke pushes earlier). Use `git fetch --unshallow` if needed.

---

## 2. Canonical build directory (standardize paths)

Use a **single** out-of-tree build dir so scripts and shortcuts do not drift:

- **Recommended:** `build/` at the repo root (e.g. `C:\app\llama-cpp-turboquant-cuda\build`).
- **Windows example (CUDA + FA for TurboQuant):**

  ```powershell
  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j
  ```

- **Binaries:** `build\bin\Release\llama-server.exe`, `llama-cli.exe`, and matching `ggml*.dll`.

Point **`Start-Llama-Server`** / **`.cmd`** launchers at this path only. Avoid ad-hoc dirs like `build-merge-verify` long-term (use them only for experimental compares, then delete or ignore in docs).

---

## 3. Routine upstream sync (merge pattern)

We use **merge commits** from `upstream/master` into **`master`** (not force-push history rewrites). This matches how this repo was already integrated.

### Before merging

1. **Working tree clean** (`git status`).
2. **Optional safety tag** (if you want an easy reset point):

   ```sh
   git tag -a maint/before-upstream-YYYY-MM-DD -m "snapshot before merging upstream"
   ```

3. **Fetch everything:**

   ```sh
   git fetch upstream
   git fetch origin   # if TurboQuant work still appears on origin/feature/* first
   ```

### Merge upstream

```sh
git checkout master
git merge upstream/master -m "merge: sync ggml-org/llama.cpp upstream/master (YYYY-MM-DD)"
```

### If there are conflicts — typical hotspots

Upstream churn overlaps TurboQuant and recent llama.cpp changes most often here:

| Area | Files / patterns |
|------|------------------|
| CUDA Flash Attention + BF16 + turbo types | `ggml/src/ggml-cuda/fattn*.cu`, `fattn*.cuh`, `CMakeLists.txt` (vec template globs), `template-instances/fattn-vec-*` |
| TurboQuant CUDA helpers | `ggml/src/ggml-cuda/turbo-*.cu`, `turbo-quant-cuda.cuh`, `getrows.cu`, `set-rows.cu` |
| Attention graph / WHT | `src/llama-graph.cpp` (turbo inverse WHT vs `self_v_rot`) |
| KV cache | `src/llama-kv-cache*.cpp`, `src/llama-kv-cache.h` |
| Server | `tools/server/*` (if upstream touched server and your branch diverged) |

**Resolution rule of thumb:** keep **both** upstream fixes (e.g. **BF16** FA paths) **and** **TurboQuant** branches in `switch` / `if constexpr` / CMake source lists—same idea as the BF16 + turbo merge done earlier on this fork.

### After merging

1. **Rebuild** the canonical `build/` (Release).
2. **Smoke:** `llama-cli --version`, `llama-server --version`; one **turbo3** + **`-fa`** run (see README).
3. **Commit** any resolved conflict result (already part of merge if commit succeeded).
4. **Push** publishing remote(s):

   ```sh
   git push carapace master
   # and/or other remotes
   ```

---

## 4. When new TurboQuant work lands on `origin/feature/turboquant-kv-cache`

If CUDA/KV improvements are developed on **spiritbuun**’s feature branch **before** `master` here:

```sh
git fetch origin feature/turboquant-kv-cache
git checkout master
git merge origin/feature/turboquant-kv-cache -m "merge: TurboQuant feature branch"
```

Expect conflicts similar to §3. Then **merge `upstream/master` again** if upstream moved, or merge feature **onto** an already-updated `master` (order depends on whether you want “upstream first” or “turbo first”; usually **upstream first**, then feature, minimizes drift from ggml-org).

Document the chosen order in the merge commit message.

---

## 5. Release / “known good” checklist

Before you tell others the tree is good:

- [ ] `build/` Release: `llama-server`, `llama-cli` succeed.
- [ ] TurboQuant: e.g. `-ctk turbo3 -ctv turbo3` with flash attention as required; spot-check perplexity or a short generation.
- [ ] Models you care about: Qwen 3.5 / hybrid GGUFs load (secondary patches).
- [ ] Launchers point at `build\bin\Release\...`.

---

## 6. Related docs

- **[FORK-CHANGES.md](FORK-CHANGES.md)** — what differs from upstream and why.  
- **[build.md](build.md)** — upstream build options.  
- **[TURBOQUANT_CUDA_IMPLEMENTATION.md](../TURBOQUANT_CUDA_IMPLEMENTATION.md)** — TurboQuant CUDA details.  
- **[README.md](../README.md)** — user-facing TurboQuant usage.

---

## License

Documentation only; does not change project licenses.
