# Example multi-model preset for `llama-server`

Upstream supports **`--models-preset path/to/models.ini`** so one `llama-server` process can load multiple GGUFs with per-model defaults.

1. Copy **`models.ini.example`** → **`models.ini`** (outside the repo or in a private path).
2. Replace **`MODEL_PATH_*`** placeholders with real GGUF paths (or Ollama blob paths if you use that layout).
3. Build this repo and run:

   ```text
   llama-server --models-preset models.ini --models-max 1 --host 127.0.0.1 --port 11436
   ```

Adjust **`--models-max`**, **`ctx-size`**, **`parallel`** in the ini for your VRAM. TurboQuant KV types use **`--cache-type-k` / `--cache-type-v`** in the global **`[*]`** section or per-model blocks as supported by your build; see the main **[README](../../README.md)**.
