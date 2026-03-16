# WinLLM CLI Reference

```
winllm <command> [options]
```

---

## Commands

| Command | Description |
|---|---|
| `serve` | Start an OpenAI-compatible API server |
| `chat` | Interactive chat session in the terminal |
| `benchmark` | Run a throughput benchmark |
| `list` | List downloaded models from HuggingFace cache |
| `detect` | Detect and display hardware info |

---

## `winllm serve`

Start the API server with OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`).

```bash
winllm serve --model <model> [options]
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | *required* | HuggingFace model name or local path |
| `--quantization` | `-q` | `auto` | Quantization: `auto`, `none`, `4bit`, `8bit` |
| `--max-model-len` | | Auto-detected | Maximum context length in tokens |
| `--trust-remote-code` | | `false` | Allow remote code execution for custom models |
| `--verbose` | `-v` | `false` | Enable debug logging |
| `--host` | | `0.0.0.0` | Server bind address |
| `--port` | `-p` | `8000` | Server port |
| `--max-batch-size` | | `4` | Maximum concurrent requests |
| `--model-alias` | | Model name | Override model name in API responses |
| `--gpu-memory-utilization` | | `0.90` | Fraction of GPU memory to use |
| `--attention-backend` | | `auto` | Attention: `auto`, `sdpa`, `flash_attention_2`, `eager` |
| `--compile` | | `false` | Use torch.compile for graph optimization |
| `--draft-model` | | None | Path to draft model for speculative decoding |
| `--tensor-parallel-size` | `-tp` | `1` | Number of GPUs for tensor parallelism |
| `--device-map-strategy` | | `auto` | GPU distribution: `auto`, `balanced`, `balanced_low_0`, `sequential` |
| `--cpu-offload` | | `false` | Offload excess layers to CPU RAM |
| `--device` | | `auto` | Device: `auto`, `cuda`, `cuda:0`, `cpu` |
| `--auto-config` | | `false` | Auto-detect hardware and set optimal config |

### Example

```bash
# 4-bit quantized, auto hardware config
winllm serve -m meta-llama/Llama-2-7b-chat-hf --auto-config

# Full precision on specific GPU
winllm serve -m mistralai/Mistral-7B-v0.1 -q none --device cuda:0 --port 9000

# Multi-GPU with tensor parallelism
winllm serve -m meta-llama/Llama-2-70b-chat-hf -tp 4 --auto-config
```

---

## `winllm chat`

Interactive terminal chat with streaming output.

```bash
winllm chat --model <model> [options]
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | *required* | HuggingFace model name or local path |
| `--quantization` | `-q` | `auto` | Quantization: `auto`, `none`, `4bit`, `8bit` |
| `--max-model-len` | | Auto-detected | Maximum context length |
| `--trust-remote-code` | | `false` | Allow remote code execution |
| `--verbose` | `-v` | `false` | Enable debug logging |
| `--max-tokens` | | `512` | Maximum tokens per response |
| `--temperature` | | `0.7` | Sampling temperature (0 = greedy) |
| `--system-prompt` | `-s` | None | System prompt for the conversation |
| `--attention-backend` | | `auto` | Attention: `auto`, `sdpa`, `flash_attention_2`, `eager` |
| `--compile` | | `false` | Use torch.compile for graph optimization |
| `--draft-model` | | None | Path to draft model for speculative decoding |
| `--tensor-parallel-size` | `-tp` | `1` | GPUs for tensor parallelism |
| `--device-map-strategy` | | `auto` | GPU distribution strategy |
| `--cpu-offload` | | `false` | Offload layers to CPU |
| `--device` | | `auto` | Device selection |
| `--auto-config` | | `false` | Auto-detect hardware |

### Example

```bash
# Quick chat with a small model
winllm chat -m Qwen/Qwen1.5-1.8B-Chat --auto-config

# Chat with a system prompt
winllm chat -m meta-llama/Llama-2-7b-chat-hf -s "You are a helpful coding assistant." --temperature 0.3

# Accelerated chat with speculative decoding and torch.compile
winllm chat -m meta-llama/Llama-3-8B --draft-model meta-llama/Llama-3-GGUF --compile
```

Type `quit`, `exit`, or `q` to end the session.

---

## `winllm benchmark`

Run a throughput benchmark with built-in prompts.

```bash
winllm benchmark --model <model> [options]
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | *required* | HuggingFace model name or local path |
| `--quantization` | `-q` | `auto` | Quantization: `auto`, `none`, `4bit`, `8bit` |
| `--max-model-len` | | Auto-detected | Maximum context length |
| `--trust-remote-code` | | `false` | Allow remote code execution |
| `--verbose` | `-v` | `false` | Enable debug logging |
| `--max-tokens` | | `256` | Max tokens per prompt |
| `--num-prompts` | | `5` | Number of prompts to run (max 5) |
| `--attention-backend` | | `auto` | Attention: `auto`, `sdpa`, `flash_attention_2`, `eager` |
| `--compile` | | `false` | Use torch.compile for graph optimization |
| `--draft-model` | | None | Path to draft model for speculative decoding |
| `--tensor-parallel-size` | `-tp` | `1` | GPUs for tensor parallelism |
| `--device-map-strategy` | | `auto` | GPU distribution strategy |
| `--cpu-offload` | | `false` | Offload layers to CPU |
| `--device` | | `auto` | Device selection |
| `--auto-config` | | `false` | Auto-detect hardware |

### Example

```bash
winllm benchmark -m meta-llama/Llama-2-7b-chat-hf --auto-config --max-tokens 128
```

---

## `winllm list`

List all models downloaded to the local HuggingFace cache.

```bash
winllm list
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--verbose` | `-v` | `false` | Enable debug logging |

### Example

```bash
winllm list
```

**Sample output:**

```
  MODEL                                       SIZE       MODIFIED
  ─────────────────────────────────────   ──────────   ────────────────
  meta-llama/Llama-2-7b-chat-hf              3.9 GB   2025-02-14 09:32
  microsoft/Phi-3-mini-4k-instruct           2.3 GB   2025-03-01 15:10
  Qwen/Qwen1.5-1.8B-Chat                    1.1 GB   2025-03-10 22:45

  3 model(s), 7.3 GB total
  Cache: C:\Users\you\.cache\huggingface\hub
```

The cache directory is resolved from `HF_HOME` or `HUGGINGFACE_HUB_CACHE` environment variables, falling back to `~/.cache/huggingface/hub`.

---

## `winllm detect`

Detect GPUs and display the recommended hardware profile with auto-tuned defaults.

```bash
winllm detect [options]
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--json` | | `false` | Also print machine-readable JSON output |
| `--verbose` | `-v` | `false` | Enable debug logging |

### Example

```bash
winllm detect --json
```

### Dynamic Defaults

All recommended defaults (quantization, batch size, context length, attention backend, etc.) are calculated mathematically from your actual VRAM and GPU count — there are no static profiles. Override any default with `WINLLM_*` environment variables.
