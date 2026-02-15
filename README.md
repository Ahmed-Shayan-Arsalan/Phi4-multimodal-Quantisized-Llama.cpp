# Phi-4 Multimodal Quantized — llama.cpp

This repository is a **custom fork of llama.cpp** for running **Microsoft Phi-4 multimodal** (text + vision, with audio Conformer support) as GGUF. It targets the **Phi-4 multimodal instruct** model: quantized LLM + multimodal projector (vision encoder and optional audio encoder) via `llama-server` or `llama-cli`.

---

## What This Fork Delivers

- **Phi-4 multimodal** running locally: **text**, **images**, and (in code) **audio** via a single stack.
- **Model files**: quantized LLM (`phi4-mm-Q4_K_M.gguf`) and multimodal projector (`phi4-mm-omni.gguf`) built from Hugging Face checkpoints.
- **Correctness fixes** for the Phi-4 **audio Conformer** encoder (SwiGLU ordering, Conv GLU activation, post-depthwise pointwise conv, T5-style relative attention bias) so behavior matches the original PyTorch design when audio is enabled.
- **Conversion pipeline**: custom handling in `convert_hf_to_gguf.py` for Phi-4’s `embed_tokens_extend` (vision + audio), mapping to GGUF tensors expected by our `clip`/`conformer` code in `tools/mtmd`.

---

## System Overview

| Component | Role |
|----------|------|
| **LLM** | `phi4-mm-Q4_K_M.gguf` — Phi-4 language model, Q4_K_M quantized. |
| **Multimodal projector (mmproj)** | `phi4-mm-omni.gguf` — Vision encoder (SigLIP/Navit-style) and, when enabled, Conformer audio encoder. Stored largely F16/F32 for quality. |
| **Server** | `llama-server` from this repo (or `llama-cli`) with Phi-4 chat template and multimodal support. |

We assume you run **vision + text** (and optionally **audio**) using the **Q4_K_M** LLM plus the **omni** mmproj.

---

## How We Built the Model Files

### 1. Source model

- **Hugging Face**: [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct), or a local clone (e.g. `phi-4-multimodal`) with the same layout (e.g. `config.json`, `model.safetensors.index.json`, `*.safetensors`, `tokenizer.json`, etc.).

### 2. Export the text LLM to GGUF

From the **llama.cpp** repo root, with the Phi-4 model directory at `<PATH_TO_PHI4>` (e.g. `../phi-4-multimodal` or a downloaded HF path):

```bash
# Full precision (for later quantization)
python convert_hf_to_gguf.py <PATH_TO_PHI4> --outtype f16 --outfile phi4-mm-f16.gguf
```

The script detects **Llama4ForConditionalGeneration** (Phi-4’s text architecture) and exports the **language model only**; vision/audio extend tensors are skipped for this step.

### 3. Quantize the LLM

```bash
# Q4_K_M (recommended for speed/quality tradeoff)
./build/bin/llama-quantize phi4-mm-f16.gguf phi4-mm-Q4_K_M.gguf Q4_K_M

# Other options if you built them:
# Q8_0:  ./build/bin/llama-quantize phi4-mm-f16.gguf phi4-mm-vision-q8.gguf Q8_0
# F16:   use phi4-mm-f16.gguf as-is (e.g. phi4-mm-vision.gguf)
```

Result: **phi4-mm-Q4_K_M.gguf** (and optionally other variants).

### 4. Export the multimodal projector (mmproj)

The **mmproj** contains the vision encoder and, in the code path, the audio Conformer. Our fork **forces** mmproj extraction to use the Phi-4 vision (and audio) mapping when `--mmproj` is passed:

```bash
python convert_hf_to_gguf.py <PATH_TO_PHI4> --mmproj --outtype f16 --outfile mmproj-phi4-omni.gguf
```

The script uses **Phi3MiniModel** (registered for Phi-4’s config) to:

- Read **vision** from `model.embed_tokens_extend.image_embed.*` and map to standard CLIP-style tensor names expected by `clip.cpp` (e.g. `v.*`).
- Read **audio** from `model.embed_tokens_extend.audio_embed.*` and map to Conformer names expected by `conformer.cpp` (e.g. `a.*`, `mm.a.mlp.*`). (Audio mapping can be toggled in code; see “Our changes to llama.cpp” below.)

Rename or copy the output to **phi4-mm-omni.gguf** (or keep the `mmproj-` prefix if you prefer).

### 5. Summary of produced files

| File | Purpose |
|------|--------|
| `phi4-mm-f16.gguf` | Full-precision LLM (optional; used to produce quantized LLMs). |
| `phi4-mm-Q4_K_M.gguf` | Quantized LLM for normal use. |
| `phi4-mm-omni.gguf` | Multimodal projector (vision + optional audio). |
| `phi4-mm-vision.gguf` / `phi4-mm-vision-q8.gguf` | Optional LLM variants (e.g. F16 or Q8) if you built them. |

No other repo is copied; everything is built from the Phi-4 checkpoint and this fork’s scripts.

---

## How We Set Up the System

### Prerequisites

- **Windows**: Visual Studio 2022 (Desktop C++), CMake 3.26+, optional CUDA for GPU.
- **Python**: 3.x with `torch`, `safetensors`, and dependencies required by `convert_hf_to_gguf.py` (see repo/scripts).
- **Model files** (built as above or obtained elsewhere): place in a folder of your choice, e.g. next to the repo or in a dedicated `models` dir:
  - `phi4-mm-Q4_K_M.gguf`
  - `phi4-mm-omni.gguf`

### Build llama.cpp

From the **llama.cpp** root (this repo):

```bash
cmake -B build
cmake --build build --config Release
```

Optional: enable CUDA, Metal, etc. (see [docs/build.md](docs/build.md)).

### Run the server

Point the server at the LLM and the mmproj:

```bash
./build/bin/llama-server -m path/to/phi4-mm-Q4_K_M.gguf -mm path/to/phi4-mm-omni.gguf --host 0.0.0.0 --port 8080
```

Or use **llama-cli** with `-m` and `-mm` for local inference. The server uses the **phi4** chat template and supports multimodal prompts (images and, when wired, audio).

### Regenerating the mmproj

If you change the Phi-4 source or our conversion logic, regenerate the mmproj with the same command as in step 4 above; no need to re-quantize the LLM unless you changed the text model.

---

## Our Updates to llama.cpp

### 1. `convert_hf_to_gguf.py`

- **Phi-4 as Mmproj**: We use **Phi3MiniModel** (registered for `Phi3ForCausalLM`; Phi-4 config is compatible) to export the **mmproj** when `--mmproj` is passed. The main logic forces mmproj extraction to this class so that `embed_tokens_extend` is handled.
- **Vision**:  
  - **Filter**: `model.embed_tokens_extend.image_embed.*` tensors are included.  
  - **Mapping**: `_map_to_clip_tensor_name()` maps Phi-4 vision names (e.g. `img_processor.*`, `img_projection.*`, `glb_GN`, `sub_GN`) to the CLIP-style names expected by our mtmd/clip code.  
  - **Config**: Vision config is derived from Phi-4’s `image_embd_layer` (e.g. hidden_size 1152, 27 layers, patch_size 14).
- **Audio (Conformer)**:  
  - **Filter**: `model.embed_tokens_extend.audio_embed.*` tensors are included.  
  - **Mapping**: `_map_to_conformer_tensor_name()` maps Phi-4 Conformer names to the LFM2A-style names used in `conformer.cpp` (e.g. `a.conv1d.*`, `a.blk.*`, `a.rel_attn_bias`, `mm.a.mlp.*`).  
  - **Synthetic tensors**: We emit identity/zero placeholders for tensors Phi-4 doesn’t have but the conformer graph expects (e.g. `pos_bias_u/v`, `linear_pos`, `conv_norm`, `a.position_embd`).  
  - **Shape/quant**: Conv biases reshaped for ggml; GLU splits for FFN; F32 for conv weights where needed.  
  - **Toggle**: In the current tree, audio mapping can be disabled by returning `None` at the top of `_map_to_conformer_tensor_name()` (vision-only mmproj).
- **Text**: For non-multimodal export, the loader uses the detected architecture (e.g. **Llama4ForConditionalGeneration**) and exports only the language model; vision/audio extend tensors are skipped.
- **Vocab**: Phi-4 uses `tokenizer.json`; we use the GPT-2–style vocab path so the script finds it.

### 2. `tools/mtmd/models/conformer.cpp`

- **SwiGLU ordering**: Phi-4 uses `value * silu(gate)` with value = first half, gate = second half. We use **ggml_swiglu_swapped** so SiLU is on the second half and multiplied by the first.
- **Conv GLU**: For `conv_glu_type = "swish"` we use **ggml_silu** (Swish = SiLU) instead of sigmoid on the gate.
- **Post-depthwise pointwise conv**: Phi-4’s `DepthWiseSeperableConv1d` has a **pw_conv** after the depthwise conv. We load it as **conv_pw_mid** and use it in the conformer graph between depthwise and activation.
- **T5-style relative attention bias**: We load `encoder.relative_attention_bias_layer.bias_values.weight` as `a.rel_attn_bias`, build a (n_head, T, T) bias on CPU, and add it to attention scores (scaled dot-product + bias) instead of the LFM2A pos_bias_u/v + linear_pos + rel_shift path.

### 3. GGUF / mmproj metadata

- For mmproj we set file type and vision/audio-related parameters from Phi-4’s config (no standard preprocessor_config); we use **hidden_states[-2]** (no post_layernorm) for the vision feature layer.

### 4. Chat template

- The server and CLI use the **phi4** chat template for Phi-4 multimodal.

---

## Quantization Summary

| Asset | Quantization |
|-------|----------------|
| **LLM** (`phi4-mm-Q4_K_M.gguf`) | Q4_K_M |
| **mmproj** (`phi4-mm-omni.gguf`) | Mostly F16/F32 for vision and audio encoder layers |

---

## File Layout (reference)

Place the built GGUF files where your run script or server expects them, for example:

```
llama.cpp/
├── build/                    # build output (llama-server, llama-quantize, etc.)
├── convert_hf_to_gguf.py     # our conversion + Phi-4 mmproj/vision/audio mapping
├── phi4-mm-Q4_K_M.gguf       # (or in a separate models/ dir)
├── phi4-mm-omni.gguf
├── tools/mtmd/
│   └── models/conformer.cpp  # Conformer + Phi-4 audio fixes
└── README.md                 # this file
```

---

## References

- [Phi-4 multimodal (Microsoft)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- [Phi-4 technical report](https://arxiv.org/abs/2503.01743)
- [llama.cpp build docs](docs/build.md)
- [Multimodal / mtmd](tools/mtmd/README.md)
- [Server options](tools/server/README.md)
