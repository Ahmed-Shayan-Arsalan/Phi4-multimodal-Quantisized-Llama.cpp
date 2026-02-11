Set-Content README.md @"
## phi4-omni-llama.cpp

This repository is a customized fork of `llama.cpp` focused on running **Phi-4 multimodal (text+audio+vision)** with the **phi4-mm-Q4_K_M.gguf** LLM and the **phi4-mm-omni.gguf** multimodal projector (“ONI model/brain”) via `llama-server`.

It contains several correctness fixes and architectural adaptations for the **Phi-4 audio conformer encoder**, so that audio transcription from WAV input behaves much closer to the original PyTorch implementation.

This README explains:

- **What is different from upstream `llama.cpp`**
- **What quantization levels are used and where**
- **How to build and run the server on a fresh machine**
- **How to regenerate the `mmproj` GGUF if needed**

---

### 1. What this fork does

#### 1.1. Targeted model setup

- **LLM**: `phi4-mm-Q4_K_M.gguf`
  - Quantized Phi-4 multimodal language model.
  - Uses **Q4_K_M** group-wise 4‑bit quantization for weights.
- **Multimodal projector (mmproj)**: `phi4-mm-omni.gguf`
  - Contains the **vision and audio encoders**, including the **Conformer audio encoder**.
  - Stored mostly in **F16 / F32** for audio-critical parts to preserve quality.
- **Server**: `llama-server` from this repo, with extended audio support and conformer fixes.

This fork assumes you want to run **audio + text (and optionally vision)** using the **Q4_K_M brain + omni projector**, not the full F16 base model.

#### 1.2. Audio conformer fixes

- **SwiGLU gate/value ordering**
  - Upstream used `ggml_swiglu` (SiLU on first half).
  - Phi‑4 uses `value * silu(gate)` with `value = first_half`, `gate = second_half`.
  - Fix: use **`ggml_swiglu_swapped`** so SiLU is applied to the second half and multiplied by the first.

- **Conv GLU activation**
  - Phi‑4 config uses `conv_glu_type = "swish"` where Swish(x) = x * sigmoid(x) = SiLU(x).
  - Upstream used `ggml_sigmoid` on the gate.
  - Fix: use **`ggml_silu`** for `value * silu(gate)`.

- **Post‑depthwise pointwise conv**
  - Phi‑4’s `DepthWiseSeperableConv1d` has a post‑depthwise **`pw_conv`** that was skipped in GGUF conversion.
  - Fix:
    - Map `conv.dw_sep_conv_1d.pw_conv.{weight,bias}` to `a.blk.{i}.conv_pw_mid.{weight,bias}` in `convert_hf_to_gguf.py`.
    - Use `conv_pw_mid` in `conformer.cpp` between depthwise conv and activation.

#### 1.3. T5‑style relative attention bias

Phi‑4 audio uses **T5‑style relative attention bias**:

\[
\text{scores} = QK^T + B_{\text{rel}}
\]

This fork:

- Loads `encoder.relative_attention_bias_layer.bias_values.weight` into GGUF.
- Computes a `(n_head, T, T)` bias matrix on CPU from that embedding for each sequence length \(T\).
- Rewrites conformer attention in `conformer.cpp` to use:
  - Scaled dot product \(QK^T / \sqrt{d_k}\)
  - Plus the T5 relative bias instead of LFM2A’s `pos_bias_u/v + linear_pos + rel_shift`.

---

### 2. Quantization

- **LLM**: `phi4-mm-Q4_K_M.gguf` – `Q4_K_M` block 4‑bit quantization.
- **mmproj**: `phi4-mm-omni.gguf` – mostly F16/F32 on critical audio/vision encoder layers.

---

### 3. Prerequisites

- **Windows**:
  - Visual Studio 2022 (Desktop C++), CMake 3.26+, CUDA (optional).
- **Model files** (not in git, download separately and put next to this README):
  - `phi4-mm-Q4_K_M.gguf`
  - `phi4-mm-omni.gguf`

---
