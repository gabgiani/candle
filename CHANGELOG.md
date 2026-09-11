# Changelog
This documents the main changes to the `candle` crate.

## DAFO fork milestone - 2026-09-11

Downstream changes on `feature/quantized-qwen35-ssm`, building on the Gemma 4
fork. This is not an upstream Candle version announcement.

### Quantized Qwen3.5/3.8 correctness

- Interpret `qwen35.nextn_predict_layers` separately from backbone layers; the
  tested Qwen3.8-27B artifact has 64 backbone layers and one auxiliary MTP block.
- Use additive causal masking rather than adding a binary mask to scores.
- Split the full-attention query and output gate **within each head**, not as
  two global halves of the projected tensor.
- Tile linear-attention Q/K heads to match the GGUF value-head ordering.
- Treat GGUF SSM `a` as the already-transformed negative decay coefficient;
  avoid exponentiating it a second time, and compute softplus stably.
- Apply recurrent-state decay before predicting the value/delta and scale the
  query by the inverse square root of its head dimension.
- Register `tokenizer.ggml.pre = qwen35` with the reference Unicode regex,
  NFC normalization and no ByteLevel prefix-space injection. Existing Qwen2
  and Llama3 tokenizer split tests continue to pass.

### State transfer for weight paging

- Add `LayerState` for full-attention KV state and linear-attention convolution
  and recurrent state, with transfer between devices.
- Add `LayerWeights::take_state` / `restore_state`, allowing callers to release
  layer weights, then restore inference state after reloading those weights.
- Reject mismatched cache/layer variants explicitly.

### Validation and responsibility boundary

Five focused Qwen tests cover per-head query/gate layout, tiled head order,
recurrent decay/query scaling, state transfer and the existing rotary check.
Two existing tokenizer split tests cover regression behavior. Swarm-node adds
real-GGUF tokenizer parity, resident/paged activation parity across tokens,
release/replay, cancellation/restart and a full-model two-paged-halves `OK`/EOS
check. Its two-device M4 + RTX 3060 Laptop deployment produced exact `OK` and
`one two three` replies with normal EOS termination.

Those smoke tests establish the tested quantized execution path, not model
quality or parity for every architecture, context, dtype or accelerator. The
disk pager and responsive network worker are supplied by swarm-node; this
fork supplies the model computation and movable state.

## v0.3.1 - Unreleased

### Added

### Modified

## v0.3.0 - 2023-10-01

### Added

- Added the Mistral 7b v0.1 model
  [983](https://github.com/huggingface/candle/pull/983).
- Quantized version of the Mistral model
  [1009](https://github.com/huggingface/candle/pull/1009).
- Add the gelu-erf op and activation function
  [969](https://github.com/huggingface/candle/pull/969).
- Add the mixformer/phi-v1.5 model
  [930](https://github.com/huggingface/candle/pull/930).
- Add the sclice-scatter op
  [927](https://github.com/huggingface/candle/pull/927).
- Add the Wuerstchen diffusion model
  [911](https://github.com/huggingface/candle/pull/911).

### Modified

- Support for simd128 intrinsics in some quantized vecdots
  [982](https://github.com/huggingface/candle/pull/982).
- Optimize the index-select cuda kernel
  [976](https://github.com/huggingface/candle/pull/976).
- Self-contained safetensor wrappers
  [946](https://github.com/huggingface/candle/pull/946).

## v0.2.2 - 2023-09-18

### Added
- Support for `top_p` sampling
  [819](https://github.com/huggingface/candle/pull/819).
- T5 model including decoding
  [864](https://github.com/huggingface/candle/pull/864).
- 1-d upsampling
  [839](https://github.com/huggingface/candle/pull/839).

### Modified
- Bugfix for conv2d
  [820](https://github.com/huggingface/candle/pull/820).
- Support tensor based indexing using `.i`
  [842](https://github.com/huggingface/candle/pull/842).

## v0.2.1 - 2023-09-11

### Added
- Add some RNNs (GRU and LSTM) in `candle-nn`
  [674](https://github.com/huggingface/candle/pull/674),
  [688](https://github.com/huggingface/candle/pull/688).
- gguf v2 support
  [725](https://github.com/huggingface/candle/pull/725).
- Quantized llama example in Python using the pyo3 api
  [716](https://github.com/huggingface/candle/pull/716).
- `candle-nn` layer for conv2d-transposed
  [760](https://github.com/huggingface/candle/pull/760).
- Add the Segment-Anything Model (SAM) as an example
  [773](https://github.com/huggingface/candle/pull/773).
- TinyViT backbone for the segment anything example
  [787](https://github.com/huggingface/candle/pull/787).
- Shape with holes support
  [770](https://github.com/huggingface/candle/pull/770).

### Modified
- Dilations are now supported in conv-transpose2d.
  [671](https://github.com/huggingface/candle/pull/671).
- Interactive mode for the quantized model
  [690](https://github.com/huggingface/candle/pull/690).
- Faster softmax operation
  [747](https://github.com/huggingface/candle/pull/747).
- Faster convolution operations on CPU and CUDA via im2col
  [802](https://github.com/huggingface/candle/pull/802).
- Moving some models to a more central location
  [796](https://github.com/huggingface/candle/pull/796).

## v0.2.0 - 2023-08-30

### Added
- Add the powf op
  [664](https://github.com/huggingface/candle/pull/664).
- Stable Diffusion XL support
  [647](https://github.com/huggingface/candle/pull/647).
- Add the conv-transpose2d op
  [635](https://github.com/huggingface/candle/pull/635).
- Refactor the VarBuilder api
  [627](https://github.com/huggingface/candle/pull/627).
- Add some quantization command
  [625](https://github.com/huggingface/candle/pull/625).
- Support more quantized types, e.g. Q2K, Q4K, Q5K...
  [586](https://github.com/huggingface/candle/pull/586).
- Add pose estimation to the yolo example
  [589](https://github.com/huggingface/candle/pull/589).
- Api to write GGUF files
  [585](https://github.com/huggingface/candle/pull/585).
- Support more quantization types
  [580](https://github.com/huggingface/candle/pull/580).
- Add EfficientNet as an example Computer Vision model
  [572](https://github.com/huggingface/candle/pull/572).
- Add a group parameter to convolutions
  [566](https://github.com/huggingface/candle/pull/566).
- New dtype: int64
  [563](https://github.com/huggingface/candle/pull/563).
- Handling of the GGUF file format.
  [559](https://github.com/huggingface/candle/pull/559).

## v0.1.2 - 2023-08-21
