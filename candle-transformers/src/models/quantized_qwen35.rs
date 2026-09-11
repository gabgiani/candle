//! Qwen 3.5 / Qwen 3.8 hybrid (Linear Attention / Gated DeltaNet + Full Attention) GGUF implementation with quantization.
//!
//! Architecture:
//! - Hybrid model: repeats 3 layers of Gated DeltaNet (Linear Attention with SSM state)
//!   and 1 layer of Full Attention (with per-head Q/K RMSNorm and RoPE).
//! - Linear Attention:
//!   - Fused QKV projection (`attn_qkv`)
//!   - 1D Causal Convolution (`ssm_conv1d`) with circular state
//!   - Grouped Value Attention ($num\_v\_heads / num\_k\_heads = 3$)
//!   - State Space / Gated Delta Rule update: $S_t = \text{decay} \cdot S_{t-1} + k^T (\beta (v - k S_{t-1}))$
//!   - Output gating ($z$) and projection (`ssm_out`)
//! - Full Attention:
//!   - Standard Multi-Head Attention with partial RoPE (first 64 dims)
//!   - Q-norm and K-norm per head
//! - Feed-forward: SwiGLU MLP across all layers

use super::with_tracing::QMatMul;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::{Activation, Embedding};
use std::io::{Read, Seek};
use std::sync::Arc;

pub struct Gguf<'a, R: Read + Seek> {
    pub ct: &'a gguf_file::Content,
    pub reader: &'a mut R,
    pub device: Device,
}

impl<'a, R: Read + Seek> Gguf<'a, R> {
    pub fn new(ct: &'a gguf_file::Content, reader: &'a mut R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub context_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_freq_base: f64,
    pub rope_dimension_count: usize,
    pub full_attention_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
}

impl Config {
    pub fn from_metadata(metadata: &std::collections::HashMap<String, gguf_file::Value>) -> Result<Self> {
        let md_get = |s: &str| match metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let arch = md_get("general.architecture")?.to_string()?;
        if arch != "qwen35" && arch != "qwen3.5" {
            candle::bail!("expected general.architecture qwen35, got {arch}");
        }

        let block_count = md_get("qwen35.block_count")?.to_u32()? as usize;
        let nextn_layers = md_get("qwen35.nextn_predict_layers")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(0);
        let block_count = block_count.saturating_sub(nextn_layers);
        let embedding_length = md_get("qwen35.embedding_length")?.to_u32()? as usize;
        let feed_forward_length = md_get("qwen35.feed_forward_length")?.to_u32()? as usize;
        let context_length = md_get("qwen35.context_length")?.to_u32()? as usize;

        let head_count = md_get("qwen35.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("qwen35.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen35.attention.key_length")
            .or_else(|_| md_get("qwen35.attention.head_dim"))
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(256);

        let rms_norm_eps = md_get("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen35.rope.freq_base")
            .and_then(|v| v.to_f32())
            .unwrap_or(10_000_000.0) as f64;
        let rope_dimension_count = md_get("qwen35.rope.dimension_count")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let full_attention_interval = md_get("qwen35.full_attention_interval")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(4);
        let ssm_conv_kernel = md_get("qwen35.ssm.conv_kernel")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(4);
        let ssm_state_size = md_get("qwen35.ssm.state_size")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(128);
        let ssm_group_count = md_get("qwen35.ssm.group_count")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(16);
        let ssm_time_step_rank = md_get("qwen35.ssm.time_step_rank")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(48);
        let ssm_inner_size = md_get("qwen35.ssm.inner_size")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(6144);

        Ok(Self {
            block_count,
            embedding_length,
            feed_forward_length,
            context_length,
            head_count,
            head_count_kv,
            head_dim,
            rms_norm_eps,
            rope_freq_base,
            rope_dimension_count,
            full_attention_interval,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
        })
    }

    pub fn from_content(content: &gguf_file::Content) -> Result<Self> {
        Self::from_metadata(&content.metadata)
    }

    pub fn from_gguf<'a, R: Read + Seek>(gg: &Gguf<'a, R>) -> Result<Self> {
        Self::from_metadata(gg.metadata())
    }
}

#[derive(Debug, Clone)]
pub struct PartialRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl PartialRotaryEmbedding {
    pub fn new(
        dtype: DType,
        rotary_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        Ok(Self {
            sin,
            cos,
            rotary_dim,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;

        if self.rotary_dim >= head_dim {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            let q_rot = q.narrow(3, 0, self.rotary_dim)?;
            let q_pass = q.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?;
            let k_rot = k.narrow(3, 0, self.rotary_dim)?;
            let k_pass = k.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?;

            let q_rot_embed = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
            let k_rot_embed = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;

            let q_embed = Tensor::cat(&[&q_rot_embed, &q_pass], 3)?;
            let k_embed = Tensor::cat(&[&k_rot_embed, &k_pass], 3)?;
            Ok((q_embed, k_embed))
        }
    }
}

#[derive(Debug, Clone)]
pub struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
}

impl MlpWeights {
    pub fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: Activation::Silu,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_v_heads: usize,
    num_kv_groups: usize,
    num_v_groups: usize,
    head_dim: usize,
    value_head_dim: usize,
    output_dim: usize,
    rotary_emb: Arc<PartialRotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl AttentionWeights {
    pub fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &Config,
        rotary_emb: Arc<PartialRotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let projection_output_dim = |name: String| -> Result<usize> {
            let Some(info) = gg.ct.tensor_infos.get(&name) else {
                candle::bail!("cannot find tensor info for {name}");
            };
            let Some(&dim) = info.shape.dims().first() else {
                candle::bail!("tensor {name} has no dimensions");
            };
            Ok(dim)
        };
        let projection_input_dim = |name: String| -> Result<usize> {
            let Some(info) = gg.ct.tensor_infos.get(&name) else {
                candle::bail!("cannot find tensor info for {name}");
            };
            let Some(&dim) = info.shape.dims().get(1) else {
                candle::bail!("tensor {name} does not have an input dimension");
            };
            Ok(dim)
        };
        let q_projection_dim = projection_output_dim(format!("{prefix}.attn_q.weight"))?;
        let k_projection_dim = projection_output_dim(format!("{prefix}.attn_k.weight"))?;
        let v_projection_dim = projection_output_dim(format!("{prefix}.attn_v.weight"))?;
        let output_dim = projection_input_dim(format!("{prefix}.attn_output.weight"))?;
        if q_projection_dim % 2 != 0 {
            candle::bail!(
                "invalid full-attention Q/gate projection dimension for {prefix}: q_and_gate={q_projection_dim}"
            );
        }
        let query_dim = q_projection_dim / 2;
        if query_dim % head_dim != 0
            || k_projection_dim % head_dim != 0
        {
            candle::bail!(
                "invalid full-attention Q/K projection dimensions for {prefix}: q={query_dim}, k={k_projection_dim}, head_dim={head_dim}"
            );
        }
        if output_dim != query_dim {
            candle::bail!(
                "invalid full-attention output dimension for {prefix}: output={output_dim}, q={query_dim}"
            );
        }
        let num_heads = query_dim / head_dim;
        let num_kv_heads = k_projection_dim / head_dim;
        if num_heads == 0 || num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            candle::bail!(
                "invalid full-attention head counts for {prefix}: q={num_heads}, kv={num_kv_heads}"
            );
        }
        let num_kv_groups = num_heads / num_kv_heads;
        let value_head_dim = output_dim / num_heads;
        if value_head_dim == 0 || v_projection_dim % value_head_dim != 0 {
            candle::bail!(
                "invalid full-attention value projection dimensions for {prefix}: v={v_projection_dim}, output={output_dim}, q_heads={num_heads}"
            );
        }
        let num_v_heads = v_projection_dim / value_head_dim;
        if num_v_heads == 0 || num_heads % num_v_heads != 0 {
            candle::bail!(
                "invalid full-attention value head count for {prefix}: q={num_heads}, v={num_v_heads}"
            );
        }
        let num_v_groups = num_heads / num_v_heads;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), cfg.rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), cfg.rms_norm_eps)?;
        let kv_cache = ConcatKvCache::new(2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_v_heads,
            num_kv_groups,
            num_v_groups,
            head_dim,
            value_head_dim,
            output_dim,
            rotary_emb,
            kv_cache,
        })
    }

    pub fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q_and_gate = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let (q, gate) = split_query_gate(&q_and_gate, self.num_heads, self.head_dim)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_v_heads, self.value_head_dim))?
            .transpose(1, 2)?;

        // Per-head Q/K norm
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q = self.q_norm.forward(&q_flat)?.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = self.k_norm.forward(&k_flat)?.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
    let v = repeat_kv(v, self.num_v_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = mask {
            let mask = if mask.dtype() == scores.dtype() {
                mask.clone()
            } else {
                mask.to_dtype(scores.dtype())?
            };
            scores = scores.broadcast_add(&mask)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, self.output_dim))?;
        let context = (context * candle_nn::ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&context)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
pub struct LinearAttentionWeights {
    attn_qkv: QMatMul,
    attn_gate: QMatMul,
    ssm_conv1d_weights: Tensor,
    ssm_alpha: Tensor,
    ssm_beta: Tensor,
    ssm_a: Tensor,
    ssm_dt_bias: Tensor,
    ssm_norm: RmsNorm,
    ssm_out: QMatMul,
    key_dim: usize,
    value_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    conv_kernel: usize,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl LinearAttentionWeights {
    pub fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &Config,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        let num_k_heads = cfg.ssm_group_count; // 16
        let num_v_heads = cfg.ssm_time_step_rank; // 48
        let k_head_dim = cfg.ssm_state_size; // 128
        let v_head_dim = cfg.ssm_state_size; // 128
        let key_dim = num_k_heads * k_head_dim; // 2048
        let value_dim = num_v_heads * v_head_dim; // 6144
        let conv_kernel = cfg.ssm_conv_kernel; // 4

        let attn_qkv = gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?;
        let attn_gate = gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?;
        let ssm_out = gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?;

        let ssm_conv1d_weights = gg
            .tensor(&format!("{prefix}.ssm_conv1d.weight"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let ssm_alpha = gg
            .tensor(&format!("{prefix}.ssm_alpha.weight"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let ssm_beta = gg
            .tensor(&format!("{prefix}.ssm_beta.weight"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let ssm_a = gg
            .tensor(&format!("{prefix}.ssm_a"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let ssm_dt_bias = gg
            .tensor(&format!("{prefix}.ssm_dt.bias"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;

        let ssm_norm = gg.rms_norm(&format!("{prefix}.ssm_norm.weight"), cfg.rms_norm_eps)?;

        Ok(Self {
            attn_qkv,
            attn_gate,
            ssm_conv1d_weights,
            ssm_alpha,
            ssm_beta,
            ssm_a,
            ssm_dt_bias,
            ssm_norm,
            ssm_out,
            key_dim,
            value_dim,
            num_k_heads,
            num_v_heads,
            k_head_dim,
            v_head_dim,
            conv_kernel,
            conv_state: None,
            recurrent_state: None,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let total_channels = self.key_dim * 2 + self.value_dim; // 10240

        // 1. QKV projection
        let qkv = self.attn_qkv.forward(x)?; // [b, l, 10240]

        // 2. Causal 1D convolution with rolling state
        let conv_out = if l == 1 {
            // Autoregressive decoding step
            let x_step = qkv.squeeze(1)?; // [b, 10240]
            let state = match &self.conv_state {
                Some(s) => s.clone(),
                None => Tensor::zeros(
                    (b, total_channels, self.conv_kernel - 1),
                    DType::F32,
                    x.device(),
                )?,
            };

            // cat([state, x_step.unsqueeze(-1)]) -> [b, 10240, 4]
            let x_expanded = x_step.to_dtype(DType::F32)?.unsqueeze(2)?;
            let window = Tensor::cat(&[&state, &x_expanded], 2)?;

            // Roll state: keep last conv_kernel - 1 elements
            self.conv_state = Some(window.narrow(2, 1, self.conv_kernel - 1)?);

            // Dot product with conv1d weights: [10240, 4]
            let weights = self.ssm_conv1d_weights.unsqueeze(0)?.broadcast_as(window.shape())?;
            let conv_sum = (window * weights)?.sum(2)?; // [b, 10240]
            let activated = candle_nn::ops::silu(&conv_sum)?;
            activated.to_dtype(x.dtype())?.unsqueeze(1)? // [b, 1, 10240]
        } else {
            // Sequence prefill: causal 1d convolution
            let qkv_f32 = qkv.to_dtype(DType::F32)?.transpose(1, 2)?; // [b, 10240, l]
            let mut outputs = Vec::with_capacity(l);
            let mut state = Tensor::zeros(
                (b, total_channels, self.conv_kernel - 1),
                DType::F32,
                x.device(),
            )?;

            for t in 0..l {
                let x_t = qkv_f32.narrow(2, t, 1)?; // [b, 10240, 1]
                let window = Tensor::cat(&[&state, &x_t], 2)?;
                state = window.narrow(2, 1, self.conv_kernel - 1)?;
                let weights = self.ssm_conv1d_weights.unsqueeze(0)?.broadcast_as(window.shape())?;
                let conv_sum = (window * weights)?.sum(2)?; // [b, 10240]
                outputs.push(candle_nn::ops::silu(&conv_sum)?);
            }
            self.conv_state = Some(state);
            let stacked = Tensor::stack(&outputs, 1)?; // [b, l, 10240]
            stacked.to_dtype(x.dtype())?
        };

        // 3. Slice Q, K, V
        let q = conv_out.narrow(2, 0, self.key_dim)?; // [b, l, 2048]
        let k = conv_out.narrow(2, self.key_dim, self.key_dim)?; // [b, l, 2048]
        let v = conv_out.narrow(2, self.key_dim * 2, self.value_dim)?; // [b, l, 6144]

        let q = q.reshape((b, l, self.num_k_heads, self.k_head_dim))?;
        let k = k.reshape((b, l, self.num_k_heads, self.k_head_dim))?;
        let v = v.reshape((b, l, self.num_v_heads, self.v_head_dim))?;

        // L2 normalization of Q and K across head_dim
        let q_norm = (q.sqr()?.sum_keepdim(3)? + 1e-6)?.sqrt()?;
        let q = q.broadcast_div(&q_norm)?;
        let k_norm = (k.sqr()?.sum_keepdim(3)? + 1e-6)?.sqrt()?;
        let k = k.broadcast_div(&k_norm)?;

        let v_per_k = self.num_v_heads / self.num_k_heads; // 3
        let q = tile_gguf_key_heads(&q, v_per_k)?;
        let k = tile_gguf_key_heads(&k, v_per_k)?;

        // 4. Decay and Beta parameters
        let x_f32 = x.to_dtype(DType::F32)?;
        // alpha = x @ ssm_alpha^T -> [b, l, 48]
        let alpha = x_f32.matmul(&self.ssm_alpha.t()?.unsqueeze(0)?)?;
        let dt = alpha.broadcast_add(&self.ssm_dt_bias)?;
        let abs_dt = dt.abs()?;
        let neg_abs_dt = abs_dt.neg()?;
        let log1p = (neg_abs_dt.exp()? + 1.0)?.log()?;
        let dt_softplus = (dt.maximum(&Tensor::zeros_like(&dt)?)? + log1p)?;
        let g = dt_softplus.broadcast_mul(&self.ssm_a)?;
        let decay = g.exp()?; // [b, l, 48]

        // beta = sigmoid(x @ ssm_beta^T) -> [b, l, 48]
        let beta = candle_nn::ops::sigmoid(&x_f32.matmul(&self.ssm_beta.t()?.unsqueeze(0)?)?)?;

        // 5. Gated Delta Rule step on state matrix S [b, 48, 128, 128]
        let mut cur_state = match &self.recurrent_state {
            Some(s) => s.clone(),
            None => Tensor::zeros(
                (b, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                DType::F32,
                x.device(),
            )?,
        };

        let mut step_outputs = Vec::with_capacity(l);
        for t in 0..l {
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // [b, 48, 128]
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // [b, 48, 128]
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // [b, 48, 128]
            let decay_t = decay
                .narrow(1, t, 1)?
                .squeeze(1)?
                .unsqueeze(2)?
                .unsqueeze(3)?; // [b, 48, 1, 1]
            let beta_t = beta
                .narrow(1, t, 1)?
                .squeeze(1)?
                .unsqueeze(2)?; // [b, 48, 1]

            let (y_t, next_state) = gated_delta_step(
                &q_t, &k_t, &v_t, &decay_t, &beta_t, &cur_state,
            )?;
            cur_state = next_state;
            step_outputs.push(y_t.to_dtype(x.dtype())?);
        }

        self.recurrent_state = Some(cur_state);
        let y = Tensor::stack(&step_outputs, 1)?; // [b, l, 48, 128]

        // 6. Normalization with ssm_norm per head
        let y_flat = y.flatten(0, 2)?; // [b * l * 48, 128]
        let y_normed = self.ssm_norm.forward(&y_flat)?;
        let y = y_normed.reshape((b, l, self.num_v_heads, self.v_head_dim))?;

        // 7. Output gate: z = silu(attn_gate(x))
        let z = self.attn_gate.forward(x)?; // [b, l, 6144]
        let z = z.reshape((b, l, self.num_v_heads, self.v_head_dim))?;
        let z_act = candle_nn::ops::silu(&z)?;
        let y = (y * z_act)?.reshape((b, l, self.value_dim))?; // [b, l, 6144]

        // 8. Output projection
        self.ssm_out.forward(&y)
    }

    pub fn clear_state(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

fn split_query_gate(projected: &Tensor, heads: usize, head_dim: usize) -> Result<(Tensor, Tensor)> {
    let (batch, sequence, _) = projected.dims3()?;
    let projected = projected.reshape((batch, sequence, heads, 2 * head_dim))?;
    let query = projected.narrow(3, 0, head_dim)?.transpose(1, 2)?;
    let gate = projected
        .narrow(3, head_dim, head_dim)?
        .reshape((batch, sequence, heads * head_dim))?;
    Ok((query, gate))
}

fn tile_gguf_key_heads(hidden: &Tensor, repetitions: usize) -> Result<Tensor> {
    Tensor::cat(&vec![hidden; repetitions], 2)
}

fn gated_delta_step(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let state = state.broadcast_mul(decay)?;
    let predicted = key.unsqueeze(2)?.matmul(&state)?.squeeze(2)?;
    let delta = (value - predicted)?.broadcast_mul(beta)?;
    let state = (state + key.unsqueeze(3)?.matmul(&delta.unsqueeze(2)?)?)?;
    let query = (query / (query.dim(2)? as f64).sqrt())?;
    let output = query.unsqueeze(2)?.matmul(&state)?.squeeze(2)?;
    Ok((output, state))
}

#[derive(Debug, Clone)]
pub enum LayerWeights {
    FullAttention {
        ln1: RmsNorm,
        attn: AttentionWeights,
        ln2: RmsNorm,
        mlp: MlpWeights,
    },
    LinearAttention {
        ln1: RmsNorm,
        attn: LinearAttentionWeights,
        ln2: RmsNorm,
        mlp: MlpWeights,
    },
}

impl LayerWeights {
    pub fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &Config,
        rotary: Arc<PartialRotaryEmbedding>,
        device: &Device,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), cfg.rms_norm_eps)?;
        let ln2 = match gg.rms_norm(&format!("{prefix}.post_attention_norm.weight"), cfg.rms_norm_eps) {
            Ok(norm) => norm,
            Err(_) => gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), cfg.rms_norm_eps)?,
        };
        let mlp = MlpWeights::new(gg, &prefix)?;

        let is_full_attn = !gg
            .ct
            .tensor_infos
            .contains_key(&format!("{prefix}.attn_qkv.weight"));
        if is_full_attn {
            let attn = AttentionWeights::new(gg, cfg, rotary, &prefix)?;
            Ok(Self::FullAttention { ln1, attn, ln2, mlp })
        } else {
            let attn = LinearAttentionWeights::new(gg, cfg, device, &prefix)?;
            Ok(Self::LinearAttention { ln1, attn, ln2, mlp })
        }
    }

    pub fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        match self {
            Self::FullAttention { ln1, attn, ln2, mlp } => {
                let h = ln1.forward(x)?;
                let h = attn.forward(&h, mask, offset)?;
                let x = (x + h)?;
                let h2 = ln2.forward(&x)?;
                let h2 = mlp.forward(&h2)?;
                x + h2
            }
            Self::LinearAttention { ln1, attn, ln2, mlp } => {
                let h = ln1.forward(x)?;
                let h = attn.forward(&h)?;
                let x = (x + h)?;
                let h2 = ln2.forward(&x)?;
                let h2 = mlp.forward(&h2)?;
                x + h2
            }
        }
    }

    pub fn clear_state(&mut self) {
        match self {
            Self::FullAttention { attn, .. } => attn.clear_kv_cache(),
            Self::LinearAttention { attn, .. } => attn.clear_state(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LayerState {
    FullAttention(ConcatKvCache),
    LinearAttention {
        convolution: Option<Tensor>,
        recurrent: Option<Tensor>,
    },
}

impl LayerState {
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let transfer = |tensor: &Tensor| tensor.detach().to_device(device);
        Ok(match self {
            Self::FullAttention(cache) => {
                let mut cache = cache.clone();
                if let Some(key) = cache.k_mut() {
                    *key = transfer(key)?;
                }
                if let Some(value) = cache.v_mut() {
                    *value = transfer(value)?;
                }
                Self::FullAttention(cache)
            }
            Self::LinearAttention { convolution, recurrent } => Self::LinearAttention {
                convolution: convolution.as_ref().map(transfer).transpose()?,
                recurrent: recurrent.as_ref().map(transfer).transpose()?,
            },
        })
    }
}

impl LayerWeights {
    pub fn take_state(&mut self) -> LayerState {
        match self {
            Self::FullAttention { attn, .. } => LayerState::FullAttention(
                std::mem::replace(&mut attn.kv_cache, ConcatKvCache::new(2)),
            ),
            Self::LinearAttention { attn, .. } => LayerState::LinearAttention {
                convolution: attn.conv_state.take(),
                recurrent: attn.recurrent_state.take(),
            },
        }
    }

    pub fn restore_state(&mut self, state: LayerState) -> Result<()> {
        match (self, state) {
            (Self::FullAttention { attn, .. }, LayerState::FullAttention(cache)) => {
                attn.kv_cache = cache;
            }
            (Self::LinearAttention { attn, .. }, LayerState::LinearAttention { convolution, recurrent }) => {
                attn.conv_state = convolution;
                attn.recurrent_state = recurrent;
            }
            _ => candle::bail!("Qwen3.5 cache type does not match layer type"),
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub embed_tokens: Embedding,
    pub layers: Vec<LayerWeights>,
    pub norm: RmsNorm,
    pub lm_head: QMatMul,
    pub device: Device,
    pub dtype: DType,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let cfg = Config::from_gguf(&gg)?;

        let tok_embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?;
        let embed_tokens = Embedding::new(tok_embeddings, cfg.embedding_length);

        let norm = gg.rms_norm("output_norm.weight", cfg.rms_norm_eps)?;
        let lm_head = match gg.qmatmul("output.weight") {
            Ok(w) => w,
            Err(_) => gg.qmatmul("token_embd.weight")?,
        };

        let rotary = Arc::new(PartialRotaryEmbedding::new(
            DType::F32,
            cfg.rope_dimension_count,
            cfg.context_length,
            cfg.rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(cfg.block_count);
        for idx in 0..cfg.block_count {
            // Check if layer exists
            if !gg.metadata().is_empty() && gg.tensor(&format!("blk.{idx}.attn_norm.weight")).is_err() {
                break;
            }
            layers.push(LayerWeights::new(&mut gg, &cfg, rotary.clone(), device, idx)?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: DType::F32,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        let mask = if seq_len > 1 {
            Some(crate::utils::build_additive_causal_mask(
                seq_len,
                offset,
                None,
                &self.device,
                hidden.dtype(),
            )?)
        } else {
            None
        };

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, mask.as_ref(), offset)?;
        }

        let hidden = self.norm.forward(&hidden)?;
        if seq_len == 1 {
            self.lm_head.forward(&hidden.squeeze(1)?)
        } else {
            self.lm_head.forward(&hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?)
        }
    }

    pub fn clear_state(&mut self) {
        for layer in &mut self.layers {
            layer.clear_state();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};

    #[test]
    fn offloaded_state_preserves_kv_and_recurrence() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1_f32, 2., 3., 4.], (1, 1, 2, 2), &Device::Cpu)?;
        let mut cache = ConcatKvCache::new(2);
        cache.append(&tensor, &tensor)?;
        let LayerState::FullAttention(mut restored) = LayerState::FullAttention(cache).to_device(&Device::Cpu)? else {
            panic!("wrong cache type");
        };
        restored.append(&tensor, &tensor)?;
        assert_eq!(restored.current_seq_len(), 4);
        assert_eq!(restored.k().unwrap().flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 3., 4., 1., 2., 3., 4.]);
        let state = LayerState::LinearAttention {
            convolution: Some(tensor.clone()),
            recurrent: Some(tensor),
        }.to_device(&Device::Cpu)?;
        let LayerState::LinearAttention { convolution, recurrent } = state else {
            panic!("wrong cache type");
        };
        assert_eq!(convolution.unwrap().flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);
        assert_eq!(recurrent.unwrap().flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);
        Ok(())
    }

    #[test]
    fn query_gate_is_split_within_each_head() -> Result<()> {
        let projected = Tensor::from_vec(
            vec![1_f32, 2., 10., 20., 3., 4., 30., 40.],
            (1, 1, 8),
            &Device::Cpu,
        )?;
        let (query, gate) = split_query_gate(&projected, 2, 2)?;
        assert_eq!(query.dims(), &[1, 2, 1, 2]);
        assert_eq!(query.flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);
        assert_eq!(gate.flatten_all()?.to_vec1::<f32>()?, vec![10., 20., 30., 40.]);
        Ok(())
    }

    #[test]
    fn gguf_value_heads_use_tiled_key_order() -> Result<()> {
        let hidden = Tensor::from_vec(vec![1_f32, 2.], (1, 1, 2, 1), &Device::Cpu)?;
        let tiled = tile_gguf_key_heads(&hidden, 3)?;
        assert_eq!(tiled.dims(), &[1, 1, 6, 1]);
        assert_eq!(tiled.flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 1., 2., 1., 2.]);
        Ok(())
    }

    #[test]
    fn gated_delta_decays_before_prediction_and_scales_query() -> Result<()> {
        let device = Device::Cpu;
        let query = Tensor::from_vec(vec![2_f32, 0.], (1, 1, 2), &device)?;
        let key = Tensor::from_vec(vec![1_f32, 0.], (1, 1, 2), &device)?;
        let value = Tensor::from_vec(vec![10_f32, 20.], (1, 1, 2), &device)?;
        let state = Tensor::from_vec(vec![2_f32, 4., 6., 8.], (1, 1, 2, 2), &device)?;
        let decay = Tensor::full(0.5_f32, (1, 1, 1, 1), &device)?;
        let beta = Tensor::full(0.25_f32, (1, 1, 1), &device)?;
        let (output, state) = gated_delta_step(&query, &key, &value, &decay, &beta, &state)?;
        assert_eq!(state.flatten_all()?.to_vec1::<f32>()?, vec![3.25, 6.5, 3., 4.]);
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        for (actual, expected) in output.iter().zip([3.25 * 2_f32.sqrt(), 6.5 * 2_f32.sqrt()]) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn test_partial_rotary_embedding() -> Result<()> {
        let dev = Device::Cpu;
        let rotary = PartialRotaryEmbedding::new(DType::F32, 64, 512, 10000.0, &dev)?;
        let q = Tensor::zeros((1, 4, 1, 256), DType::F32, &dev)?;
        let k = Tensor::zeros((1, 4, 1, 256), DType::F32, &dev)?;
        let (q_rot, k_rot) = rotary.apply(&q, &k, 0)?;
        assert_eq!(q_rot.dims4()?, (1, 4, 1, 256));
        assert_eq!(k_rot.dims4()?, (1, 4, 1, 256));
        Ok(())
    }
}
