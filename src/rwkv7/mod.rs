mod burn_impl;
mod fused_impl;
mod fused_impl_kernel;

use burn::module::{Param};
use burn::nn::{Linear, LinearConfig, Initializer, LayerNorm, LayerNormConfig, GroupNormConfig, GroupNorm, Tanh, Sigmoid};
use burn::prelude::{Tensor, Backend};
use burn::tensor::{Int};
use burn::tensor::module::embedding;
pub use super::{RWKVForward};

#[derive(burn::prelude::Module, Debug)]
struct Embedding<B: Backend> {
    weight: Param<Tensor<B, 2>>
}

impl<B: Backend> Embedding<B> {
    pub fn new(config: &RWKV7Config, device: &B::Device) -> Self {
        let weight = Initializer::Zeros.init([config.vocab_size, config.d_model], device);
        Self { weight }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        embedding(self.weight.val(), input)
    }
}

#[derive(burn::prelude::Module, Debug)]
struct TimeMixer<B: Backend> {
    x_r: Param<Tensor<B, 3>>,
    x_w: Param<Tensor<B, 3>>,
    x_k: Param<Tensor<B, 3>>,
    x_v: Param<Tensor<B, 3>>,
    x_a: Param<Tensor<B, 3>>,
    x_g: Param<Tensor<B, 3>>,
    w0: Param<Tensor<B, 3>>,
    r_k: Param<Tensor<B, 2>>,
    w1: Param<Tensor<B, 2>>,
    w2: Param<Tensor<B, 2>>,
    a1: Param<Tensor<B, 2>>,
    a2: Param<Tensor<B, 2>>,
    a0: Param<Tensor<B, 3>>,
    g1: Param<Tensor<B, 2>>,
    g2: Param<Tensor<B, 2>>,
    k_k: Param<Tensor<B, 3>>,
    k_a: Param<Tensor<B, 3>>,
    receptance: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    v2: Option<Param<Tensor<B, 2>>>,
    v1: Option<Param<Tensor<B, 2>>>,
    v0: Option<Param<Tensor<B, 3>>>,
    ln_x: GroupNorm<B>,
}

fn linear<B: Backend>(d_input: usize, d_output: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(d_input, d_output).with_bias(false).init(device)
}

impl <B: Backend> TimeMixer<B> {


    fn new(layer_num: usize, config: &RWKV7Config, device: &B::Device) -> Self {

        struct LoraRanks {
            decay_lora: usize,
            iclr_lora: usize,
            v0_mix_amt_lora: usize,
            gate_lora: usize
        }
        let lora_ranks = if config.d_model < 2048 {
            LoraRanks {
                decay_lora: 64,
                iclr_lora: 64,
                v0_mix_amt_lora: 32,
                gate_lora: 128
            }
        } else if config.d_model < 4096 {
            LoraRanks {
                decay_lora: 128,
                iclr_lora: 64,
                v0_mix_amt_lora: 64,
                gate_lora: 256
            }
        } else if config.d_model < 6144 {
            LoraRanks {
                decay_lora: 192,
                iclr_lora: 96,
                v0_mix_amt_lora: 96,
                gate_lora: 384
            }
        } else {
            LoraRanks {
                decay_lora: 256,
                iclr_lora: 128,
                v0_mix_amt_lora: 128,
                gate_lora: 512
            }
        };
        let (v0, v1, v2) = if layer_num > 0 {
            (
                Some(Initializer::Zeros.init([1, 1, config.d_model], device)),
                Some(Initializer::Zeros.init([config.d_model, lora_ranks.v0_mix_amt_lora], device)),
                Some(Initializer::Zeros.init([lora_ranks.v0_mix_amt_lora, config.d_model], device))
            )
        } else {
            (None, None, None)
        };
        let ln_x = GroupNormConfig::new(config.n_heads, config.d_model).with_epsilon(1e-5*(config.d_model/config.n_heads) as f64).init(device);
        Self {
            x_r: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_w: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_k: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_v: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_a: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_g: Initializer::Zeros.init([1, 1, config.d_model], device),
            w0: Initializer::Zeros.init([1, 1, config.d_model], device),
            r_k: Initializer::Zeros.init([config.n_heads, config.d_model/config.n_heads], device),
            w1: Initializer::Zeros.init([config.d_model, lora_ranks.decay_lora], device),
            w2: Initializer::Zeros.init([lora_ranks.decay_lora, config.d_model], device),
            a1: Initializer::Zeros.init([config.d_model, lora_ranks.iclr_lora], device),
            a2: Initializer::Zeros.init([lora_ranks.iclr_lora, config.d_model], device),
            a0: Initializer::Zeros.init([1, 1, config.d_model], device),
            g1: Initializer::Zeros.init([config.d_model, lora_ranks.gate_lora], device),
            g2: Initializer::Zeros.init([lora_ranks.gate_lora, config.d_model], device),
            k_k: Initializer::Zeros.init([1, 1, config.d_model], device),
            k_a: Initializer::Zeros.init([1, 1, config.d_model], device),
            receptance: linear(config.d_model, config.d_model, device),
            key: linear(config.d_model, config.d_model, device),
            value: linear(config.d_model, config.d_model, device),
            output: linear(config.d_model, config.d_model, device),
            v2,
            v1,
            v0,
            ln_x,
        }
    }
    
}

#[derive(burn::prelude::Module, Debug)]
struct ChannelMixer<B: Backend> {
    x_k: Param<Tensor<B, 3>>,
    key: Linear<B>,
    value: Linear<B>,
}

impl<B: Backend> ChannelMixer<B> {
    fn new(config: &RWKV7Config, device: &B::Device) -> Self {
        Self {
            x_k: Initializer::Zeros.init([1, 1, config.d_model], device),
            key: linear(config.d_model, config.d_model*4, device),
            value: linear(config.d_model*4, config.d_model, device),
        }
    }
}

#[derive(burn::prelude::Module, Debug)]
pub struct Block<B: Backend> {
    pub ln0: Option<LayerNorm<B>>,
    ln1: LayerNorm<B>,
    att: TimeMixer<B>,
    ln2: LayerNorm<B>,
    ffn: ChannelMixer<B>
}

impl<B: Backend> Block<B> {
    fn new( layer_num: usize, config: &RWKV7Config, device: &B::Device) -> Self {
        let ln_config = LayerNormConfig::new(config.d_model);
        let ln0 = if layer_num == 0 {
            Some(ln_config.init(device))
        } else {
            None
        };
        Block {
            ln0,
            ln1: ln_config.init(device),
            att: TimeMixer::new(layer_num, config, device),
            ln2: ln_config.init(device),
            ffn: ChannelMixer::new(&config, device)
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerState<B: Backend> {
    time_mixer_x_state: Tensor<B, 2>,
    vk_state: Tensor<B, 4>,
    channel_mixer_x_state: Tensor<B, 2>
}

impl<B: Backend> LayerState<B> {
    pub fn new_from_input(batch: usize, d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        let k = d_model/n_heads;
        let v = k;
        let time_mixer_x_state = Tensor::zeros([batch, d_model], device);
        let vk_state = Tensor::zeros([batch, n_heads, v, k], device);
        let channel_mixer_x_state = Tensor::zeros([batch, d_model], device);
        LayerState {
            time_mixer_x_state,
            vk_state,
            channel_mixer_x_state
        }
    }

}

impl<B: Backend> RWKVLayerState for LayerState<B> {
    fn detach(self) -> Self {
        LayerState {
            channel_mixer_x_state: self.channel_mixer_x_state.detach(),
            time_mixer_x_state: self.time_mixer_x_state.detach(),
            vk_state: self.vk_state.detach()
        }
    }
}

#[derive(Debug)]
pub struct RWKV7Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

impl RWKV7Config {
    pub fn rwkv_7_0b1 () -> Self {
        RWKV7Config {
            vocab_size: 65536,
            d_model: 768,
            n_layers: 12,
            n_heads: 12,
        }
    }

    pub fn rwkv_7_1b5 () -> Self {
        RWKV7Config {
            vocab_size: 65536,
            d_model: 2048,
            n_layers: 24,
            n_heads: 32,
        }
    }

    pub fn rwkv_7_2b9 () -> Self {
        RWKV7Config {
            vocab_size: 65536,
            d_model: 2560,
            n_layers: 32,
            n_heads: 40,
        }
    }

    pub fn from_record<B: Backend>(record: &RWKV7ModelRecord<B>) -> Self {
        let n_layers = record.blocks.len();
        let s: [usize; 2] = record.emb.weight.shape().dims();
        let vocab_size = s[0];
        let d_model = s[1];
        let s: [usize; 2] = record.blocks[0].att.r_k.shape().dims();
        let n_heads = s[0];
        RWKV7Config {
            vocab_size,
            d_model,
            n_layers,
            n_heads,
        }
    }
}

#[derive(burn::prelude::Module, Debug)]
pub struct RWKV7Model<B: Backend> {
    emb: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    ln_out: LayerNorm<B>,
    head: Linear<B>,
    n_heads: usize,
    d_model: usize,
}

impl<B: Backend> RWKV7Model<B> {
    pub fn new(config: RWKV7Config, device: &B::Device) -> Self{
        let mut blocks = Vec::new();
        for i in 0..config.n_layers {
            blocks.push(Block::new(i, &config, device))
        }
        let ln_out_config = LayerNormConfig::new(config.d_model);
        let head_config = LinearConfig::new(config.d_model, config.vocab_size).with_bias(false);
        Self {
            emb: Embedding::new(&config, device),
            blocks,
            ln_out: ln_out_config.init(device),
            head: head_config.init(device),
            n_heads: config.n_heads,
            d_model: config.d_model
        }
    }


}

#[cfg(feature = "ndarray")]
use burn::backend::NdArray;

#[cfg(feature = "ndarray")]
impl RWKVForward<NdArray> for RWKV7Model<NdArray> {
    type LayerState = LayerState<NdArray>;
    
    fn forward(&self, input: Tensor<NdArray, 2, Int>, channel_states: Option<&[Self::LayerState]>) -> (Tensor<NdArray, 3>, Vec<Self::LayerState>) {
        self.unfused_forward(input, channel_states)
    }
}

impl <B: RWKVFusedBackend>  RWKVForward<B> for RWKV7Model<B> {
    type LayerState = LayerState<B>;
    
    fn forward(&self, input: Tensor<B, 2, Int>, channel_states: Option<&[LayerState<B>]>) -> (Tensor<B, 3>, Vec<LayerState<B>>) {
        self.fused_forward(input, channel_states)
    }
}

#[cfg(feature = "fusion")]
use burn_fusion::{Fusion, FusionBackend};

#[cfg(feature = "fusion")]
impl <B: Backend>  RWKVForward<Fusion<B>> for RWKV7Model<Fusion<B>>
where
    Fusion<B>: Backend,
    B: FusionBackend
{
    type LayerState = LayerState<Fusion<B>>;
    
    fn forward(&self, input: Tensor<Fusion<B>, 2, Int>, channel_states: Option<&[Self::LayerState]>) -> (Tensor<Fusion<B>, 3>, Vec<Self::LayerState>) {
        self.unfused_forward(input, channel_states)
    }
}

use burn_autodiff::{Autodiff};
use crate::rwkv7::fused_impl::RWKVFusedBackend;
use crate::RWKVLayerState;

impl <B: Backend>  RWKVForward<Autodiff<B>> for RWKV7Model<Autodiff<B>>
where
    Autodiff<B>: Backend
{
    type LayerState = LayerState<Autodiff<B>>;
    
    fn forward(&self, input: Tensor<Autodiff<B>, 2, Int>, channel_states: Option<&[LayerState<Autodiff<B>>]>) -> (Tensor<Autodiff<B>, 3>, Vec<LayerState<Autodiff<B>>>) {
        self.unfused_forward(input, channel_states)
    }
}

fn lerp<B: Backend, const D: usize>(start: Tensor<B, D>, end: Tensor<B, D>, weight: Tensor<B, D>) -> Tensor<B, D> {
    start.clone() + weight * ( end - start)
}

fn lora_forward<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x1 = x.matmul(l1.unsqueeze());
    let x = x1.matmul(l2.unsqueeze());
    if let Some(base) = base {
        x + base
    } else {
        x
    }
}

fn lora_forward_sigmoid<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x = x.matmul(l1.unsqueeze());
    let activation = Sigmoid::new();
    let x = activation.forward(x).matmul(l2.unsqueeze());
    if let Some(base) = base {
        x + base
    } else {
        x
    }
}

fn lora_forward_tanh<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x = x.matmul(l1.unsqueeze());
    let activation = Tanh::new();
    let x = activation.forward(x).matmul(l2.unsqueeze());
    if let Some(base) = base {
        x + base
    } else {
        x
    }
}

fn inner_norm<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize, p: f32) -> Tensor<B, D> {
    x.abs().powf_scalar(p).sum_dim(dim).powf_scalar(1./p)
}

fn normalize<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize, p: f32) -> Tensor<B, D> {
    // In python:
    /*
     eps = 1e-12
     denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
     return input / denom
     */
    let denom = inner_norm(x.clone(), dim, p).clamp_min(1e-12);
    x / denom
}