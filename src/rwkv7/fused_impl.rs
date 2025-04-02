use burn::nn::{Linear, Sigmoid, Tanh};
use burn::prelude::{Backend, Shape, Tensor};
use burn::tensor::{DType, TensorMetadata, TensorPrimitive};
use burn::tensor::activation::{relu, sigmoid, softplus};
use burn::tensor::ops::{FloatTensor};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{CubeCount, CubeDim};
use crate::rwkv7::{fused_impl_kernel, Block, ChannelMixer, LayerState, RWKV7Model, TimeMixer};
use crate::rwkv7::fused_impl_kernel::TimeMixKernelConfig;

fn matmul_vec_mat<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, 2>) -> Tensor<B, D> {

    #[cfg(feature = "fusion")]
    let use_fused = true;
    #[cfg(not(feature = "fusion"))]
    let use_fused = false;

    let original_a_dims = a.dims();
    let original_b_dims = b.dims();

    let mut intermediate_tensor_size = original_b_dims[0]*original_b_dims[1];
    for i in 0..D-1 {
        intermediate_tensor_size *= original_a_dims[i];
    }

    //let use_fused = use_fused && intermediate_tensor_size < 1024*1024*64;

    if false {
        assert_eq!(original_a_dims[D-1], original_b_dims[0]);
        let a = a.reshape([-1, original_a_dims[D-1] as i32, 1]);
        let b = b.unsqueeze::<3>();
        let x = (a*b).sum_dim(1);
        let mut new_dims = original_a_dims;
        new_dims[D - 1] = original_b_dims[1];
        x.reshape(new_dims)
    } else {
        a.matmul(b.unsqueeze())
    }
}

fn linear_forward<B: Backend, const D: usize>(l: &Linear<B>, x: Tensor<B, D>) -> Tensor<B, D> {
    assert!(l.bias.is_none());
    matmul_vec_mat(x, l.weight.val())
}

fn lerp<B: Backend, const D: usize>(start: Tensor<B, D>, end: Tensor<B, D>, weight: Tensor<B, D>) -> Tensor<B, D> {
    start.clone() + weight * ( end - start)
}

fn lora_forward<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x = matmul_vec_mat(x, l1);
    let x = matmul_vec_mat(x, l2);
    if let Some(base) = base {
        x + base
    } else {
        x
    }
}

fn lora_forward_sigmoid<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x = matmul_vec_mat(x, l1);
    let activation = Sigmoid::new();
    let x = matmul_vec_mat(activation.forward(x), l2);
    if let Some(base) = base {
        x + base
    } else {
        x
    }
}

fn lora_forward_tanh<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x = matmul_vec_mat(x, l1);
    let activation = Tanh::new();
    let x = matmul_vec_mat(activation.forward(x), l2);
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

pub struct TimeMixOutput<B: Backend> {
    pub state: Tensor<B, 4>,
    pub y: Tensor<B, 3>,
}

pub struct TimeMixOutputInner<B: Backend> {
    pub state: FloatTensor<B>,
    pub y: FloatTensor<B>,
}

pub trait RWKVFusedBackend: burn::tensor::backend::Backend {
    fn fused_time_mix_forward_inner(
        state_in: FloatTensor<Self>,
        r: FloatTensor<Self>,
        w: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        a: FloatTensor<Self>,
        b: FloatTensor<Self>,
    ) -> TimeMixOutputInner<Self>;
}

fn fused_time_mix_forward<B: RWKVFusedBackend>(
    state_in: Tensor<B, 4>,
    r: Tensor<B, 3>,
    w: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
) -> TimeMixOutput<B> {

    let d_batch = state_in.shape().dims[0];
    let n_heads = state_in.shape().dims[1];
    let d_value = state_in.shape().dims[2];
    let d_key = state_in.shape().dims[3];

    assert_eq!(d_key, d_value);
    assert_eq!(state_in.dtype(), DType::F32);

    let d_tokens = r.shape().dims[1];

    assert_eq!(r.dims()[0], d_batch);
    assert_eq!(r.dims()[1], d_tokens);
    assert_eq!(r.dims()[2], n_heads*d_value);

    assert_eq!(w.dims()[0], d_batch);
    assert_eq!(w.dims()[1], d_tokens);
    assert_eq!(w.dims()[2], n_heads*d_value);
    assert_eq!(w.dtype(), DType::F32);

    assert_eq!(k.dims()[0], d_batch);
    assert_eq!(k.dims()[1], d_tokens);
    assert_eq!(k.dims()[2], n_heads*d_value);

    assert_eq!(v.dims()[0], d_batch);
    assert_eq!(v.dims()[1], d_tokens);
    assert_eq!(v.dims()[2], n_heads*d_value);

    assert_eq!(a.dims()[0], d_batch);
    assert_eq!(a.dims()[1], d_tokens);
    assert_eq!(a.dims()[2], n_heads*d_value);

    assert_eq!(b.dims()[0], d_batch);
    assert_eq!(b.dims()[1], d_tokens);
    assert_eq!(b.dims()[2], n_heads*d_value);

    let state_in = state_in.into_primitive().tensor();
    let r = r.into_primitive().tensor();
    let w = w.into_primitive().tensor();
    let k = k.into_primitive().tensor();
    let v = v.into_primitive().tensor();
    let a = a.into_primitive().tensor();
    let b = b.into_primitive().tensor();

    let TimeMixOutputInner{state, y} = B::fused_time_mix_forward_inner(state_in, r, w, k, v, a, b);


    let state_out = Tensor::from_primitive(TensorPrimitive::Float(state));
    let y_out = Tensor::from_primitive(TensorPrimitive::Float(y));

    assert_eq!(state_out.dims()[0], d_batch);
    assert_eq!(state_out.dims()[1], n_heads);
    assert_eq!(state_out.dims()[2], d_value);
    assert_eq!(state_out.dims()[3], d_key);
    assert_eq!(state_out.dtype(), DType::F32);


    assert_eq!(y_out.dims()[0], d_batch);
    assert_eq!(y_out.dims()[1], d_tokens);
    assert_eq!(y_out.dims()[2], n_heads*d_value);

    TimeMixOutput {state: state_out, y: y_out}
}

/// Implement our custom backend trait for the generic `CubeBackend`.
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> RWKVFusedBackend
for CubeBackend<R, F, I, BT>
{
    fn fused_time_mix_forward_inner(
        state_in: FloatTensor<Self>,
        r: FloatTensor<Self>,
        w: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        a: FloatTensor<Self>,
        b: FloatTensor<Self>,
    ) -> TimeMixOutputInner<Self> {

        let _d_batch = state_in.shape().dims[0];
        let _n_heads = state_in.shape().dims[1];
        let d_value = state_in.shape().dims[2];
        let d_key = state_in.shape().dims[3];

        assert_eq!(d_key, d_value);
        assert_eq!(state_in.dtype(), DType::F32);

        assert_eq!(r.dtype(), F::dtype());

        assert_eq!(w.dtype(), DType::F32);

        assert_eq!(k.dtype(), F::dtype());

        assert_eq!(v.dtype(), F::dtype());

        assert_eq!(a.dtype(), F::dtype());

        assert_eq!(b.dtype(), F::dtype());


        let d_batch = state_in.shape().dims[0];
        let n_heads = state_in.shape().dims[1];
        let d_value = state_in.shape().dims[2];
        let d_key = state_in.shape().dims[3];

        let d_tokens = r.shape().dims[1];

        let cube_dim = CubeDim { x: d_key as u32, y: 1, z: 1 };

        let state_in = into_contiguous(state_in);
        let r = into_contiguous(r);
        let w = into_contiguous(w);
        let k = into_contiguous(k);
        let v = into_contiguous(v);
        let a = into_contiguous(a);
        let b = into_contiguous(b);

        let client = state_in.client.clone();
        let device = state_in.device.clone();

        let state_out = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            state_in.shape.clone(),
            state_in.client.empty(state_in.shape.num_elements() * core::mem::size_of::<f32>()),
            DType::F32
        );

        let y_out_shape: Shape = [d_batch, d_tokens, n_heads*d_value].into();
        let y_out = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            y_out_shape.clone(),
            client.empty(y_out_shape.num_elements() * core::mem::size_of::<F>()),
            F::dtype()
        );

        let config = TimeMixKernelConfig{
            n_heads: n_heads as u32,
            d_key_value: d_key as u32
        };

        let cube_count = CubeCount::Static(
            n_heads as u32, d_batch as u32, 1
        );
        
        fused_impl_kernel::fused_time_mix_forward::launch::<F, R>(
            &client,
            cube_count,
            cube_dim,
            state_in.as_tensor_arg::<f32>(1),
            state_out.as_tensor_arg::<f32>(1),
            r.as_tensor_arg::<F>(1),
            w.as_tensor_arg::<f32>(1),
            k.as_tensor_arg::<F>(1),
            v.as_tensor_arg::<F>(1),
            a.as_tensor_arg::<F>(1),
            b.as_tensor_arg::<F>(1),
            y_out.as_tensor_arg::<F>(1),
            config
        );

        assert_eq!(y_out.dtype(), F::dtype());

        TimeMixOutputInner {state: state_out, y: y_out}
    }
}


#[cfg(feature = "fusion")]
use burn_fusion::{Fusion, FusionBackend, FusionRuntime};
use burn_fusion::client::FusionClient;
use burn_fusion::stream::Operation;
use burn_ir::{CustomOpIr, HandleContainer, OperationIr};

#[cfg(feature = "fusion")]
impl<B: FusionBackend + RWKVFusedBackend> RWKVFusedBackend for Fusion<B>
{
    fn fused_time_mix_forward_inner(
        state_in: FloatTensor<Self>,
        r: FloatTensor<Self>,
        w: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        a: FloatTensor<Self>,
        b: FloatTensor<Self>,
    ) -> TimeMixOutputInner<Self>
    {
        let d_batch = state_in.shape().dims[0];
        let n_heads = state_in.shape().dims[1];
        let d_value = state_in.shape().dims[2];
        let _d_key = state_in.shape().dims[3];
        let d_tokens = r.shape().dims[1];

        let stream = state_in.stream.clone();
        let client = state_in.client.clone();

        struct InnerStuff<B> {
            desc: CustomOpIr,
            _b: core::marker::PhantomData<B>
        }

        impl<B1: FusionBackend + RWKVFusedBackend> Operation<B1::FusionRuntime> for InnerStuff<B1> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<<B1::FusionRuntime as FusionRuntime>::FusionHandle>) {
                let ([state_in, r, w, k, v, a, b], [state_out, y_out]) = self.desc.consume();
                let state_in = handles.get_float_tensor::<B1>(&state_in);
                let r = handles.get_float_tensor::<B1>(&r);
                let w = handles.get_float_tensor::<B1>(&w);
                let k = handles.get_float_tensor::<B1>(&k);
                let v = handles.get_float_tensor::<B1>(&v);
                let a = handles.get_float_tensor::<B1>(&a);
                let b = handles.get_float_tensor::<B1>(&b);
                let inner_output = B1::fused_time_mix_forward_inner(state_in, r, w, k, v, a, b);
                handles.register_float_tensor::<B1>(&state_out.id, inner_output.state);
                handles.register_float_tensor::<B1>(&y_out.id, inner_output.y);
            }
        }

        let state_in_shape = state_in.shape.clone();
        let r_dtype = r.dtype();

        let inputs = [
            state_in.into_ir(),
            r.into_ir(),
            w.into_ir(),
            k.into_ir(),
            v.into_ir(),
            a.into_ir(),
            b.into_ir()
        ];

        let state_out = client.tensor_uninitialized(state_in_shape, DType::F32);
        let y_out = client.tensor_uninitialized(vec![d_batch, d_tokens, n_heads*d_value], r_dtype);

        let desc = CustomOpIr::new("fused_time_mix_forward", &inputs, &[state_out.to_ir_out(), y_out.to_ir_out()]);
        client.register(
            vec![stream],
            OperationIr::Custom(desc.clone()),
            InnerStuff::<B> {
                desc,
                _b: Default::default(),
            }
        );

        TimeMixOutputInner{state: state_out, y: y_out}
    }
}

impl<B: RWKVFusedBackend> Block<B> {
    pub(crate) fn fused_forward(&self, _layer_num: usize, x: Tensor<B, 3>, v0: Option<Tensor<B, 3>>, s: Option<&LayerState<B>>, d_model: usize, n_heads: usize) -> (Tensor<B, 3>, Tensor<B, 3>, LayerState<B>) {
        let num_tokens = x.dims()[1];

        let x = if let Some(ln0) = &self.ln0 {
            ln0.forward(x)
        } else {
            x
        };

        let s: LayerState<B> = if let Some(s) = s {
            s.clone()
        } else {
            LayerState::new_from_input(x.shape().dims[0], d_model, n_heads, &x.device())
        };
        let x1 = self.ln1.forward(x.clone());
        let new_time_mixer_x_state: Tensor<B, 2> = if num_tokens > 1 {
            x1.clone().slice([None, Some((-2, -1))])
        } else {
            x1.clone()
        }.squeeze(1);
        let (x2, new_v0, new_vk_state) = self.att.fused_forward(x1.clone(), v0, s.time_mixer_x_state, s.vk_state, d_model, n_heads);
        let x3 = x + x2;
        let x4 = self.ln2.forward(x3.clone());
        let new_channel_mixer_x_state: Tensor<B, 2> = if num_tokens > 1 {
            x4.clone().slice([None, Some((-2, -1))])
        } else {
            x4.clone()
        }.squeeze(1);
        let x5 = self.ffn.fused_forward(x4.clone(), s.channel_mixer_x_state);

        let x6 = x5 + x3;

        let new_s = LayerState {
            time_mixer_x_state: new_time_mixer_x_state,
            vk_state: new_vk_state,
            channel_mixer_x_state: new_channel_mixer_x_state
        };
        (x6, new_v0, new_s)
    }
}

impl<B: RWKVFusedBackend> ChannelMixer<B> {
    pub(crate) fn fused_forward(&self, hidden_state_in: Tensor<B, 3>, x_state: Tensor<B, 2>) -> Tensor<B, 3> {
        //let d_batch = hidden_state_in.shape().dims[0];
        let d_tokens = hidden_state_in.shape().dims[1];
        let x_shifted_one_to_the_past : Tensor<B, 3> = if d_tokens > 1 {
            Tensor::cat(vec![x_state.unsqueeze(), hidden_state_in.clone().slice([None, Some((0, -1))])], 1)
        } else {
            x_state.unsqueeze()
        };

        let x_in = lerp(hidden_state_in, x_shifted_one_to_the_past, self.x_k.val());
        //let hidden = self.key.forward(x_in);
        let hidden = linear_forward(&self.key, x_in);
        let hidden = relu(hidden).powi_scalar(2);
        //let out = self.value.forward(hidden);
        let out = linear_forward(&self.value, hidden);
        out
    }
}

impl <B: RWKVFusedBackend> TimeMixer<B> {
    pub(crate) fn fused_forward(&self, hidden_state_in: Tensor<B, 3>, v0: Option<Tensor<B, 3>>, time_mixer_x_state: Tensor<B, 2>, vk_state: Tensor<B, 4>, d_model: usize, n_heads: usize) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 4>) {
        let d_batch = hidden_state_in.shape().dims[0];
        let d_tokens = hidden_state_in.shape().dims[1];
        let d_k = d_model/ n_heads;
        let d_v = d_k;
        assert_eq!(hidden_state_in.shape().dims[2], d_model);

        assert_eq!(time_mixer_x_state.shape().dims[0], d_batch);
        assert_eq!(time_mixer_x_state.shape().dims[1], d_model);

        assert_eq!(vk_state.shape().dims[0], d_batch);
        assert_eq!(vk_state.shape().dims[1], n_heads);
        assert_eq!(vk_state.shape().dims[2], d_v);
        assert_eq!(vk_state.shape().dims[3], d_k);

        let x_shifted_one_to_the_past : Tensor<B, 3> = if d_tokens > 1 {
            Tensor::cat(vec![time_mixer_x_state.unsqueeze(), hidden_state_in.clone().slice([None, Some((0, -1))])], 1)
        } else {
            time_mixer_x_state.unsqueeze()
        };

        let x_receptance = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_r.val());
        let x_decay = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_w.val());
        let x_key  = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_k.val());
        let x_value  = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_v.val());
        let x_iclr = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_a.val());
        let x_gate = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_g.val());

        //let r = self.receptance.forward(x_receptance);
        //let k = self.key.forward(x_key);
        //let v = self.value.forward(x_value.clone());
        let r = linear_forward(&self.receptance, x_receptance);
        let k = linear_forward(&self.key, x_key);
        let v = linear_forward(&self.value, x_value.clone());

        let (v0, v) = if let Some(v0) = v0 {
            let v0_mix = lora_forward(self.v1.clone().unwrap().val(), self.v2.clone().unwrap().val(), Some(self.v0.clone().unwrap().val()), x_value);
            (v0.clone(), lerp(v, v0, Sigmoid::new().forward(v0_mix)))
        } else {
            (v.clone(), v)
        };
        
        let gate = lora_forward_sigmoid(self.g1.val(), self.g2.val(), None, x_gate);

        let log_neglog_of_decay = lora_forward_tanh(self.w1.val(), self.w2.val(), Some(self.w0.val()), x_decay).cast(DType::F32);
        let log_neglog_of_decay = (- softplus(-log_neglog_of_decay, 1.0)).add_scalar(-0.5);

        let deformed_key = k.clone()*self.k_k.val();
        let deformed_key = normalize(deformed_key.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]), 3, 2.0).reshape([d_batch as i32, d_tokens as i32, -1]);

        let iclr = sigmoid(lora_forward(self.a1.val(), self.a2.val(), Some(self.a0.val()), x_iclr));

        let k = lerp(k.clone(), k.clone()*iclr.clone(), self.k_a.val());


        let TimeMixOutput{state, y} = fused_time_mix_forward(vk_state, r.clone(), log_neglog_of_decay, k.clone(), v.clone(), deformed_key, iclr);

        // apply group normalization to each head and recombine the heads
        let out = self.ln_x.forward(y.reshape([(d_batch*d_tokens) as i32, d_model as i32])).reshape([d_batch as i32, d_tokens as i32, d_model as i32]);

        let r: Tensor<B, 4> = r.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let k: Tensor<B, 4> = k.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let v: Tensor<B, 4> = v.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);

        //println!("b: {out:}");
        let temp = r*k*self.r_k.val().unsqueeze();

        let bonus = temp.sum_dim(3) * v;

        //println!("bonus: {bonus:?}");
        // Recombine bonus heads
        let bonus: Tensor<B, 3> = bonus.reshape([d_batch as i32, d_tokens as i32, d_model as i32]);
        let out = out + bonus;

        //println!("c: {out:?}");

        // Apply output gate
        let out = out*gate;

        //println!("d: {out:?}");

        // Project the output
        //let out = self.output.forward(out);
        let out = linear_forward(&self.output, out);

        (out, v0, state)
    }

}


impl <B: RWKVFusedBackend> RWKV7Model<B> {
    pub fn fused_forward(&self, input: Tensor<B, 2, burn::prelude::Int>, channel_states: Option<&[LayerState<B>]>) -> (Tensor<B, 3>, Vec<LayerState<B>>) {
        let mut x = self.emb.forward(input);

        let mut v0 = None;
        let mut new_channel_states = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let channel_state = if let Some(s) = channel_states{
                Some(&s[i])
            } else {
                None
            };
            let (new_x, new_v0, new_channel_state) = block.fused_forward(i, x, v0, channel_state, self.d_model, self.n_heads);
            x = new_x;

            v0 = Some(new_v0);
            new_channel_states.push(new_channel_state);
        }

        let x = self.ln_out.forward(x);

        let logits = linear_forward(&self.head, x);

        (logits, new_channel_states)
    }
}