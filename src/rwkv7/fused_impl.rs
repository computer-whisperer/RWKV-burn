use burn::nn::Sigmoid;
use burn::prelude::{Backend, Shape, Tensor};
use burn::tensor::{DType, TensorPrimitive};
use burn::tensor::activation::{sigmoid, softplus};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{CubeCount, CubeDim};
use crate::rwkv7::{fused_impl_kernel, lerp, lora_forward, lora_forward_sigmoid, lora_forward_tanh, normalize, Block, LayerState, RWKV7Model, TimeMixer};
use crate::rwkv7::fused_impl_kernel::TimeMixKernelConfig;

pub struct TimeMixOutput<B: Backend> {
    pub state: Tensor<B, 4>,
    pub y: Tensor<B, 3>,
}

pub trait RWKVFusedBackend: burn::tensor::backend::Backend {
    fn fused_time_mix_forward(
        state_in: Tensor<Self, 4>,
        r: Tensor<Self, 3>,
        w: Tensor<Self, 3>,
        k: Tensor<Self, 3>,
        v: Tensor<Self, 3>,
        a: Tensor<Self, 3>,
        b: Tensor<Self, 3>,
    ) -> TimeMixOutput<Self>;
}

/// Implement our custom backend trait for the generic `CubeBackend`.
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> RWKVFusedBackend
for CubeBackend<R, F, I, BT>
{
    fn fused_time_mix_forward(
        state_in: Tensor<Self, 4>,
        r: Tensor<Self, 3>,
        w: Tensor<Self, 3>,
        k: Tensor<Self, 3>,
        v: Tensor<Self, 3>,
        a: Tensor<Self, 3>,
        b: Tensor<Self, 3>,
    ) -> TimeMixOutput<Self> {

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
        assert_eq!(r.dtype(), F::dtype());

        assert_eq!(w.dims()[0], d_batch);
        assert_eq!(w.dims()[1], d_tokens);
        assert_eq!(w.dims()[2], n_heads*d_value);
        assert_eq!(w.dtype(), DType::F32);

        assert_eq!(k.dims()[0], d_batch);
        assert_eq!(k.dims()[1], d_tokens);
        assert_eq!(k.dims()[2], n_heads*d_value);
        assert_eq!(k.dtype(), F::dtype());

        assert_eq!(v.dims()[0], d_batch);
        assert_eq!(v.dims()[1], d_tokens);
        assert_eq!(v.dims()[2], n_heads*d_value);
        assert_eq!(v.dtype(), F::dtype());

        assert_eq!(a.dims()[0], d_batch);
        assert_eq!(a.dims()[1], d_tokens);
        assert_eq!(a.dims()[2], n_heads*d_value);
        assert_eq!(a.dtype(), F::dtype());

        assert_eq!(b.dims()[0], d_batch);
        assert_eq!(b.dims()[1], d_tokens);
        assert_eq!(b.dims()[2], n_heads*d_value);
        assert_eq!(b.dtype(), F::dtype());

        let cube_dim = CubeDim { x: d_key as u32, y: 1, z: 1 };

        let state_in = state_in.into_primitive().tensor();
        let r = into_contiguous(r.into_primitive().tensor());
        let w = into_contiguous(w.into_primitive().tensor());
        let k = into_contiguous(k.into_primitive().tensor());
        let v = into_contiguous(v.into_primitive().tensor());
        let a = into_contiguous(a.into_primitive().tensor());
        let b = into_contiguous(b.into_primitive().tensor());

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

        let state_out = Tensor::from_primitive(TensorPrimitive::Float(state_out));
        let y_out = Tensor::from_primitive(TensorPrimitive::Float(y_out));

        assert_eq!(state_out.dims()[0], d_batch);
        assert_eq!(state_out.dims()[1], n_heads);
        assert_eq!(state_out.dims()[2], d_value);
        assert_eq!(state_out.dims()[3], d_key);
        assert_eq!(state_out.dtype(), DType::F32);
        
        
        assert_eq!(y_out.dims()[0], d_batch);
        assert_eq!(y_out.dims()[1], d_tokens);
        assert_eq!(y_out.dims()[2], n_heads*d_value);
        assert_eq!(y_out.dtype(), F::dtype());

        TimeMixOutput {state: state_out, y: y_out}
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
        let x5 = self.ffn.unfused_forward(x4.clone(), s.channel_mixer_x_state);

        let x6 = x5 + x3;

        let new_s = LayerState {
            time_mixer_x_state: new_time_mixer_x_state,
            vk_state: new_vk_state,
            channel_mixer_x_state: new_channel_mixer_x_state
        };
        (x6, new_v0, new_s)
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

        let r = self.receptance.forward(x_receptance);
        let k = self.key.forward(x_key);
        let v = self.value.forward(x_value.clone());

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

        //print!("state in: {vk_state}");

        let TimeMixOutput{state, y} = B::fused_time_mix_forward(vk_state, r.clone(), log_neglog_of_decay, k.clone(), v.clone(), -deformed_key.clone(), deformed_key*iclr);
        //println!("y: {y}");
        //print!("state: {state}");
        //panic!();

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
        let out = self.output.forward(out);

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

        let logits = self.head.forward(x);

        (logits, new_channel_states)
    }
}