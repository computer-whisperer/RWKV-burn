use burn::nn::{Sigmoid};
use burn::prelude::{Backend, Tensor};
use burn::tensor::activation::{relu, sigmoid, softplus};
use burn::tensor::DType;
use crate::rwkv7::{Block, ChannelMixer, LayerState, TimeMixer, RWKV7Model, lerp, lora_forward_sigmoid, lora_forward_tanh, normalize, lora_forward};


impl <B: Backend> TimeMixer<B> {
    pub(crate) fn unfused_forward(&self, hidden_state_in: Tensor<B, 3>, v0: Option<Tensor<B, 3>>, time_mixer_x_state: Tensor<B, 2>, vk_state: Tensor<B, 4>, d_model: usize, n_heads: usize) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 4>) {
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
        let log_of_decay = -(log_neglog_of_decay.cast(DType::F32).exp());
        let decay = log_of_decay.exp();

        let deformed_key = k.clone()*self.k_k.val();
        let deformed_key = normalize(deformed_key.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]), 3, 2.0);

        let iclr = sigmoid(lora_forward(self.a1.val(), self.a2.val(), Some(self.a0.val()), x_iclr));

        //println!("gate: {gate:?}"); 

        let k = lerp(k.clone(), k.clone()*iclr.clone(), self.k_a.val());

        // Separate into heads
        let r: Tensor<B, 4> = r.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let k: Tensor<B, 4> = k.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let v: Tensor<B, 4> = v.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let decay: Tensor<B, 4> = decay.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let iclr: Tensor<B, 4> = iclr.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let deformed_key: Tensor<B, 4> = deformed_key.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);

        //println!("r: {r:?}");
        //println!("w: {decay:?}");
        //println!("k: {k:?}");
        //println!("v: {v:?}");

        let mut vk_state = vk_state;
        let mut outputs = Vec::new();
        for t in 0..d_tokens {
            let ranges = [None, Some((t as i64, ((t)+1) as i64))];
            let r = r.clone().slice(ranges).squeeze(1);
            let k = k.clone().slice(ranges).squeeze(1);
            let v = v.clone().slice(ranges).squeeze(1);
            let decay = decay.clone().slice(ranges).squeeze(1);
            let iclr = iclr.clone().slice(ranges).squeeze(1);
            let deformed_key = deformed_key.clone().slice(ranges).squeeze(1);
            let (new_out, new_vk_state) = self.unfused_single_timestep(r, k, v, decay, iclr, deformed_key, vk_state);
            vk_state = new_vk_state;
            outputs.push(new_out);
        }
        let out: Tensor<B, 4> = Tensor::stack(outputs, 1);
        let out = out.reshape([d_batch, d_tokens, d_model]);
        
        //println!("a: {out}");

        //println!("ln_x gamma: {:?}", self.ln_x.gamma.clone().unwrap().val());
        //println!("ln_x beta: {:?}", self.ln_x.beta.clone().unwrap().val());

        // apply group normalization to each head and recombine the heads
        let out = self.ln_x.forward(out.reshape([(d_batch*d_tokens) as i32, d_model as i32])).reshape([d_batch as i32, d_tokens as i32, d_model as i32]);

        //println!("b: {out:}");

        let bonus = (r*k*self.r_k.val().unsqueeze()).sum_dim(3) * v;

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

        (out, v0, vk_state)
    }

    fn unfused_single_timestep(&self,
                       r: Tensor<B, 3>,
                       k: Tensor<B, 3>,
                       v: Tensor<B, 3>,
                       decay: Tensor<B, 3>,
                       iclr: Tensor<B, 3>,
                       deformed_key: Tensor<B, 3>,
                       vk_state: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 4>) {

        let d_batch = vk_state.dims()[0];
        let d_heads = vk_state.dims()[1];
        let d_value = vk_state.dims()[2];
        let d_key = vk_state.dims()[3];

        let input_dtype = r.dtype();

        assert_eq!(r.dims()[0], d_batch);
        assert_eq!(r.dims()[1], d_heads);
        assert_eq!(r.dims()[2], d_value);

        assert_eq!(k.dims()[0], d_batch);
        assert_eq!(k.dims()[1], d_heads);
        assert_eq!(k.dims()[2], d_key);

        assert_eq!(v.dims()[0], d_batch);
        assert_eq!(v.dims()[1], d_heads);
        assert_eq!(v.dims()[2], d_value);

        assert_eq!(decay.dims()[0], d_batch);
        assert_eq!(decay.dims()[1], d_heads);
        assert_eq!(decay.dims()[2], d_value);

        assert_eq!(iclr.dims()[0], d_batch);
        assert_eq!(iclr.dims()[1], d_heads);
        assert_eq!(iclr.dims()[2], d_value);

        assert_eq!(deformed_key.dims()[0], d_batch);
        assert_eq!(deformed_key.dims()[1], d_heads);
        assert_eq!(deformed_key.dims()[2], d_key);

        // transform inputs from BHK into column vectors BHK1, and put everything in float format for higher precision

        let vk_state = vk_state.cast(DType::F32);

        let d = 3;
        let r = r.unsqueeze_dim(d).cast(DType::F32);
        let k = k.unsqueeze_dim(d).cast(DType::F32);
        let v = v.unsqueeze_dim(d).cast(DType::F32);
        let decay = decay.unsqueeze_dim(d);
        let iclr = iclr.unsqueeze_dim(d).cast(DType::F32);
        let deformed_key = deformed_key.unsqueeze_dim(d).cast(DType::F32);

        // decay the kv state and remove the iclr amount of the value stored within the state at the deformed key
        let t_decay = decay.transpose();
        let vk_state = vk_state.clone() * t_decay - vk_state.matmul(deformed_key.clone()).matmul((iclr*deformed_key).transpose());

        // add in an dynamically iclr and 1-decay mixed amount of the latest value at the key
        // (key has been pre-adjusted in the calling code by the amount of iclr mixing)
        let vk_state = vk_state + (v.matmul(k.transpose()));

        // Apply receptance to the new stat
        let out = vk_state.clone().matmul(r);

        // Remove extra useless dimension from the output
        (out.squeeze(3).cast(input_dtype), vk_state)
    }
}

impl<B: Backend> ChannelMixer<B> {
    pub(crate) fn unfused_forward(&self, hidden_state_in: Tensor<B, 3>, x_state: Tensor<B, 2>) -> Tensor<B, 3> {
        //let d_batch = hidden_state_in.shape().dims[0];
        let d_tokens = hidden_state_in.shape().dims[1];
        let x_shifted_one_to_the_past : Tensor<B, 3> = if d_tokens > 1 {
            Tensor::cat(vec![x_state.unsqueeze(), hidden_state_in.clone().slice([None, Some((0, -1))])], 1)
        } else {
            x_state.unsqueeze()
        };

        let x_in = lerp(hidden_state_in, x_shifted_one_to_the_past, self.x_k.val());
        let hidden = self.key.forward(x_in);
        let hidden = relu(hidden).powi_scalar(2);
        let out = self.value.forward(hidden);
        out
    }
}

impl<B: Backend> Block<B> {
    pub(crate) fn unfused_forward(&self, _layer_num: usize, x: Tensor<B, 3>, v0: Option<Tensor<B, 3>>, s: Option<&LayerState<B>>, d_model: usize, n_heads: usize) -> (Tensor<B, 3>, Tensor<B, 3>, LayerState<B>) {
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
        let (x2, new_v0, new_vk_state) = self.att.unfused_forward(x1.clone(), v0, s.time_mixer_x_state, s.vk_state, d_model, n_heads);
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

impl<B: Backend> RWKV7Model<B> {
    pub fn unfused_forward(&self, input: Tensor<B, 2, burn::prelude::Int>, channel_states: Option<&[LayerState<B>]>) -> (Tensor<B, 3>, Vec<LayerState<B>>) {
        let mut x = self.emb.forward(input);

        //println!("after emb: {x:?}");

        let mut v0 = None;
        let mut new_channel_states = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let channel_state = if let Some(s) = channel_states{
                Some(&s[i])
            } else {
                None
            };
            let (new_x, new_v0, new_channel_state) = block.unfused_forward(i, x, v0, channel_state, self.d_model, self.n_heads);
            x = new_x;
            //println!("after block {i}: {x:?}");

            if i > 0 {
                //panic!()
            }
            v0 = Some(new_v0);
            new_channel_states.push(new_channel_state);
        }

        let x = self.ln_out.forward(x);
        //println!("after ln_out: {x:?}");


        let logits = self.head.forward(x);

        (logits, new_channel_states)
    }
}