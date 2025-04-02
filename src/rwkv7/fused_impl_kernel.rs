use cubecl::{cube, prelude::*, CubeType};

#[derive(CubeType, Clone, Hash, Eq, PartialEq, Debug)]
pub(crate) struct TimeMixKernelConfig {
    pub n_heads: u32,
    pub d_key_value: u32
}

#[cube(launch)]
pub fn fused_time_mix_forward<F: Float>(
    state_in: &Tensor<f32>,
    state_out: &mut Tensor<f32>,
    r_in: &Tensor<F>,
    w_in: &Tensor<f32>,
    //w_in: &Tensor<F>,
    k_in: &Tensor<F>,
    v_in: &Tensor<F>,
    a_in: &Tensor<F>,
    b_in: &Tensor<F>,
    y_out: &mut Tensor<F>,
    #[comptime] config: TimeMixKernelConfig
) {
    let num_tokens = w_in.shape(1);
    let hh = CUBE_POS_X;
    let bb = CUBE_POS_Y;
    let i = UNIT_POS_X;

    let mut state = Array::<f32>::new(config.d_key_value);
    let s_idx = bb*config.n_heads *config.d_key_value *config.d_key_value + hh*config.d_key_value *config.d_key_value + i*config.d_key_value;
    #[unroll]
    for j in 0..config.d_key_value {
        state[j] = state_in[s_idx + j]
    }

    let mut r = SharedMemory::<f32>::new(config.d_key_value);
    let mut w = SharedMemory::<f32>::new(config.d_key_value);
    let mut k = SharedMemory::<f32>::new(config.d_key_value);
    let mut a = SharedMemory::<f32>::new(config.d_key_value);
    let mut b = SharedMemory::<f32>::new(config.d_key_value);

    for t in 0..num_tokens {
        let ind = bb* num_tokens *config.n_heads*config.d_key_value + t*config.n_heads*config.d_key_value + hh*config.d_key_value + i;

        sync_units();
        r[i] = f32::cast_from(r_in[ind]);
        //w[i] = Exp::exp(-Exp::exp(f32::cast_from(w_in[ind])));
        w[i] = Exp::exp(-Exp::exp(w_in[ind]));
        k[i] = f32::cast_from(k_in[ind]);
        a[i] = f32::cast_from(-a_in[ind]);
        b[i] = f32::cast_from(a_in[ind]*b_in[ind]);
        sync_units();

        let mut sa = 0.0f32;
        #[unroll]
        for j in 0..config.d_key_value {
            sa += a[j] * state[j];
        }

        let v = f32::cast_from(v_in[ind]);
        let mut y = 0.0f32;
        #[unroll]
        for j in 0..config.d_key_value {
            state[j] = state[j] * w[j] + sa * b[j] + k[j] * v;
            y += state[j] * r[j];
        }
        y_out[ind] = F::cast_from(y);
    }

    #[unroll]
    for j in 0..config.d_key_value {
        state_out[s_idx + j] = state[j];
    }
    sync_units();
}