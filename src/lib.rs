#![recursion_limit = "256"]

use burn::nn::loss::CrossEntropyLoss;
use burn::prelude::{Backend, Device, Int, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;

pub mod sampling;
pub mod context_manager;
pub mod rwkv7;

#[cfg(test)]
mod test_accuracy;

pub trait RWKVLayerState: Clone {
    fn detach(self) -> Self;
}

pub trait RWKVForward<B: Backend> {
    type LayerState: RWKVLayerState;
    
    fn forward(&self, input: Tensor<B, 2, Int>, channel_states: Option<&[Self::LayerState]>) -> (Tensor<B, 3>, Vec<Self::LayerState>);

    fn get_loss(&self, input_tokens: Tensor<B, 1, Int>, input_state: Option<&[Self::LayerState]>, device: &Device<B>) -> Tensor<B, 1> {
        let num_tokens = input_tokens.shape().dims[0];
        let expected_values = input_tokens.clone().slice([1..num_tokens]);

        let (logits, _next_layer_state) = self.forward(input_tokens.clone().unsqueeze::<2>(), input_state);
        let testable_logits = logits.squeeze(0).slice([0..num_tokens-1]);

        CrossEntropyLoss::new(None, device).forward(testable_logits, expected_values.clone())
    }


    fn forward_from_vec(&self, inputs: &[u16], device: &Device<B>, channel_states: Option<&[Self::LayerState]>) -> (Tensor<B, 3>, Vec<Self::LayerState>) {
        self.forward(Tensor::<B, 1, Int>::from_ints(&inputs[..], device).unsqueeze(), channel_states)
    }

    fn forward_from_vec_and_greedy_sample(&self, inputs: &[u16], device: &Device<B>, channel_states: Option<&[Self::LayerState]>) -> (Tensor<B, 1, Int>, u16, Vec<Self::LayerState>) {
        let (logits, new_channel_states) = self.forward(Tensor::<B, 1, Int>::from_ints(&inputs[..], device).unsqueeze(), channel_states);
        let (new_token_tensor, new_token_value) = Self::do_greedy_sample(logits.slice([0..1, (inputs.len()-1)..inputs.len()]));
        (new_token_tensor, new_token_value, new_channel_states)
    }

    fn do_greedy_sample(logits: Tensor<B, 3>) -> (Tensor<B, 1, Int>, u16) {
        let logits_transformed: Tensor<B, 1> = logits.squeeze::<2>(0).squeeze(0);
        let output_token = softmax(logits_transformed, 0).argmax(0);
        let output_u16 = output_token.to_data().as_slice::<B::IntElem>().unwrap()[0].to_u16();
        (output_token, output_u16)
    }
}