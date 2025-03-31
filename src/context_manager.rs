use std::str::Utf8Error;
use std::sync::Arc;
use burn::prelude::{Backend, Device, Int, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use rwkv_tokenizer::WorldTokenizer;
use crate::sampling::Sampler;
use crate::{RWKVForward, RWKVLayerState};

#[derive(Debug, Clone, Default)]
pub enum UnprocessedTokens<B: Backend> {
    #[default]
    None,
    Logit(Tensor<B, 1>),
    TokensLocal(Vec<u16>),
    TokensDevice(Tensor<B, 1, Int>),
    Text(String)
}

#[derive(Debug, Clone)]
pub enum ContextManagerError {
    MissingContextError,
    MissingLogitError,
    DecodeError(Utf8Error)
}

#[derive(Debug, Clone)]
pub struct ContextManager<B: Backend, R: RWKVForward<B>> {
    processed_tokens: Option<Tensor<B, 1, Int>>,
    unprocessed_tokens: UnprocessedTokens<B>,
    initial_layer_state: Option<Vec<R::LayerState>>,
    last_layer_state: Option<Vec<R::LayerState>>,
    tokenizer: Arc<WorldTokenizer>,
    decoded_text: String,
    device: Device<B>,
    num_decoded_tokens: usize
}

impl<B, R: RWKVForward<B>> ContextManager<B, R> 
where
    B: Backend,
{
    pub fn new(tokenizer: Arc<WorldTokenizer>, initial_layer_state: Option<Vec<R::LayerState>>, device: Device<B>) -> Self {
        Self {
            processed_tokens: None,
            unprocessed_tokens: UnprocessedTokens::None,
            initial_layer_state,
            last_layer_state: None,
            tokenizer,
            decoded_text: String::new(),
            device,
            num_decoded_tokens: 0
        }
    }

    pub fn new_from_previous_context(previous_context: &Self) -> Self {
        Self {
            processed_tokens: None,
            unprocessed_tokens: previous_context.unprocessed_tokens.clone(),
            initial_layer_state: previous_context.last_layer_state.clone(),
            last_layer_state: previous_context.last_layer_state.clone(),
            tokenizer: previous_context.tokenizer.clone(),
            decoded_text: String::new(),
            device: previous_context.device.clone(),
            num_decoded_tokens: 0
        }
    }
    
    pub fn get_tokens(&self) -> Vec<u16> {
        match &self.processed_tokens {
            None => {Vec::new()}
            Some(x) => {x.to_data().iter::<u16>().collect()}
        }
    }
    
    pub fn get_initial_layer_state(&self) -> Option<&[R::LayerState]> {
        self.initial_layer_state.as_deref()
    }

    pub fn add_new_context(&mut self, new_context: &Self) {
        self.processed_tokens = match self.processed_tokens.clone() {
            None => {
                new_context.processed_tokens.clone()
            }
            Some(self_tokens) => {
                match &new_context.processed_tokens {
                    None => {Some(self_tokens)}
                    Some(new_tokens) => {
                        Some(Tensor::cat(vec![self_tokens, new_tokens.clone()], 0))
                    }
                }
            }
        };
        self.unprocessed_tokens = new_context.unprocessed_tokens.clone();
        self.last_layer_state = new_context.last_layer_state.clone();
    }

    pub fn decode_processed_tokens(&mut self) -> Result<&str, Utf8Error> {
        match &self.processed_tokens {
            None => {Ok("")}
            Some(processed_tokens) => {
                let num_processed_tokens = processed_tokens.shape().dims[0];
                if num_processed_tokens > self.num_decoded_tokens {
                    let tokens_vec = self.get_tokens();
                    self.decoded_text = self.tokenizer.decode(tokens_vec)?;
                    self.num_decoded_tokens = num_processed_tokens;
                }
                Ok(&self.decoded_text)
            }
        }
    }

    pub fn decode_processed_and_unprocessed_tokens(&mut self) -> Result<String, ContextManagerError> {
        let processed_str = self.decode_processed_tokens().map_err(|x| ContextManagerError::DecodeError(x))?.to_string();
        let unprocessed_string = match &self.unprocessed_tokens {
            UnprocessedTokens::Logit(_logit) => {
                String::new()
            }
            UnprocessedTokens::TokensLocal(tokens) => {
                self.tokenizer.decode(tokens.clone()).map_err(|x| ContextManagerError::DecodeError(x))?
            }
            UnprocessedTokens::TokensDevice(tokens) => {
                let tokens = tokens.clone().to_data().iter::<u16>().collect();
                self.tokenizer.decode(tokens).map_err(|x| ContextManagerError::DecodeError(x))?
            }
            UnprocessedTokens::None => {
                String::new()
            }
            UnprocessedTokens::Text(text) => {
                text.clone()
            }
        };
        Ok(processed_str + &unprocessed_string)
    }

    pub fn get_last_layer_state(&self) -> Option<&Vec<R::LayerState>> {
        self.last_layer_state.as_ref()
    }

    pub fn add_tokens(&mut self, tokens: &[u16]) {
        self.unprocessed_tokens = match std::mem::take(&mut self.unprocessed_tokens) {
            UnprocessedTokens::Logit(_logit) => {
                // Unsampled logits get discarded
                UnprocessedTokens::TokensLocal(tokens.to_vec())
            }
            UnprocessedTokens::TokensLocal(mut unprocessed_tokens) => {
                unprocessed_tokens.extend(tokens);
                UnprocessedTokens::TokensLocal(unprocessed_tokens)
            }
            UnprocessedTokens::TokensDevice(tensor) => {
                let input: Tensor<B, 1, Int> = Tensor::from_ints(&tokens[..], &self.device);
                UnprocessedTokens::TokensDevice(Tensor::cat(vec![tensor, input], 0))
            }
            UnprocessedTokens::None => {
                UnprocessedTokens::TokensLocal(tokens.to_vec())
            }
            UnprocessedTokens::Text(text) => {
                let mut unprocessed_tokens = self.tokenizer.encode(&text);
                unprocessed_tokens.extend(tokens);
                UnprocessedTokens::TokensLocal(unprocessed_tokens)
            }
        };
    }

    pub fn get_score(&self, rwkv: &R, text: &str, device: &Device<B>) -> f32 {
        let tokens = self.tokenizer.encode(text);
        let input: Tensor<B, 1, Int> = Tensor::from_ints(&tokens[..], device);
        let last_layer_state = match &self.last_layer_state{
            None => None,
            Some(x) => Some(&x[..])
        };
        let (logits, _next_layer_state) = rwkv.forward(input.clone().unsqueeze(), last_layer_state);
        let mut values = vec![];
        let mut sum = 0f32;
        for i in 0..tokens.len() - 1 {
            let tid = tokens[i+1] as usize;
            let value = logits.clone().slice([0..1, i..i+1, tid..tid+1]).into_scalar().to_f32();
            values.push(value);
            sum += value;
        }
        sum/values.len() as f32
    }

    pub fn add_text(&mut self, text: &str) -> Result<(), Utf8Error> {
        self.unprocessed_tokens = match std::mem::take(&mut self.unprocessed_tokens){
            UnprocessedTokens::Logit(_) => {
                UnprocessedTokens::Text(text.to_string())
            }
            UnprocessedTokens::TokensLocal(existing_tokens) => {
                UnprocessedTokens::Text(self.tokenizer.decode(existing_tokens)? + text)
            }
            UnprocessedTokens::TokensDevice(existing_tokens) => {
                // This may be a source of logit mismatches in the future, but for now we will have to try it
                let new_tokens = self.tokenizer.encode(text);
                let new_tokens_tensor: Tensor<B, 1, Int> = Tensor::from_ints(&new_tokens[..], &self.device);
                UnprocessedTokens::TokensDevice(Tensor::cat(vec![existing_tokens, new_tokens_tensor], 0))
            }
            UnprocessedTokens::None => {
                UnprocessedTokens::Text(text.to_string())
            }
            UnprocessedTokens::Text(existing_text) => {
                UnprocessedTokens::Text(existing_text + text)
            }
        };
        Ok(())
    }

    pub fn rwkv_forward(&mut self, rwkv: &R) -> Result<(), ContextManagerError> {
        let (input, input_len) = match std::mem::take(&mut self.unprocessed_tokens) {
            UnprocessedTokens::Logit(_) => {return Err(ContextManagerError::MissingContextError)}
            UnprocessedTokens::TokensLocal(tokens) => {
                let input: Tensor<B, 1, Int> = Tensor::from_ints(&tokens[..], &self.device);
                (input, tokens.len())
            }
            UnprocessedTokens::TokensDevice(tokens) => {
                let len = tokens.shape().dims[0];
                (tokens, len)
            }
            UnprocessedTokens::None => {return Err(ContextManagerError::MissingContextError)}
            UnprocessedTokens::Text(text) => {
                let input_vec = self.tokenizer.encode(&text);
                let input: Tensor<B, 1, Int> = Tensor::from_ints(&input_vec[..], &self.device);
                (input, input_vec.len())
            }
        };
        
        let last_layer_state = match &self.last_layer_state{
            None => None,
            Some(x) => Some(&x[..])
        };
        let (logits, next_layer_state) = rwkv.forward(input.clone().unsqueeze(), last_layer_state);
        let new_next_layer_state: Vec<R::LayerState> = next_layer_state.into_iter().map(|x| x.detach()).collect();
        let logits = logits.slice([0..1, (input_len-1)..input_len]).detach();
        self.processed_tokens = match core::mem::take(&mut self.processed_tokens) {
            None => {Some(input)}
            Some(old_tokens) => {
                Some(Tensor::cat(vec![old_tokens, input], 0))
            }
        };
        self.unprocessed_tokens = UnprocessedTokens::Logit(logits.squeeze::<2>(0).squeeze(0));
        self.last_layer_state = Some(new_next_layer_state);
        Ok(())
    }

    pub fn rwkv_fused_forward(&mut self, rwkv: &R) -> Result<(), ContextManagerError> {
        let (input, input_len) = match std::mem::take(&mut self.unprocessed_tokens) {
            UnprocessedTokens::Logit(_) => {return Err(ContextManagerError::MissingContextError)}
            UnprocessedTokens::TokensLocal(tokens) => {
                let input: Tensor<B, 1, Int> = Tensor::from_ints(&tokens[..], &self.device);
                (input, tokens.len())
            }
            UnprocessedTokens::TokensDevice(tokens) => {
                let len = tokens.shape().dims[0];
                (tokens, len)
            }
            UnprocessedTokens::None => {return Err(ContextManagerError::MissingContextError)}
            UnprocessedTokens::Text(text) => {
                let input_vec = self.tokenizer.encode(&text);
                let input: Tensor<B, 1, Int> = Tensor::from_ints(&input_vec[..], &self.device);
                (input, input_vec.len())
            }
        };

        let last_layer_state = match &self.last_layer_state{
            None => None,
            Some(x) => Some(&x[..])
        };
        let (logits, next_layer_state) = rwkv.forward(input.clone().unsqueeze(), last_layer_state);
        let new_next_layer_state: Vec<R::LayerState> = next_layer_state.into_iter().map(|x| x.detach()).collect();
        let logits = logits.slice([0..1, (input_len-1)..input_len]).detach();
        self.processed_tokens = match core::mem::take(&mut self.processed_tokens) {
            None => {Some(input)}
            Some(old_tokens) => {
                Some(Tensor::cat(vec![old_tokens, input], 0))
            }
        };
        self.unprocessed_tokens = UnprocessedTokens::Logit(logits.squeeze::<2>(0).squeeze(0));
        self.last_layer_state = Some(new_next_layer_state);
        Ok(())
    }
    
    pub fn sample_forward(&mut self, rwkv: &R, num_tokens: usize, decode_and_print: bool) -> Result<String, ContextManagerError> {
        let do_pre_forward = match &self.unprocessed_tokens {
            UnprocessedTokens::None => {false}
            UnprocessedTokens::Logit(_) => {false}
            UnprocessedTokens::TokensLocal(_) => {true}
            UnprocessedTokens::TokensDevice(_) => {true}
            UnprocessedTokens::Text(_) => {true}
        };
        
        if do_pre_forward {
            self.rwkv_forward(rwkv)?
        }
        
        let tokens_already_decoded = match &self.processed_tokens {
            None => {0}
            Some(x) => {x.shape().dims[0]}
        };
        
        let mut token_buffer = vec![];
        for _ in 0..num_tokens {
            self.greedy_sample()?;
            if decode_and_print {
                token_buffer.push(self.get_latest_token().unwrap());

                if decode_and_print {
                    if let Ok(s) = self.tokenizer.decode(token_buffer.clone()) {
                        token_buffer.clear();
                        print!("{s}");
                    }
                }
            }

            self.rwkv_forward(rwkv)?;
        }
        let tokens = self.get_tokens();
        let new_tokens = tokens[tokens_already_decoded..].to_vec();
        self.tokenizer.decode(new_tokens).map_err(|x| ContextManagerError::DecodeError(x))
    }

    pub fn greedy_sample(&mut self) -> Result<(), ContextManagerError> {
        let token_tensor = match std::mem::take(&mut self.unprocessed_tokens) {
            UnprocessedTokens::Logit(logit) => {
                softmax(logit, 0).argmax(0)
            }
            UnprocessedTokens::None |
            UnprocessedTokens::TokensLocal(_) |
            UnprocessedTokens::TokensDevice(_) |
            UnprocessedTokens::Text(_) => {return Err(ContextManagerError::MissingLogitError)}
        };
        self.unprocessed_tokens = UnprocessedTokens::TokensDevice(token_tensor);
        Ok(())
    }

    pub fn sample(&mut self, sampler: &mut Sampler) -> Result<(), ContextManagerError> {
        let (_tensor, token) = match std::mem::take(&mut self.unprocessed_tokens) {
            UnprocessedTokens::Logit(logit) => {
                sampler.rwkv_sample_single(logit)
            }
            UnprocessedTokens::None|
            UnprocessedTokens::TokensLocal(_) |
            UnprocessedTokens::TokensDevice(_) |
            UnprocessedTokens::Text(_) => {return Err(ContextManagerError::MissingLogitError)}
        };
        self.unprocessed_tokens = UnprocessedTokens::TokensLocal(vec![token]);
        Ok(())
    }
    
    pub fn get_latest_token(&self) -> Option<u16> {
        match &self.unprocessed_tokens {
            UnprocessedTokens::None => {None}
            UnprocessedTokens::Logit(_) => {None}
            UnprocessedTokens::TokensLocal(tokens) => {tokens.last().map(|x| *x)}
            UnprocessedTokens::TokensDevice(tokens) => {
                let tokens: Vec<u16> = tokens.clone().to_data().iter::<u16>().collect();
                tokens.last().map(|x| *x)
            }
            UnprocessedTokens::Text(text) => {
                let tokens = self.tokenizer.encode(text);
                tokens.last().map(|x| *x)
            }
        }
    }
}