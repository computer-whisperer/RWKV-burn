use burn::tensor::{backend::Backend, Int, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use llm_samplers::prelude::{HasSamplerResources, Logits, Sampler as LLMSampler, SamplerChain, SimpleSamplerResources};
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};

pub enum Sampler {
    TopP(TopP),
    Argmax,
    LLMSamplers((SamplerChain, SimpleSamplerResources))
}

impl Sampler {
    pub fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        match self {
            Self::TopP(s) => s.sample(logits),
            Self::Argmax => logits.argmax(1),
            Self::LLMSamplers((sc, sc_res)) => {
                let mut samplers_logits = Logits::try_from_iter(logits.clone().squeeze::<1>(0).to_data().iter::<f32>()).unwrap();

                let token_id = sc.sample_token(sc_res, &mut samplers_logits).unwrap().unwrap();

                sc_res.with_last_tokens_mut(&mut |tokens| tokens.push(token_id)).unwrap();

                Tensor::<B, 1, Int>::from_ints([token_id as u16], &logits.device()).unsqueeze()
            },
        }
    }

    pub fn rwkv_sample<B: Backend>(&mut self, logits: Tensor<B, 3>) -> (Tensor<B, 1, Int>, u16) {
        let output_token: Tensor<B, 1, Int> = self.sample(softmax(logits.squeeze(1), 1)).squeeze(0);
        let output_u16 = output_token.to_data().as_slice::<B::IntElem>().unwrap()[0].to_u16();
        (output_token, output_u16)
    }

    pub fn rwkv_sample_single<B: Backend>(&mut self, logits: Tensor<B, 1>) -> (Tensor<B, 1, Int>, u16) {
        let output_token: Tensor<B, 1, Int> = self.sample(softmax(logits, 0).unsqueeze()).squeeze(0);
        let output_u16 = output_token.to_data().as_slice::<B::IntElem>().unwrap()[0].to_u16();
        (output_token, output_u16)
    }
}

pub trait Sampling {
    fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int>;
}

/// Top-p sampling (nucleus sampling) selects the smallest set of tokens whose cumulative
/// probability mass exceed the threshold p.
pub struct TopP {
    /// Probability threshold for sampling.
    p: f64,
    /// RNG.
    rng: StdRng,
}

impl TopP {
    pub fn new(p: f64, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { p, rng }
    }
}

impl Sampling for TopP {
    fn sample<B: Backend>(&mut self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        assert_eq!(
            probs.dims()[0],
            1,
            "Naive top-p sampling only supports single-batch tensors"
        );
        let (probs_sort, probs_idx) = probs.sort_descending_with_indices(1);

        // TODO: cumsum + Distribution::Multinomial support

        let mut probs_sort = probs_sort.to_data().iter::<f64>().collect::<Vec<_>>();

        let mut cumsum = 0.;
        probs_sort.iter_mut().for_each(|x| {
            if cumsum >= self.p {
                *x = 0.0;
            } else {
                cumsum += *x;
            }
        });

        let next_token_idx = WeightedIndex::new(probs_sort)
            .unwrap()
            .sample(&mut self.rng);

        probs_idx.slice([0..1, next_token_idx..next_token_idx + 1])
    }
}
