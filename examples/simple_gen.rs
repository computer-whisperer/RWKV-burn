#![recursion_limit = "256"]

use std::io;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use burn::module::Module;
use burn::prelude::{Backend, Device};
use rwkv::rwkv7::{RWKV7Model, RWKV7Config};

use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use llm_samplers::prelude::{SampleFreqPresence, SampleGreedy, SampleTemperature, SamplerChain, SimpleSamplerResources};
use rand::prelude::StdRng;
use rand::SeedableRng;
use rwkv_tokenizer::WorldTokenizer;
use rwkv::context_manager::ContextManager;
use rwkv::RWKVForward;

fn main_inner<B>(device: Device<B>)
where
    B: Backend,
    RWKV7Model<B>: RWKVForward<B>
{

    let tokenizer = Arc::new(WorldTokenizer::new(None).unwrap());

    //let model_path = "/mnt/secondary/temp-latest-training-models/RWKV7-G1-1.5B-32%trained-20250319-ctx4k.pth";
    let model_path = Path::new("/mnt/secondary/temp-latest-training-models/RWKV7-G1-2.9B-32%trained-20250327-ctx4k.pth");
    println!("Loading model {}", model_path.file_stem().unwrap().to_str().unwrap());
    
    let mut sc = SamplerChain::new();
    sc += SampleFreqPresence::new(0.1, 0.1, 128);
    sc += SampleTemperature::new(1.0);
    sc.push_sampler(SampleGreedy::new());
    let mut sampler = rwkv::sampling::Sampler::LLMSamplers((sc,
                                                        SimpleSamplerResources::new(
                                                            Some(Box::new(StdRng::seed_from_u64(12345))),
                                                            Some(vec![])
                                                        )
    ));

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(model_path.to_path_buf().into(), &device).unwrap();
    let rwkv = RWKV7Model::<B>::new(RWKV7Config::from_record(&record), &device);
    let rwkv = rwkv.load_record(record);

    let mut context_manager = ContextManager::new(tokenizer.clone(), None, device.clone());
    let prompt = "User: How many cats will fit in your average school bus?\n\nAssistant: <think>\nAlright";
    context_manager.add_text(prompt).unwrap();

    print!("Processing prompt: \n{}", prompt);

    context_manager.rwkv_forward(&rwkv).unwrap();


    let mut tokens = vec![];
    let now = Instant::now();
    let mut token_buffer = vec![];
    for _ in 0..2000 {
        //let token = context_manager.greedy_sample().unwrap();
        context_manager.sample(&mut sampler).unwrap();
        let token = context_manager.get_latest_token().unwrap();
            token_buffer.push(token);
        tokens.push(token);

        if let Ok(s) = tokenizer.decode(token_buffer.clone()) {
            token_buffer.clear();
            print!("{s}");
            io::stdout().flush().unwrap();
        }
        context_manager.rwkv_forward(&rwkv).unwrap();
    }

    let elapsed = now.elapsed().as_secs_f32();
    println!(
        "\n{} tokens processed ({:.4} tokens/s)\n",
        tokens.len(),
        tokens.len() as f32 / elapsed
    );

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60.0) as u32,
        (elapsed % 60.0) as u32
    );

}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;
        main_inner::<Wgpu<f32, i32>>(device);
    }
}


#[cfg(feature = "hip")]
mod hip {
    use super::*;
    use burn::backend::hip::{Hip, HipDevice};

    pub fn run() {
        let device = HipDevice{index: 0};
        main_inner::<Hip<f32, i32>>(device);
    }
}

#[cfg(feature = "candle")]
mod candle {
    use super::*;
    use burn::backend::candle::{Candle, CandleDevice};

    pub fn run() {
        let device = CandleDevice::default();
        main_inner::<Candle>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};

    pub fn run() {
        let device = CudaDevice::default();
        main_inner::<Cuda<f32, i32>>(device);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use burn::backend::{Vulkan};
    use burn::backend::wgpu::{WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;
        main_inner::<Vulkan<f32, i32>>(device);
    }
}

#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        main_inner::<NdArray>(device);
    }
}


#[allow(unreachable_code)]
pub fn main() {
    #[cfg(feature = "cuda")]
    {
        cuda::run();
        return;
    }
    #[cfg(feature = "vulkan")]
    {
        vulkan::run();
        return
    }
    #[cfg(feature = "wgpu")]
    {
        wgpu::run();
        return
    }
    #[cfg(feature = "hip")]
    {
        hip::run();
        return;
    }
    #[cfg(feature = "candle")]
    {
        candle::run();
        return;
    }

    #[cfg(feature = "ndarray")]
    {
        ndarray::run();
        return
    }
}