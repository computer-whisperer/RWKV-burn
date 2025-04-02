#![recursion_limit = "256"]

use std::env;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use burn::module::Module;
use burn::prelude::{Backend, Device};
use burn_cubecl::FloatElement;
use rwkv_burn::rwkv7::{RWKV7Model, RWKV7Config, RWKVForward};

use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use rwkv_tokenizer::WorldTokenizer;
use rwkv_burn::context_manager::ContextManager;

fn main_inner<B>(device: Device<B>)
where
    B: Backend,
    RWKV7Model<B>: RWKVForward<B> {

    let tokenizer = Arc::new(WorldTokenizer::new(None).unwrap());

    let model_repo = Path::new("/mnt/secondary/");

    // Use other model repo if this one doesn't exit
    let model_repo = if model_repo.exists() {
        model_repo
    } else {
        Path::new("/ceph-fuse/public/neural_models/llms/")
    };
    
    let model_path = model_repo.join("temp-latest-training-models/RWKV7-G1-1.5B-50%trained-20250330-ctx4k.pth");
    println!("Loading model {}", model_path.file_stem().unwrap().to_str().unwrap());
    //let model_path = "/mnt/secondary/temp-latest-training-models/RWKV7-G1-2.9B-16%trained-20250313-ctx4k.pth";
    let load_start = Instant::now();
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(model_path.into(), &device).unwrap();
    let rwkv = RWKV7Model::<B>::new(RWKV7Config::from_record(&record), &device);
    let rwkv = rwkv.load_record(record);
    println!("Loaded {:?} in {:?}", rwkv.get_main_dtype(), Instant::now() - load_start);

    let mut context_manager = ContextManager::new(tokenizer.clone(), None, device.clone());
    let prompt = "User: How many cats will fit in your average school bus?\n\nAssistant: <think>\nAlright";
    context_manager.add_text(prompt).unwrap();

    println!("Processing prompt: \n{}", prompt);

    context_manager.rwkv_forward(&rwkv).unwrap();
    // Warm up
    for _ in 0..100 {
        context_manager.greedy_sample().unwrap();
        context_manager.rwkv_forward(&rwkv).unwrap();
    }
    B::sync(&device);
    println!("Warm up complete.");
    let mut context_manager = ContextManager::new(tokenizer.clone(), None, device.clone());
    if true {
        let prefill_tokens = [100; 5000];
        let prefill_start = Instant::now();
        context_manager.add_tokens(&prefill_tokens);
        context_manager.rwkv_forward(&rwkv).unwrap();
        B::sync(&device);
        println!("Prefill complete: {}tps", prefill_tokens.len() as f32/(Instant::now() - prefill_start).as_secs_f32());
    }

    let forward_start = Instant::now();
    let n_forward = 1000;
    for _ in 0..n_forward {
        //context_manager.greedy_sample().unwrap();
        context_manager.add_tokens(&[100]);
        context_manager.rwkv_forward(&rwkv).unwrap();
    }
    B::sync(&device);

    println!("Forward complete: {}tps", n_forward as f32/(Instant::now() - forward_start).as_secs_f32());

}


#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;
        main_inner::<Wgpu>(device);
    }
}


#[cfg(feature = "hip")]
mod hip {
    use super::*;
    use burn::backend::hip::{Hip, HipDevice};

    pub fn run<F: FloatElement>() {
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

    pub fn run<F: FloatElement>() {
        let device = CudaDevice::default();
        main_inner::<Cuda<F, i32>>(device);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use burn::backend::{Vulkan};
    use burn::backend::wgpu::{WgpuDevice};

    pub fn run<F: FloatElement>() {
        let device = WgpuDevice::DefaultDevice;
        main_inner::<Vulkan<F, i32>>(device);
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
    let args: Vec<String> = env::args().collect();
    
    let use_bf16 = args.contains(&"--bf16".to_string());
    let use_fp16 = args.contains(&"--fp16".to_string());
    

    #[cfg(feature = "cuda")]
    {
        if use_bf16 {
            cuda::run::<half::bf16>();
        } else if use_fp16 {
            cuda::run::<half::f16>();
        } else {
            cuda::run::<f32>();
        }
        return;
    }
    #[cfg(feature = "vulkan")]
    {
        if use_bf16 {
            vulkan::run::<half::bf16>();
        } else if use_fp16 {
            vulkan::run::<half::f16>();
        } else {
            vulkan::run::<f32>();
        }
        return
    }
    #[cfg(feature = "wgpu")]
    {
        wgpu::run();
        return
    }
    #[cfg(feature = "hip")]
    {
        if use_bf16 {
            hip::run::<half::bf16>();
        } else if use_fp16 {
            hip::run::<half::f16>();
        } else {
            hip::run::<f32>();
        }
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